#include "ray-predictor.h"
#include "../../libcuda/gpgpu_context.h"
#include "shader.h"

ray_predictor::ray_predictor(unsigned sid, ray_predictor_config config, shader_core_ctx *core ) {
                      
  m_core = core;
  
  m_predictor_table = {};
  m_busy = false;
  
  m_go_up_level = config.go_up_level;
  m_number_of_entries_cap = config.entry_cap;
  m_replacement_policy = config.replacement_policy;
  m_placement_policy = config.placement_policy;
  m_ways = config.n_ways;
  m_table_size = config.max_size;
  m_cycle_delay = config.latency;
  m_virtualize = config.virtualize;
  m_virtualize_delay = config.virtualize_delay;
  m_node_replacement_policy = config.entry_replacement_policy;
  
  m_sid = sid;
  
  num_rays = 0;
  num_predicted = 0;
  num_miss = 0;
  num_valid = 0;
  num_evicted = 0;
  num_entry_overflow = 0;
  num_virtual_predicted = 0;
  num_virtual_miss = 0;
  num_virtual_valid = 0;
  mem_access_saved = 0;
}
                    
                    
warp_inst_t ray_predictor::lookup(const warp_inst_t& inst) {
  if (inst.empty()) {
    // Not a valid warp, return previous warp
    warp_inst_t prev_warp = m_current_warp;
    // Clear current warp
    m_current_warp.clear();
    return prev_warp;
  }
  
  // Mark predictor busy
  m_busy = true;
  reset_cycle_delay(m_cycle_delay);
  warp_inst_t prev_warp = m_current_warp;
  m_current_warp = inst;
  
  // Iterate through every thread
  unsigned warp_size = inst.warp_size();
  num_rays += warp_size;
  for (unsigned i=0; i<warp_size; i++) {
    unsigned long long ray_hash = m_current_warp.rt_ray_hash(i);
    
    // Check if valid ray
    if (ray_hash == 0) continue;
    
    unsigned long long index;
    // Check predictor table
    if (check_table(ray_hash, index)) {
      // If predictor hit, iterate through predictions
      num_predicted++;
      
      // Read list of predictions from predictor table
      std::deque<new_addr_type> &prediction_list = m_predictor_table[index].m_nodes;
      // Validate prediction
      bool valid = validate_prediction(prediction_list, m_current_warp.rt_ray_properties(i), i);
      if (valid) {
        num_valid++;
      }
      
      // If invalid, regular traversal
      else {
        // Add to table if ray intersects with something
        if (m_current_warp.rt_ray_intersect(i)) {
          new_addr_type predict_node = m_current_warp.rt_ray_prediction(i);
          add_entry(ray_hash, predict_node);
          
          if (m_virtualize) {
            m_core->get_cluster()->add_ray_predictor_entry(ray_hash, predict_node);
          }
        }
      }
    }
    
    // Regular traversal.
    else {
      num_miss++;
      
      if (m_virtualize) {
        // Check second level table
        reset_cycle_delay(m_cycle_delay + m_virtualize_delay);
        predictor_entry virtual_entry = m_core->get_cluster()->check_ray_predictor_table(ray_hash);
        bool valid;
        
        // If hit in virtual table, validate
        if (virtual_entry.m_valid) {
          num_virtual_predicted++;
          valid = validate_prediction(virtual_entry.m_nodes, m_current_warp.rt_ray_properties(i), i);
          if (valid) num_virtual_valid++;
        }
        
        // If miss in table or invalid predictions, perform regular traversal
        if (!virtual_entry.m_valid || !valid) {
          if (!virtual_entry.m_valid) num_virtual_miss++;
          if (m_current_warp.rt_ray_intersect(i)) {
            new_addr_type predict_node = m_current_warp.rt_ray_prediction(i);
            m_core->get_cluster()->add_ray_predictor_entry(ray_hash, predict_node);
          }
        }
      }
      
      // Add to table if ray intersects with something
      if (m_current_warp.rt_ray_intersect(i)) {
        new_addr_type predict_node = m_current_warp.rt_ray_prediction(i);
        add_entry(ray_hash, predict_node);
      }
    }
  }
  return prev_warp;
  
}

bool ray_predictor::check_table(unsigned long long hash, unsigned long long &index) {
  // Direct mapped predictor
  if (m_placement_policy == 'd') {
    index = hash & (m_table_size - 1);
    if (m_predictor_table.find(index) != m_predictor_table.end())
      return (m_predictor_table[index].m_tag == hash && m_predictor_table[index].m_valid);
    else
      return false;
  }
  
  // Fully associative predictor
  else if (m_placement_policy == 'a') {
    index = hash;
    bool hit = (m_predictor_table.find(hash) != m_predictor_table.end());
    
    if (hit) assert(m_predictor_table[hash].m_valid);
    
    // Update LRU
    if (hit && m_placement_policy == 'a' && m_replacement_policy == 'l') {
      m_predictor_table[hash].m_timestamp = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
    }
    
    return hit;
  }
  
  // Set associative predictor
  else if (m_placement_policy == 's') {
    unsigned i = hash & (m_table_size/m_ways - 1);
    
    // Check each way
    for (unsigned way=0; way<m_ways; way++) {
      if (m_predictor_table[i + way*m_table_size/m_ways].m_valid && m_predictor_table[i + way*m_table_size/m_ways].m_tag == hash) {
        // Update LRU
        if (m_replacement_policy == 'l') {
          m_predictor_table[i + way*m_table_size/m_ways].m_timestamp = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
        }
        // Update index
        index = i + way*m_table_size/m_ways;
        return true;
      }
    }
    // Otherwise miss
    return false;
  }
  
  assert(0);
}

void ray_predictor::add_entry(unsigned long long hash, new_addr_type predict_node) {
  
  // Direct mapped
  if (m_placement_policy == 'd') {
    // Check if hash is already in the predictor table
    unsigned index = hash & (m_table_size - 1);
    if (m_predictor_table.find(index) != m_predictor_table.end()) {
      if (m_predictor_table[index].m_tag == hash) {
        assert(std::find(m_predictor_table[index].m_nodes.begin(), m_predictor_table[index].m_nodes.end(), predict_node) == m_predictor_table[index].m_nodes.end());
        if (m_predictor_table[index].m_nodes.size() < m_number_of_entries_cap) {
          m_predictor_table[index].m_nodes.push_back(predict_node);
        }
        else {
          // If FIFO, remove oldest node, add new node
          if (m_node_replacement_policy == 'f') {
            m_predictor_table[index].m_nodes.pop_front();
            m_predictor_table[index].m_nodes.push_back(predict_node);
          }
          // Otherwise, leave as is
          
          num_entry_overflow++;
        }  
        return;
      }
      else {
        num_evicted++;
      }
    }
  }
  // Fully associative
  else if (m_placement_policy == 'a') {
    // Check if hash is already in the predictor table
    if (m_predictor_table.find(hash) != m_predictor_table.end()) {
        assert(std::find(m_predictor_table[hash].m_nodes.begin(), m_predictor_table[hash].m_nodes.end(), predict_node) == m_predictor_table[hash].m_nodes.end());
      if (m_predictor_table[hash].m_nodes.size() < m_number_of_entries_cap) {
        m_predictor_table[hash].m_nodes.push_back(predict_node);
      }
      else {
        if (m_node_replacement_policy == 'f') {
          m_predictor_table[hash].m_nodes.pop_front();
          m_predictor_table[hash].m_nodes.push_back(predict_node);
        }
        num_entry_overflow++;
      }
      m_predictor_table[hash].m_timestamp = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
      return;
    }
  }
  // Set Associative
  else if (m_placement_policy == 's') {
    // Check if hash is already in the predictor table
    unsigned index = hash & (m_table_size/m_ways - 1);
    for (unsigned way=0; way<m_ways; way++) {
      if (m_predictor_table[index + way*m_table_size/m_ways].m_tag == hash && m_predictor_table[index + way*m_table_size/m_ways].m_valid) {
        if (m_predictor_table[index + way*m_table_size/m_ways].m_nodes.size() < m_number_of_entries_cap) {
          // There shouldn't be any duplicate "predict_node" because if it was in the table, the ray would have been predicted and verified. 
          assert(std::find(m_predictor_table[index + way*m_table_size/m_ways].m_nodes.begin(), m_predictor_table[index + way*m_table_size/m_ways].m_nodes.end(), predict_node) == m_predictor_table[index + way*m_table_size/m_ways].m_nodes.end());
          m_predictor_table[index + way*m_table_size/m_ways].m_nodes.push_back(predict_node);
        }
        else {
          if (m_node_replacement_policy == 'f') {
            m_predictor_table[index + way*m_table_size/m_ways].m_nodes.pop_front();
            m_predictor_table[index + way*m_table_size/m_ways].m_nodes.push_back(predict_node);
          }
          num_entry_overflow++;
        }
        m_predictor_table[index + way*m_table_size/m_ways].m_timestamp = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
        return;
      }
    }
  }
  
  // Otherwise, create new entry
  predictor_entry new_entry;
  new_entry.m_tag = hash;
  new_entry.m_timestamp = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
  new_entry.m_valid = true;
  new_entry.m_nodes.push_back(predict_node);
  
  if (m_placement_policy == 'd') {
    unsigned index = hash & (m_table_size - 1);
    m_predictor_table[index] = new_entry;
  }
  else if (m_placement_policy == 'a') {
    // Check if table is full 
    if (m_predictor_table.size() >= m_table_size) {
      evict_entry();
    }
    m_predictor_table[hash] = new_entry;
  }
  else if (m_placement_policy == 's') {
    unsigned index = hash & (m_table_size/m_ways - 1);
    // Choose way
    unsigned w;
    unsigned long long lru = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle + 1;
    bool evict = true;
    for (unsigned way=0; way<m_ways; way++) {
      if (!m_predictor_table[index + way*m_table_size/m_ways].m_valid) {
        evict = false;
        w = way;
        break;
      }
      else if (m_predictor_table[index + way*m_table_size/m_ways].m_timestamp < lru) {
        w = way;
        lru = m_predictor_table[index + way*m_table_size/m_ways].m_timestamp;
      }
    }
    
    if (evict) num_evicted++;
    m_predictor_table[index + w*m_table_size/m_ways] = new_entry;
  }
  else {
    assert(0);
  }
  
  assert(m_predictor_table.size() <= m_table_size);
}

void ray_predictor::evict_entry() {
  // LRU
  if (m_replacement_policy == 'l') {
    unsigned long long lru = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle + 1;
    unsigned long long evicted_hash;
    for (auto it=m_predictor_table.begin(); it!=m_predictor_table.end(); ++it) {
      if (it->second.m_timestamp < lru) {
        evicted_hash = it->first;
        lru = it->second.m_timestamp;
      }
    }
    assert(lru != (GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle + 1));
    m_predictor_table.erase(evicted_hash);
  }
  num_evicted++;
}

void ray_predictor::cycle() {
  if (m_cycles == 0) m_busy = false;
  else if (m_cycles > 0) m_cycles--;
}

void ray_predictor::display_state(FILE* fout) {
  m_current_warp.print(fout);
}

void ray_predictor::print_stats(FILE* fout) {
  fprintf(fout, "Shader Core %d Predictor Stats:\n", m_sid);
  fprintf(fout, "Total ray predictor hits: %d (%.3f) + virtual: %d\n", 
          num_predicted, (float)num_predicted / num_rays, num_virtual_predicted);
  fprintf(fout, "Total ray predictor misses: %d (%.3f) + virtual: %d\n", 
          num_miss, (float)num_miss / num_rays, num_virtual_miss);
  fprintf(fout, "Total number of valid predictions: %d (%.3f) + virtual: %d\n", 
          num_valid, (float)num_valid / num_rays, num_virtual_valid);
  fprintf(fout, "Total memory access savings: %d\n", mem_access_saved);
  fprintf(fout, "Evicted entries: %d\n", num_evicted);
  fprintf(fout, "Per entry overflow: %d\n", num_entry_overflow);
  fprintf(fout, "--------------------------------------------\n");
}

bool ray_predictor::validate_prediction(const std::deque<new_addr_type> prediction_list, const Ray ray_properties, unsigned tid) {
  bool valid;
  
  // Save list of memory requests required to validate prediction
  std::deque<new_addr_type> predictor_mem_accesses;
  
  // Iterate through predictions
  for (auto it=prediction_list.begin(); it!=prediction_list.end(); ++it) {
    valid = traverse_intersect(*it, ray_properties, predictor_mem_accesses);
    // If intersected, the prediction is valid
    if (valid) {
      mem_access_saved += m_current_warp.update_rt_mem_accesses(tid, valid, predictor_mem_accesses);
      return true;
    }
  }
  
  assert(!valid);
  mem_access_saved += m_current_warp.update_rt_mem_accesses(tid, valid, predictor_mem_accesses);
  return valid;
}

bool ray_predictor::traverse_intersect(const new_addr_type prediction, const Ray ray_properties, std::deque<new_addr_type> &mem_accesses) {
  
  memory_space *mem = GPGPU_Context()->the_gpgpusim->g_the_gpu->get_global_memory();
  new_addr_type tri_addr = prediction;
  float thit = ray_properties.get_tmax();

  // while triangle address is within triangle primitive range
  while (1) {
      
    // Matches rtao Woopify triangle
    float4 p0, p1, p2;
    mem->read(tri_addr, sizeof(float4), &p0);
    mem->read(tri_addr + sizeof(float4), sizeof(float4), &p1);
    mem->read(tri_addr + 2*sizeof(float4), sizeof(float4), &p2);
      
    // Check if triangle is valid (if (__float_as_int(v00.x) == 0x80000000))
    if (*(int*)&p0.x ==  0x80000000) {
      return false;
    }
    mem_accesses.push_back(tri_addr);
      
    bool hit = rtao_ray_triangle_test(p0, p1, p2, ray_properties, &thit);
    if (hit) {
      assert(ray_properties.anyhit);
      return true;
    }
      
          
    // Go to next triangle
    tri_addr += 0x30;

  }
}