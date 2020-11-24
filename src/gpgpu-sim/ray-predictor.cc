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
      std::vector<new_addr_type> prediction_list = get_prediction_list(index);
      // Validate prediction
      new_addr_type hit_node;
      bool valid =
        validate_prediction(prediction_list, m_current_warp.rt_ray_properties(i), i, hit_node);
      if (valid) {
        update_entry_use(index, hit_node);
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
          std::vector<new_addr_type> prediction_list(
            virtual_entry.m_nodes.begin(), virtual_entry.m_nodes.end());

          // Sort into insertion order
          // TODO: This should be sorted based on the node replacement policy
          std::sort(prediction_list.begin(), prediction_list.end(),
            [&](new_addr_type a, new_addr_type b) {
              return virtual_entry.m_node_use_map.at(a) < virtual_entry.m_node_use_map.at(b);
            });

          new_addr_type hit_node;
          valid = validate_prediction(prediction_list, m_current_warp.rt_ray_properties(i), i, hit_node);
          // TODO: This should update the node_use_map
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
  unsigned long long cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle +
    GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;

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
      m_predictor_table[hash].m_timestamp = cycle;
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
          m_predictor_table[i + way*m_table_size/m_ways].m_timestamp = cycle;
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

void ray_predictor::add_node_to_predictor_entry(unsigned long long index, new_addr_type node) {
  predictor_entry& entry = m_predictor_table[index];
  unsigned long long cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle +
    GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;

  // Entry full: need to evict a node
  if (entry.m_nodes.size() >= m_number_of_entries_cap) {
    num_entry_overflow++;

    new_addr_type node_to_evict = 0;
    unsigned long long min_use = 0;

    switch (m_node_replacement_policy) {
      // No replacement: Just don't add any more
      case 'n':
        return;
      // FIFO: Evict the first (with oldest/smallest timestamp) node
      case 'f':
      // LRU: Evict the least recently used (with oldest/smallest timestamp) node
      case 'l':
        min_use = cycle + 1;
        break;
      // Least used: Evict the least used (smallest used element) node
      case 'u':
        min_use = ULLONG_MAX;
        break;
      default:
        assert(false);
    }

    for (const auto& node_use : entry.m_node_use_map) {
      if (node_use.second < min_use) {
        node_to_evict = node_use.first;
        min_use = node_use.second;
      }
    }

    assert(entry.m_nodes.find(node_to_evict) != entry.m_nodes.end());
    assert(entry.m_node_use_map.find(node_to_evict) != entry.m_node_use_map.end());
    entry.m_nodes.erase(node_to_evict);
    entry.m_node_use_map.erase(node_to_evict);
  }

  assert(entry.m_nodes.find(node) == entry.m_nodes.end());
  entry.m_nodes.insert(node);

  switch (m_node_replacement_policy) {
    case 'n':
    case 'f':
    case 'l':
      entry.m_node_use_map[node] = cycle;
      break;
    case 'u':
      entry.m_node_use_map[node] = 0;
      break;
    default:
      assert(false);
  }
}

std::vector<new_addr_type> ray_predictor::get_prediction_list(unsigned long long index) const {
  const predictor_entry& entry = m_predictor_table.at(index);
  std::vector<new_addr_type> nodes(entry.m_nodes.begin(), entry.m_nodes.end());

  switch (m_node_replacement_policy) {
    // Sort into insertion order
    case 'n':
    case 'f':
      std::sort(nodes.begin(), nodes.end(), [&](new_addr_type a, new_addr_type b) {
        return entry.m_node_use_map.at(a) < entry.m_node_use_map.at(b);
      });
      break;
    // LRU: sort most recently used first
    // Least used: sort most used first
    case 'l':
    case 'u':
      std::sort(nodes.begin(), nodes.end(), [&](new_addr_type a, new_addr_type b) {
        return entry.m_node_use_map.at(a) > entry.m_node_use_map.at(b);
      });
      break;
    default:
      assert(false);
  }

  return nodes;
}

void ray_predictor::update_entry_use(unsigned long long index, new_addr_type node) {
  predictor_entry& entry = m_predictor_table[index];
  unsigned long long cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle +
    GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;

  switch (m_node_replacement_policy) {
    case 'n':
    case 'f':
      // Do nothing
      break;
    case 'l':
      assert(entry.m_node_use_map.find(node) != entry.m_node_use_map.end());
      entry.m_node_use_map[node] = cycle;
      break;
    case 'u':
      assert(entry.m_node_use_map.find(node) != entry.m_node_use_map.end());
      entry.m_node_use_map[node]++;
      break;
    default:
      assert(false);
  }
}

void ray_predictor::add_entry(unsigned long long hash, new_addr_type predict_node) {
  unsigned long long cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle +
    GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
  
  // Direct mapped
  if (m_placement_policy == 'd') {
    // Check if hash is already in the predictor table
    unsigned index = hash & (m_table_size - 1);
    if (m_predictor_table.find(index) != m_predictor_table.end()) {
      if (m_predictor_table[index].m_tag == hash) {
        add_node_to_predictor_entry(index, predict_node);
        m_predictor_table[index].m_timestamp = cycle;
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
      add_node_to_predictor_entry(hash, predict_node);
      m_predictor_table[hash].m_timestamp = cycle;
      return;
    }
  }
  // Set Associative
  else if (m_placement_policy == 's') {
    // Check if hash is already in the predictor table
    unsigned index = hash & (m_table_size/m_ways - 1);
    for (unsigned way=0; way<m_ways; way++) {
      if (m_predictor_table[index + way*m_table_size/m_ways].m_tag == hash && m_predictor_table[index + way*m_table_size/m_ways].m_valid) {
        add_node_to_predictor_entry(index + way*m_table_size/m_ways, predict_node);
        m_predictor_table[index + way*m_table_size/m_ways].m_timestamp = cycle;
        return;
      }
    }
  }
  
  // Otherwise, create new entry
  predictor_entry new_entry;
  new_entry.m_tag = hash;
  new_entry.m_timestamp = cycle;
  new_entry.m_valid = true;
  new_entry.m_nodes.insert(predict_node);

  switch (m_node_replacement_policy) {
    case 'n':
    case 'f':
    case 'l':
      new_entry.m_node_use_map[predict_node] = new_entry.m_timestamp;
      break;
    case 'u':
      new_entry.m_node_use_map[predict_node] = 0;
      break;
    default:
      assert(false);
  }
  
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
    unsigned long long lru = cycle + 1;
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
  unsigned long long cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle +
    GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;

  // LRU
  if (m_replacement_policy == 'l') {
    unsigned long long lru = cycle + 1;
    unsigned long long evicted_hash;
    for (auto it=m_predictor_table.begin(); it!=m_predictor_table.end(); ++it) {
      if (it->second.m_timestamp < lru) {
        evicted_hash = it->first;
        lru = it->second.m_timestamp;
      }
    }
    assert(lru != (cycle + 1));
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

void ray_predictor::reset_stats() {
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

bool ray_predictor::validate_prediction(const std::vector<new_addr_type>& prediction_list, const Ray ray_properties, unsigned tid, new_addr_type& hit_node) {
  bool valid;
  
  // Save list of memory requests required to validate prediction
  std::deque<new_addr_type> predictor_mem_accesses;

  cuda_sim* func_sim = GPGPU_Context()->func_sim;
  unsigned verified_node_accesses = 0;
  unsigned verified_triangle_accesses = 0;
  
  // Iterate through predictions
  for (auto it=prediction_list.begin(); it!=prediction_list.end(); ++it) {
    verified_node_accesses++;

    valid = traverse_intersect(*it, ray_properties, predictor_mem_accesses, verified_triangle_accesses);
    // If intersected, the prediction is valid
    if (valid) {
      mem_access_saved += m_current_warp.update_rt_mem_accesses(tid, valid, predictor_mem_accesses);

      // Only add when we have a valid prediction
      func_sim->g_total_raytrace_perfect_verified_node_accesses += verified_node_accesses;
      func_sim->g_total_raytrace_perfect_verified_triangle_accesses += verified_triangle_accesses;

      hit_node = *it;
      return true;
    }
  }
  
  assert(!valid);
  mem_access_saved += m_current_warp.update_rt_mem_accesses(tid, valid, predictor_mem_accesses);
  return valid;
}

bool ray_predictor::traverse_intersect(const new_addr_type prediction, const Ray ray_properties, std::deque<new_addr_type> &mem_accesses, unsigned& num_triangles_tested) {
  
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
    num_triangles_tested++;
      
    bool hit = rtao_ray_triangle_test(p0, p1, p2, ray_properties, &thit);
    if (hit) {
      GPGPU_Context()->func_sim->g_total_raytrace_verified_rays++;
      assert(ray_properties.anyhit);
      return true;
    }
      
          
    // Go to next triangle
    tri_addr += 0x30;

  }
}