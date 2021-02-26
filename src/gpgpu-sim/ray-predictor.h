#ifndef RAY_PREDICTOR_FUNCTION_INCLUDED
#define RAY_PREDICTOR_FUNCTION_INCLUDED

#include "../abstract_hardware_model.h"


struct {
    bool m_valid;
    unsigned long long m_tag;
    unsigned long long m_timestamp;
    std::set<new_addr_type> m_nodes;
    std::map<new_addr_type, unsigned long long> m_node_use_map;
} typedef predictor_entry;

class shader_core_ctx;

class ray_predictor {
  public:
    ray_predictor(unsigned sid, struct ray_predictor_config config, shader_core_ctx *core);
    ~ray_predictor();
    
    
    // Backwards pointer
    shader_core_ctx *m_core;
    
    // TODO: Implement these parameters
    unsigned m_go_up_level;
    unsigned m_number_of_entries_cap;
    char m_node_replacement_policy;
    char m_replacement_policy;
    char m_placement_policy;
    unsigned m_table_size;
    unsigned m_cycle_delay;
    bool m_virtualize;
    unsigned m_virtualize_delay;
    unsigned m_ways;
    bool m_repack_warps;
    bool m_repack_oracle;
    bool m_repack_unpredicted_warps;
    unsigned m_per_thread_latency;
    unsigned m_lookup_bandwidth;
    bool m_magic_verify;
    bool m_oracle_update;
    unsigned m_repacking_timer;
    
    unsigned m_verified_warp_id;
    unsigned m_unverified_warp_id;
    
    unsigned m_sid;
    
    bool busy() { return m_busy; }
    bool ready() { return m_ready; }
    void insert(const warp_inst_t& inst);
    warp_inst_t retrieve();
    void cycle();
    void display_state(FILE* fout);
    void print_stats(FILE* fout);
    void reset_stats();
    
    unsigned predictor_table_size() {return m_predictor_table.size(); }
    float predictor_prediction_rate() {return (float) num_predicted / num_rays; }
    float predictor_verification_rate() {return (float) num_valid / num_rays; }
    unsigned predictor_num_predicted() {return num_predicted; }
    unsigned predictor_num_verified() {return num_valid; }
    unsigned predictor_num_rays() {return num_rays; }
    void add_entry(unsigned long long hash, new_addr_type prediction);
    
    unsigned num_predictor_warps() { return m_predictor_warps.size() + !m_current_warp.empty(); }
    
  private:
  
    // TODO:
    void evict_entry();
    bool check_table(unsigned long long hash, unsigned long long &index);
    void reset_cycle_delay(unsigned delay) { m_cycles = delay; };
    bool validate_prediction(const std::vector<new_addr_type>& prediction_list, const Ray ray_properties, unsigned tid, new_addr_type& hit_node);
    bool traverse_intersect(const new_addr_type prediction, const Ray ray_properties, std::deque<new_addr_type> &mem_accesses, unsigned& num_triangles_tested);

    void add_node_to_predictor_entry(unsigned long long index, new_addr_type node);
    std::vector<new_addr_type> get_prediction_list(unsigned long long index) const;
    void update_entry_use(unsigned long long index, new_addr_type node);

    unsigned long long compute_index(unsigned long long hash, unsigned num_bits) const;
    
    std::map<unsigned long long, predictor_entry> m_predictor_table;
    warp_inst_t m_current_warp;
    
    std::deque<warp_inst_t> m_predictor_warps;
    
    // Requires oracle knowledge
    std::deque<struct warp_inst_t::per_thread_info> verified_threads;
    std::deque<struct warp_inst_t::per_thread_info> unverified_threads;
    
    std::deque<struct warp_inst_t::per_thread_info> unpredicted_threads;
    std::deque<struct warp_inst_t::per_thread_info> predicted_threads;
    unsigned m_total_threads;
    
    bool m_busy;
    bool m_ready;
    
    int m_cycles;
    
    // Stats
    unsigned num_predicted;
    unsigned num_miss;
    unsigned num_valid;
    unsigned num_evicted;
    unsigned num_entry_overflow;
    unsigned num_virtual_predicted;
    unsigned num_virtual_miss;
    unsigned num_virtual_valid;
    unsigned num_rays;
    int mem_access_saved;
    unsigned verified_packets;
    unsigned unverified_packets;
    unsigned unpredicted_packets;
    unsigned predicted_packets;
    unsigned mixed_packets;
    unsigned new_warps;
};

#endif