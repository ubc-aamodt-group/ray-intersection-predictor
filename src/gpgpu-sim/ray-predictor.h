#ifndef RAY_PREDICTOR_FUNCTION_INCLUDED
#define RAY_PREDICTOR_FUNCTION_INCLUDED

#include "../abstract_hardware_model.h"


struct {
    bool m_valid;
    unsigned long long m_tag;
    unsigned long long m_timestamp;
    std::deque<new_addr_type> m_nodes;
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
    char m_replacement_policy;
    char m_placement_policy;
    unsigned m_table_size;
    unsigned m_cycle_delay;
    bool m_virtualize;
    unsigned m_virtualize_delay;
    
    unsigned m_sid;
    
    bool empty() { return !m_busy; }
    warp_inst_t lookup(const warp_inst_t& inst);
    void cycle();
    void display_state(FILE* fout);
    void print_stats(FILE* fout);
    
    unsigned predictor_table_size() {return m_predictor_table.size(); }
    float predictor_prediction_rate() {return (float) num_predicted / num_rays; }
    float predictor_verification_rate() {return (float) num_valid / num_rays; }
    unsigned predictor_num_predicted() {return num_predicted; }
    unsigned predictor_num_verified() {return num_valid; }
    unsigned predictor_num_rays() {return num_rays; }
    
  private:
  
    // TODO:
    void evict_entry();
    void add_entry(unsigned long long hash, new_addr_type prediction);
    bool check_table(unsigned long long hash);
    void reset_cycle_delay(unsigned delay) { m_cycles = delay; };
    bool validate_prediction(const std::deque<new_addr_type> prediction_list, const Ray ray_properties, unsigned tid);
    bool traverse_intersect(const new_addr_type prediction, const Ray ray_properties, std::deque<new_addr_type> &mem_accesses);
    
    std::map<unsigned long long, predictor_entry> m_predictor_table;
    warp_inst_t m_current_warp;
    bool m_busy;
    
    unsigned m_cycles;
    
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
};

#endif