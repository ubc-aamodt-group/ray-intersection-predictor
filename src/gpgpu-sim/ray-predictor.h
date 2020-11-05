#ifndef RAY_PREDICTOR_FUNCTION_INCLUDED
#define RAY_PREDICTOR_FUNCTION_INCLUDED

#include "shader.h"


struct {
    unsigned long long m_tag;
    unsigned long long m_timestamp;
} typedef predictor_entry;

class ray_predictor {
  public:
    ray_predictor(unsigned sid, struct ray_predictor_config config);
    ~ray_predictor();
    
    
    // TODO: Implement these parameters
    unsigned m_go_up_level;
    unsigned m_number_of_entries_cap;
    char * m_replacement_policy;
    unsigned m_entry_threshold;
    unsigned m_cycle_delay;
    
    unsigned m_sid;
    
    bool empty() { return !m_busy; }
    warp_inst_t lookup(warp_inst_t inst);
    void cycle();
    void display_state(FILE* fout);
    void print_stats(FILE* fout);
    
    unsigned predictor_table_size() {return m_predictor_table.size(); }
    
  private:
  
    // TODO:
    void evict_entry(unsigned long long hash);
    void add_entry(unsigned long long hash);
    bool check_table(unsigned long long hash);
    void update_LRU(unsigned long long hash);
    void reset_cycle_delay() { m_cycles = m_cycle_delay; };
    
    std::map<unsigned long long, predictor_entry> m_predictor_table;
    warp_inst_t m_current_warp;
    bool m_busy;
    
    unsigned m_cycles;
    
    // Stats
    unsigned num_predicted;
    unsigned num_valid;
    unsigned capacity_miss;
    int mem_access_saved;
};

#endif