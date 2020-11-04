#ifndef RAY_PREDICTOR_FUNCTION_INCLUDED
#define RAY_PREDICTOR_FUNCTION_INCLUDED

class ray_predictor {
  public:
    ray_predictor(  unsigned go_up_level, unsigned number_of_entries_cap,
                    bool use_replacement_policy, unsigned entry_threshold, unsigned cycle_delay );
    ~ray_predictor();
    
    
    // TODO: Implement these parameters
    unsigned m_go_up_level;
    unsigned m_number_of_entries_cap;
    bool m_use_replacement_policy;
    unsigned m_entry_threshold;
    unsigned m_cycle_delay;
    
    bool empty() { return !m_busy; }
    warp_inst_t lookup(warp_inst_t inst);
    void cycle();
    void display_state(FILE* fout);
    
    unsigned predictor_table_size() {return m_predictor_table.size(); }
    
  private:
  
    // TODO:
    void evict_entry();
    void add_entry();
    void reset_cycle_delay() { m_cycles = m_cycle_delay; };
    
    std::map<unsigned long long, unsigned long long> m_predictor_table;
    
    warp_inst_t m_current_warp;
    
    bool m_busy;
    
    unsigned m_cycles;
};

#endif