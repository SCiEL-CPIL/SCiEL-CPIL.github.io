    # Fix V_s construction
    # Old: V_s[j * 2 + k, m_idx] += c
    # New: V_s[k * 2 + j, m_idx] += c
    
    # Implement 4x4*4x4 part
    # Loop T, t_min, I, i_min, K, k_min, J, j_min
    # Fill U, V, W columns 0..48
    
    # Fix V_s retrieval in 4x4*4x2 part
    # Old: val = V_s[j * 2 + k, s]
    # New: val = V_s[k * 2 + j, s]
