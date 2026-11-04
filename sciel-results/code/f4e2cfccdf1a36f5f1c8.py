  def add_strassen_2x2x2_gen(U_dict, V_dict, W_dict):
      # U_dict, V_dict, W_dict are dicts mapping 4 relative indices to actual indices/coeffs
      # We assume the 4 entries are at (0,0), (0,1), (1,0), (1,1)
      
      U00, U01, U10, U11 = U_dict[0], U_dict[1], U_dict[2], U_dict[3]
      V00, V01, V10, V11 = V_dict[0], V_dict[1], V_dict[2], V_dict[3]
      W00, W01, W10, W11 = W_dict[0], W_dict[1], W_dict[2], W_dict[3]
      
      # Term 1: (U00+U11)(V00+V11)
      add_term({**U00, **U11}, {**V00, **V11}, {**W00, **W11})
      # ... and so on
  