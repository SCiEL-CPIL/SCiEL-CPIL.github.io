  def construct():
      n, m, p = 3, 3, 7
      # We will build U, V, W for rank 55
      # Part 1: k in [0, 2] -> M_3,3,3
      # Part 2: k in [3, 5] -> M_3,3,3
      # Part 3: k in [6]    -> M_3,3,1
      
      # We need the 23 (u,v,w) tuples for M_3,3,3
      # And the 9 (u,v,w) tuples for M_3,3,1
      
      # M_3,3,1 is easy.
      # M_3,3,3 requires the specific 23 terms.
  