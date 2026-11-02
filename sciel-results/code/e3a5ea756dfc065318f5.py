  def construct():
      n, m, p = 2, 4, 4
      rank = 28
      U = [[0] * rank for _ in range(n * m)]
      V = [[0] * rank for _ in range(m * p)]
      W = [[0] * rank for _ in range(p * n)]
      
      # Strassen steps for 2x2
      # Each step: (u_coeffs, v_coeffs, w_coeffs)
      # u_coeffs: dict {(i,j): val}
      # v_coeffs: dict {(j,k): val}
      # w_coeffs: dict {(i,k): val}
      
      steps = []
      
      # M1
      steps.append({
          'u': {(0,0):1, (1,1):1},
          'v': {(0,0):1, (1,1):1},
          'w': {(0,0):1, (1,1):1}
      })
      # M2
      steps.append({
          'u': {(1,0):1, (1,1):1},
          'v': {(0,0):1},
          'w': {(1,0):1, (1,1):-1}
      })
      # M3
      steps.append({
          'u': {(0,0):1},
          'v': {(0,1):1, (1,1):-1},
          'w': {(0,1):1, (1,1):1}
      })
      # M4
      steps.append({
          'u': {(1,1):1},
          'v': {(1,0):1, (0,0):-1},
          'w': {(0,0):1, (1,0):1}
      })
      # M5
      steps.append({
          'u': {(0,0):1, (0,1):1},
          'v': {(1,1):1},
          'w': {(0,0):-1, (0,1):1}
      })
      # M6
      steps.append({
          'u': {(1,0):1, (0,0):-1},
          'v': {(0,0):1, (0,1):1},
          'w': {(1,1):1}
      })
      # M7
      steps.append({
          'u': {(0,1):1, (1,1):-1},
          'v': {(1,0):1, (1,1):1},
          'w': {(0,0):1, (1,0):-1}
      })
      
      col = 0
      for block in range(4):
          # Determine offsets
          # block 0: A1 B11 -> C11
          # block 1: A2 B21 -> C11
          # block 2: A1 B12 -> C22
          # block 3: A2 B22 -> C22
          
          if block == 0:
              a_off_j = 0
              b_off_j = 0
              b_off_k = 0
              c_off_k = 0
          elif block == 1:
              a_off_j = 2
              b_off_j = 2
              b_off_k = 0
              c_off_k = 0
          elif block == 2:
              a_off_j = 0
              b_off_j = 0
              b_off_k = 2
              c_off_k = 2
          else: # block 3
              a_off_j = 2
              b_off_j = 2
              b_off_k = 2
              c_off_k = 2
              
          for step in steps:
              # Fill U
              for (i, j), val in step['u'].items():
                  global_i = i
                  global_j = j + a_off_j
                  U[global_i * m + global_j][col] += val
                  
              # Fill V
              for (j, k), val in step['v'].items():
                  global_j = j + b_off_j
                  global_k = k + b_off_k
                  V[global_j * p + global_k][col] += val
                  
              # Fill W
              for (i, k), val in step['w'].items():
                  global_i = i
                  global_k = k + c_off_k
                  # W index is k*n + i
                  W[global_k * n + global_i][col] += val
                  
              col += 1
              
      return {"U": U, "V": V, "W": W}
  