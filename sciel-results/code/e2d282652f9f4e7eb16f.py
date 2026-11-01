   def construct():
       n, m, p = 2, 4, 4
       rank = 28
       U = [[0]*rank for _ in range(n*m)]
       V = [[0]*rank for _ in range(m*p)]
       W = [[0]*rank for _ in range(p*n)]
       
       # Strassen 2x2x2 components
       U_s = [
           [1, 0, 1, 0, 1, -1, 0],
           [0, 0, 0, 0, 1, 0, 1],
           [0, 1, 0, 0, 0, 1, 0],
           [1, 1, 0, 1, 0, 0, -1]
       ]
       V_s = [
           [1, 0, 0, 1, 0, 1, 0],
           [0, 0, 1, 0, 0, 1, 0],
           [0, 0, 0, 1, 0, 0, 1],
           [1, 0, -1, 0, 1, 0, 1]
       ]
       # Wait, I need to be careful with the column/row orientation.
       # In my derivation:
       # U is 4x7 (rows are local i,j indices)
       # V is 4x7 (rows are local j,k indices)
       # W is 4x7 (rows are local k,i indices)
       
       # Let's stick to the lists I derived.
       
       # Blocks:
       # 0: j_offset=0, k_offset=0
       # 1: j_offset=2, k_offset=0
       # 2: j_offset=0, k_offset=2
       # 3: j_offset=2, k_offset=2
       
       col_offset = 0
       for block in range(4):
           j_off = (block % 2) * 2
           k_off = (block // 2) * 2
           
           for c in range(7):
               # Fill U
               for u_idx in range(4):
                   i = u_idx // 2
                   j_loc = u_idx % 2
                   j_glob = j_off + j_loc
                   U[i*4 + j_glob][col_offset] = U_s[u_idx][c]
                   
               # Fill V
               for v_idx in range(4):
                   j_loc = v_idx // 2
                   k_loc = v_idx % 2
                   j_glob = j_off + j_loc
                   k_glob = k_off + k_loc
                   V[j_glob*4 + k_glob][col_offset] = V_s[v_idx][c]
                   
               # Fill W
               for w_idx in range(4):
                   k_loc = w_idx // 2
                   i = w_idx % 2
                   k_glob = k_off + k_loc
                   W[k_glob*2 + i][col_offset] = W_s[w_idx][c]
                   
               col_offset += 1
               
       return {"U": U, "V": V, "W": W}
   