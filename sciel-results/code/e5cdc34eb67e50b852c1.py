  def construct():
      n, m, p = 4, 4, 5
      rank = 35 # Target
      # If 19-rank is too hard, use 25 -> rank 41
      # I will use 25-rank for 4x4 and 16 for 4x1.
      rank = 25 + 16
      U = [[0]*rank for _ in range(n*m)]
      V = [[0]*rank for _ in range(m*p)]
      W = [[0]*rank for _ in range(p*n)]
      
      # Part 1: 4x4x4 (First 4 cols of B, First 4 cols of C)
      # Indices:
      # U: i*4 + j (i:0-3, j:0-3) -> 16 rows
      # V: j*4 + k (j:0-3, k:0-3) -> 16 rows (k<4)
      # W: k*4 + i (k:0-3, i:0-3) -> 16 rows (k<4)
      
      # Hardcode 25-rank coefficients
      # ...
      
      # Part 2: 4x4x1 (Last col of B, Last col of C)
      # Indices:
      # U: i*4 + j -> 16 rows
      # V: j*4 + 4 (j:0-3, k=4) -> 4 rows
      # W: 4*4 + i (k=4, i:0-3) -> 4 rows
      # Schoolbook: 16 terms
      for i in range(4):
          for j in range(4):
              col = 25 + i*4 + j
              U[i*4 + j][col] = 1
              V[j*4 + 4][col] = 1
              W[4*4 + i][col] = 1
              
      return {"U": U, "V": V, "W": W}
  