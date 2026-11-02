   def construct():
       n, m, p = 3, 5, 6
       rank = 27
       U = [[0]*rank for _ in range(n*m)]
       V = [[0]*rank for _ in range(m*p)]
       W = [[0]*rank for _ in range(p*n)]
       
       r = 0
       
       # Helper to add a simple 1x1x1 style block (rank 1)
       # for n x 1 x p
       def add_simple_block(i_start, j, k_start, k_len, r_idx):
           # i_start: start i
           # j: single j
           # k_start, k_len: range of k
           for i in range(n):
               U[i*m + j][r_idx] += 1
           for k in range(k_start, k_start+k_len):
               V[j*p + k][r_idx] += 1
               W[k*n + i_start][r_idx] += 1 # Wait, W index is k*n + i
               # Need to loop i for W
           # Correct W loop:
           for k in range(k_start, k_start+k_len):
               for i in range(n):
                   W[k*n + i][r_idx] += 1
                   
       # Block 1: j=0
       add_simple_block(0, 0, 0, 6, 0)
       r = 1
       
       # Block 2: j=1
       add_simple_block(0, 1, 0, 6, 1)
       r = 2
       
       # Block 3b: j=4
       add_simple_block(0, 4, 0, 6, 2)
       r = 3
       
       # Block 3a: j=2,3. 3 x 2 x 6.
       # Split k into 3 pairs.
       for k_pair in range(3):
           k_start = k_pair * 2
           # Add 3x2x2 block at i=0..2, j=2..3, k=k_start..k_start+1
           # This block uses 8 ranks.
           
           # 1. Top row (i=0) simple block for this 2x2 k range
           # i=0, j=2,3. k=k_start, k_start+1
           U[0*m + 2][r] += 1
           U[0*m + 3][r] += 1
           V[2*p + k_start][r] += 1
           V[2*p + k_start+1][r] += 1
           V[3*p + k_start][r] += 1
           V[3*p + k_start+1][r] += 1
           for i in range(n):
               W[k_start*n + i][r] += 1
               W[(k_start+1)*n + i][r] += 1
           r += 1
           
           # 2. Bottom 2 rows (i=1,2) Strassen
           # Local i: 0->global 1, 1->global 2
           # Local j: 0->global 2, 1->global 3
           # Local k: 0->global k_start, 1->global k_start+1
           
           # Strassen mults
           # 1: A00 (B00+B11)
           U[1*m+2][r] += 1
           V[2*p+k_start][r] += 1
           V[3*p+k_start+1][r] += 1
           W[k_start*n+1][r] += 1
           W[(k_start+1)*n+2][r] += 1
           r += 1
           
           # 2: (A00+A01) B11
           U[1*m+2][r] += 1
           U[1*m+3][r] += 1
           V[3*p+k_start+1][r] += 1
           W[(k_start+1)*n+2][r] += 1
           r += 1
           
           # 3: A10 (B00+B01)
           U[2*m+2][r] += 1
           V[2*p+k_start][r] += 1
           V[2*p+k_start+1][r] += 1
           W[k_start*n+2][r] += 1
           r += 1
           
           # 4: A11 (B10+B11)
           U[2*m+3][r] += 1
           V[3*p+k_start][r] += 1
           V[3*p+k_start+1][r] += 1
           W[(k_start+1)*n+2][r] += 1
           r += 1
           
           # 5: (A00+A10) B00
           U[1*m+2][r] += 1
           U[2*m+2][r] += 1
           V[2*p+k_start][r] += 1
           W[k_start*n+1][r] += 1
           W[k_start*n+2][r] += 1
           r += 1
           
           # 6: (A01-A10) (B10+B00)
           U[1*m+3][r] += 1
           U[2*m+2][r] -= 1
           V[3*p+k_start][r] += 1
           V[2*p+k_start][r] += 1
           W[k_start*n+2][r] += 1
           r += 1
           
           # 7: (A00-A01) (B01+B11)
           U[1*m+2][r] += 1
           U[1*m+3][r] -= 1
           V[2*p+k_start+1][r] += 1
           V[3*p+k_start+1][r] += 1
           W[(k_start+1)*n+1][r] += 1
           r += 1
           
       return {"U": U, "V": V, "W": W}
   