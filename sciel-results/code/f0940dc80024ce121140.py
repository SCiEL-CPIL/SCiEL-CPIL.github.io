  import numpy as np

  def _strassen_222():
      # Returns U, V, W for 2x2x2 multiplication
      # A is 2x2, B is 2x2, C = AB is 2x2
      # Indices for A: 0,1,2,3 (row-major: a00, a01, a10, a11)
      # Indices for B: 0,1,2,3 (row-major: b00, b01, b10, b11)
      # Indices for C: 0,1,2,3 (row-major: c00, c01, c10, c11)
      # 
      # M1 = (a00+a11)(b00+b11)
      # M2 = (a10+a11)b00
      # M3 = a00(b01-b11)
      # M4 = a11(b10-b00)
      # M5 = (a00+a01)b11
      # M6 = (a10-a00)(b00+b01)
      # M7 = (a01-a11)(b10+b11)
      #
      # c00 = M1+M4-M5+M7
      # c01 = M3+M5
      # c10 = M2+M4
      # c11 = M1-M2+M3+M6
      
      rank = 7
      U = np.zeros((4, rank))
      V = np.zeros((4, rank))
      W = np.zeros((4, rank))
      
      # M1
      U[0,0] = 1; U[3,0] = 1
      V[0,0] = 1; V[3,0] = 1
      W[0,0] = 1; W[3,0] = 1
      
      # M2
      U[2,1] = 1; U[3,1] = 1
      V[0,1] = 1
      W[2,1] = 1; W[3,1] = 1
      
      # M3
      U[0,2] = 1
      V[1,2] = 1; V[3,2] = -1
      W[1,2] = 1; W[3,2] = 1
      
      # M4
      U[3,3] = 1
      V[2,3] = 1; V[0,3] = -1
      W[0,3] = 1; W[2,3] = 1
      
      # M5
      U[0,4] = 1; U[1,4] = 1
      V[3,4] = 1
      W[0,4] = -1; W[1,4] = 1
      
      # M6
      U[2,5] = 1; U[0,5] = -1
      V[0,5] = 1; V[1,5] = 1
      W[3,5] = 1
      
      # M7
      U[1,6] = 1; U[3,6] = -1
      V[2,6] = 1; V[3,6] = 1
      W[0,6] = 1
      
      return U, V, W

  def _schoolbook(n, m, p):
      rank = n * m * p
      U = np.zeros((n * m, rank))
      V = np.zeros((m * p, rank))
      W = np.zeros((p * n, rank))
      col = 0
      for i in range(n):
          for j in range(m):
              for k in range(p):
                  U[i * m + j, col] = 1
                  V[j * p + k, col] = 1
                  W[k * n + i, col] = 1
                  col += 1
      return U, V, W

  def _decompose(n, m, p):
      if n == 2 and m == 2 and p == 2:
          return _strassen_222()
      if n == 1 or m == 1 or p == 1:
          return _schoolbook(n, m, p)
      
      # Split strategy: Split the dimension with size > 2 into 2 and rest, or 1 and rest?
      # To maximize 2x2x2 blocks, we want sizes to be 2.
      # If n > 2, split n into 2 and n-2? Or 1 and n-1?
      # Let's try to keep dimensions as 2 as much as possible.
      
      # Simple heuristic: Split the largest dimension into two parts.
      # If n is largest, split n into n1, n2.
      # If n is even, n1=n/2, n2=n/2.
      # If n is odd, n1=(n-1)//2, n2=(n+1)//2.
      
      # For 3x5x5:
      # n=3, m=5, p=5.
      # Split n=3 into 1 and 2? Or 2 and 1?
      # If we split into 2 and 1:
      # Block 1: 2x5x5
      # Block 2: 1x5x5
      # 1x5x5 is schoolbook (rank 25).
      # 2x5x5: Split m=5 into 2 and 3? Or 2 and 2 and 1?
      # This recursion might be deep.
      
      # Let's just use schoolbook for anything not 2x2x2 to ensure correctness and simplicity,
      # but we can manually construct the 3x5x5 using blocks of 2x2x2 and schoolbook.
      
      # Actually, the recursive approach above is a valid "new approach".
      # Let's implement it carefully.
      
      if n >= m and n >= p:
          if n % 2 == 0:
              n1, n2 = n // 2, n // 2
          else:
              n1, n2 = (n - 1) // 2, (n + 1) // 2
          U1, V1, W1 = _decompose(n1, m, p)
          U2, V2, W2 = _decompose(n2, m, p)
          
          U = np.zeros((n * m, U1.shape[1] + U2.shape[1]))
          V = np.zeros((m * p, U1.shape[1] + U2.shape[1]))
          W = np.zeros((p * n, U1.shape[1] + W1.shape[1]))
          
          # U1 corresponds to rows 0..n1-1 of A
          U[0:n1*m, 0:U1.shape[1]] = U1
          # U2 corresponds to rows n1..n-1 of A
          U[n1*m:n*m, U1.shape[1]:] = U2
          
          # V is same for both
          V[:, 0:V1.shape[1]] = V1
          V[:, V1.shape[1]:] = V2
          
          # W1 corresponds to cols 0..n1-1 of C (which are rows 0..n1-1 in C's row-major p*n)
          # C is p x n. Index is k*n + i.
          # For block 1 (i in 0..n1-1), W indices are k*n + i.
          # For block 2 (i in n1..n-1), W indices are k*n + i.
          
          # We need to place W1 and W2 into the correct positions.
          # W1 has shape (p*n1, rank1).
          # W2 has shape (p*n2, rank2).
          # Target W has shape (p*n, total_rank).
          
          for k in range(p):
              # Block 1 rows in W: k*n to k*n + n1
              W[k*n : k*n + n1, 0:W1.shape[1]] = W1[k*n1 : k*n1 + n1, :]
              # Block 2 rows in W: k*n + n1 to k*n + n
              W[k*n + n1 : k*n + n, W1.shape[1]:] = W2[k*n2 : k*n2 + n2, :]
              
          return U, V, W
      elif m >= n and m >= p:
          # Split m
          if m % 2 == 0:
              m1, m2 = m // 2, m // 2
          else:
              m1, m2 = (m - 1) // 2, (m + 1) // 2
          U1, V1, W1 = _decompose(n, m1, p)
          U2, V2, W2 = _decompose(n, m2, p)
          
          U = np.zeros((n * m, U1.shape[1] + U2.shape[1]))
          V = np.zeros((m * p, U1.shape[1] + U2.shape[1]))
          W = np.zeros((p * n, U1.shape[1] + W1.shape[1]))
          
          # U1: cols 0..m1-1 of A. A is n x m. Index i*m + j.
          for i in range(n):
              U[i*m : i*m + m1, 0:U1.shape[1]] = U1[i*m1 : i*m1 + m1, :]
              U[i*m + m1 : i*m + m, U1.shape[1]:] = U2[i*m2 : i*m2 + m2, :]
              
          # V1: rows 0..m1-1 of B. B is m x p. Index j*p + k.
          V[0:m1*p, 0:V1.shape[1]] = V1
          V[m1*p:m*p, V1.shape[1]:] = V2
              
          # W is same
          W[:, 0:W1.shape[1]] = W1
          W[:, W1.shape[1]:] = W2
              
          return U, V, W
      else:
          # Split p
          if p % 2 == 0:
              p1, p2 = p // 2, p // 2
          else:
              p1, p2 = (p - 1) // 2, (p + 1) // 2
          U1, V1, W1 = _decompose(n, m, p1)
          U2, V2, W2 = _decompose(n, m, p2)
          
          U = np.zeros((n * m, U1.shape[1] + U2.shape[1]))
          V = np.zeros((m * p, U1.shape[1] + U2.shape[1]))
          W = np.zeros((p * n, U1.shape[1] + W1.shape[1]))
          
          # U is same
          U[:, 0:U1.shape[1]] = U1
          U[:, U1.shape[1]:] = U2
          
          # V1: cols 0..p1-1 of B.
          for j in range(m):
              V[j*p : j*p + p1, 0:V1.shape[1]] = V1[j*p1 : j*p1 + p1, :]
              V[j*p + p1 : j*p + p, V1.shape[1]:] = V2[j*p2 : j*p2 + p2, :]
              
          # W1: cols 0..p1-1 of C. C is p x n. Index k*n + i.
          W[0:p1*n, 0:W1.shape[1]] = W1
          W[p1*n:p*n, W1.shape[1]:] = W2
              
          return U, V, W

  def construct():
      n, m, p = 3, 5, 5
      U, V, W = _decompose(n, m, p)
      return {"U": U, "V": V, "W": W}
  