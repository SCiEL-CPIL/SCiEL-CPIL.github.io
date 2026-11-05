      def build_rank_16():
          rank = 16
          U = np.zeros((nm, rank))
          V = np.zeros((mp, rank))
          W = np.zeros((pn, rank))
          col = 0
          
          # Strassen 1: j in {0,1}, k in {0,1}
          # ... (same as before)
          
          # Strassen 2: j in {0,1}, k in {2,3}
          # ... (same as before)
          
          # Term 3: j=2, all k
          # U[i*m+2, col] = 1 for i in 0,1
          # V[2*p+k, col] = 1 for k in 0..4
          # W[k*n+i, col] = 1 for all k, i
          
          # Term 4: k=4, j in {0,1}
          # U[i*m+j, col] = 1 for i in 0,1, j in 0,1
          # V[j*p+4, col] = 1 for j in 0,1
          # W[4*n+i, col] = 1 for i in 0,1
      