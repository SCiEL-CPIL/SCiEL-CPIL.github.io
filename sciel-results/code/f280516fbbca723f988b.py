<<<<<<< SEARCH
        else:
            # Peeling construction
            # For simplicity, fall back to schoolbook if peeling rank not achieved by split
            # This is a placeholder - the actual peeling construction is complex
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            return U, V, W, rank
=======
        else:
            # Peeling construction
            # C[0,0] = sum_j A[0,j]*B[j,0]
            # C[0,k] = sum_j A[0,j]*B[j,k] for k=1..p-1
            # C[i,0] = sum_j A[i,j]*B[j,0] for i=1..n-1
            # C[i,k] = sum_j A[i,j]*B[j,k] for i=1..n-1, k=1..p-1
            
            # Subproblem: (n-1) x (m-1) x (p-1) for C[1:,1:]
            # This uses A[1:,1:], B[1:,1:]
            U_sub, V_sub, W_sub, r_sub = self.construct(n-1, m-1, p-1)
            
            # Additional terms:
            # Term 0: C[0,0] = sum_j A[0,j]*B[j,0]
            #   U[0,j] += 1 for j=0..m-1
            #   V[j,0] += 1 for j=0..m-1
            #   W[0,0] += 1
            # Terms 1..p-1: C[0,k] = sum_j A[0,j]*B[j,k] for k=1..p-1
            #   U[0,j] += 1 for j=0..m-1
            #   V[j,k] += 1 for j=0..m-1
            #   W[k,0] += 1
            # Terms p..p+n-2: C[i,0] = sum_j A[i,j]*B[j,0] for i=1..n-1
            #   U[i,j] += 1 for j=0..m-1
            #   V[j,0] += 1 for j=0..m-1
            #   W[0,i] += 1
            
            total_rank = r_sub + 1 + (p-1) + (n-1)
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Fill subproblem: C[1:,1:] uses A[1:,1:], B[1:,1:]
            # Subproblem indices: A' has (n-1)*(m-1) rows, B' has (m-1)*(p-1) rows, C' has (p-1)*(n-1) rows
            for i in range(1, n):
                for j in range(1, m):
                    for c in range(r_sub):
                        U[i*m+j][c] = U_sub[(i-1)*(m-1)+(j-1)][c]
            for j in range(1, m):
                for k in range(1, p):
                    for c in range(r_sub):
                        V[j*p+k][c] = V_sub[(j-1)*(p-1)+(k-1)][c]
            for k in range(1, p):
                for i in range(1, n):
                    for c in range(r_sub):
                        W[k*n+i][c] = W_sub[(k-1)*(n-1)+(i-1)][c]
            
            # Term 0: C[0,0] = sum_j A[0,j]*B[j,0]
            col = r_sub
            for j in range(m):
                U[0*m+0][col] = 1  # A[0,j] - wait, need to index correctly
                # A[0,j] is at row 0*m + j = j
                U[j][col] = 1
                V[j*p+0][col] = 1  # B[j,0]
            W[0*n+0][col] = 1  # C[0,0]
            
            # Terms 1..p-1: C[0,k] = sum_j A[0,j]*B[j,k]
            for k in range(1, p):
                col = r_sub + k
                for j in range(m):
                    U[j][col] = 1  # A[0,j]
                    V[j*p+k][col] = 1  # B[j,k]
                W[0*n+0][col] = 1  # C[0,k] -> W[k*n+0]
                W[k*n+0][col] = 1
            
            # Terms p..p+n-2: C[i,0] = sum_j A[i,j]*B[j,0]
            for i in range(1, n):
                col = r_sub + p - 1 + i
                for j in range(m):
                    U[i*m+j][col] = 1  # A[i,j]
                    V[j*p+0][col] = 1  # B[j,0]
                W[0*n+i][col] = 1  # C[i,0]
            
            return U, V, W, total_rank
>>>>>>> REPLACE

Wait, I made an indexing error. Let me fix the peeling construction:

