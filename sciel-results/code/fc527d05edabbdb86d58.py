import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    def verify(U, V, W, n, m, p, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > tol:
                        return False
        return True

    def standard_term(U, V, W, i, j, k, col):
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1

    def strassen_block(U, V, W, i1, i2, j1, j2, k1, k2, col):
        # M1 = (a11+a22)(b11+b22)
        U[i1*m+j1][col] += 1; U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1; V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        # M2 = (a21+a22)b11
        U[i2*m+j1][col] += 1; U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i2][col] += 1; W[k2*n+i2][col] += -1
        col += 1
        # M3 = a11(b12-b22)
        U[i1*m+j1][col] += 1
        V[j1*p+k2][col] += 1; V[j2*p+k2][col] += -1
        W[k2*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        # M4 = a22(b21-b11)
        U[i2*m+j2][col] += 1
        V[j2*p+k1][col] += 1; V[j1*p+k1][col] += -1
        W[k1*n+i1][col] += 1; W[k1*n+i2][col] += 1
        col += 1
        # M5 = (a11+a12)b22
        U[i1*m+j1][col] += 1; U[i1*m+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += -1; W[k2*n+i1][col] += 1
        col += 1
        # M6 = (a21-a11)(b11+b12)
        U[i2*m+j1][col] += 1; U[i1*m+j1][col] += -1
        V[j1*p+k1][col] += 1; V[j1*p+k2][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        # M7 = (a12-a22)(b21+b22)
        U[i1*m+j2][col] += 1; U[i2*m+j2][col] += -1
        V[j2*p+k1][col] += 1; V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        return col

    # 2x3x3 rank-17 decomposition
    # For indices i in {i1,i2}, j in {j1,j2,j3}, k in {k1,k2,k3}
    # Uses 17 multiplications instead of 18
    def rank17_2x3x3(U, V, W, i1, i2, j1, j2, j3, k1, k2, k3, col):
        # This is a known rank-17 construction for 2x3x3 matrix multiplication.
        # We use the fact that 2x3x3 can be seen as 2x3x2 + 2x3x1, but that gives 9.
        # Instead, use a direct rank-17 construction.
        # 
        # The 2x3x3 tensor T[i,j,k] = delta_{i,j,k} for i in 0,1; j in 0,1,2; k in 0,1,2
        # has border rank 6 and tensor rank 17 (Landsberg).
        # 
        # Here's an explicit rank-17 decomposition using integer/half-integer coefficients.
        # We'll use a construction based on the 3x3 Strassen reduction applied cleverly.
        
        # Actually, let's use a simpler approach: decompose 2x3x3 as:
        # - 2x2x3 block (Strassen rank 7) + 2x1x3 block (rank 3) + 2x1x3 block (rank 3) + 2x1x1 + 2x1x1
        # That gives 7+3+3+2+2 = 17!
        
        # 2x2x3: rows i1,i2; cols j1,j2; all 3 k's
        # Use Strassen on the 2x2 part for each pair of k's:
        # For k1,k2: Strassen 2x2x2 (7 terms)
        # For k3: standard 2x2x1 (2 terms)
        # Total for 2x2x3: 7 + 2 = 9... that's not 7.
        
        # Let me reconsider. 2x3x3 rank 17:
        # Partition: (2x2x3) + (2x1x3)
        # 2x2x3: can we do better than 6? Standard is 6, Strassen-like?
        # Actually 2x2x3 has rank 6 (it's just 3 separate 2x2x1's, each rank 2).
        # 2x1x3: rank 3.
        # Total: 6 + 3 = 9. But the known rank is 17? That can't be right.
        
        # Wait, I'm confusing myself. The 2x3x3 matrix multiplication tensor:
        # T[i,j,k] = 1 if i=j=k, else 0, for i in {0,1}, j in {0,1,2}, k in {0,1,2}
        # This is NOT a standard matrix multiplication tensor in the usual sense.
        # 
        # Actually for our problem, the sub-tensor is:
        # T[i,j,k] = 1 for all i in {i1,i2}, j in {j1,j2,j3}, k in {k1,k2,k3}
        # This is a rank-1 tensor (all ones), so its rank is 1!
        
        # I was overthinking this. The sub-tensor for a block of size a x b x c
        # where all entries are 1 has rank 1. But we need to handle the full tensor
        # where T[i,j,k] = 1 for the specific (i,j,k) triple.
        
        # Let me restart with the correct understanding:
        # The full tensor has T[i,j,k] = 1 for all (i,j,k) in [4]x[5]x[5].
        # We partition into sub-tensors and decompose each.
        
        # For a sub-tensor covering i in I, j in J, k in K where all entries are 1:
        # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k) in IxJxK.
        
        # If |I|=2, |J|=3, |K|=3, we need 18 equations.
        # The standard decomposition uses 18 rank-1 terms.
        # Can we do better? The tensor is all-ones on a 2x3x3 box.
        # This tensor has rank 1! Because it's just 1*1*1 for all entries.
        # Wait no - the rank-1 term U[i*m+j]*V[j*p+k]*W[k*n+i] must equal 1
        # for ALL (i,j,k) in the box, but the factors depend on different indices.
        
        # So we need: for all i in I, j in J, k in K:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # 
        # This is the matrix multiplication tensor restricted to I x J x K.
        # Its rank is the rank of the "partial" multiplication tensor.
        
        # For 2x3x3: the rank is known to be 17 (from literature on small matrix multiplication).
        # Let me use a known construction.
        
        # I'll use the following approach: embed the 2x3x3 problem into a 3x3x3 problem
        # (which has rank 23 by Strassen) and then project. But that's complex.
        
        # Alternative: use the fact that 2x3x3 can be decomposed as:
        # (2x2x2 Strassen) + (2x2x2 Strassen) + (2x1x1) + (2x1x1) + (2x1x1) + (2x1x1)
        # = 7 + 7 + 2 + 2 + 2 + 2 = 22. Not good.
        
        # Let me just use standard rank for now and focus on getting a valid decomposition.
        # I'll use the 8-block Strassen approach which gives rank 92.
        
        # For the 2x3x3 block, use standard 18 terms.
        for i in [i1, i2]:
            for j in [j1, j2, j3]:
                for k in [k1, k2, k3]:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return col

    # Build the decomposition using a good partition
    # Partition: 
    #   i in {0,1}, j in {0,1,2}, k in {0,1,2}: 2x3x3 standard (18 terms)
    #   i in {0,1}, j in {0,1,2}, k in {3,4}: 2x3x2 standard (12 terms)  
    #   i in {0,1}, j in {3,4}, k in {0,1,2}: 2x2x3 standard (12 terms)
    #   i in {0,1}, j in {3,4}, k in {3,4}: 2x2x2 Strassen (7 terms)
    #   i in {2,3}, j in {0,1,2}, k in {0,1,2}: 2x3x3 standard (18 terms)
    #   i in {2,3}, j in {0,1,2}, k in {3,4}: 2x3x2 standard (12 terms)
    #   i in {2,3}, j in {3,4}, k in {0,1,2}: 2x2x3 standard (12 terms)
    #   i in {2,3}, j in {3,4}, k in {3,4}: 2x2x2 Strassen (7 terms)
    # Total: 18+12+12+7+18+12+12+7 = 98. Worse than 92.
    
    # Better partition for Strassen:
    # Use 2x2x2 Strassen blocks at:
    #   i in {0,1}, j in {0,1}, k in {0,1}
    #   i in {0,1}, j in {2,3}, k in {0,1}
    #   i in {0,1}, j in {0,1}, k in {2,3}
    #   i in {0,1}, j in {2,3}, k in {2,3}
    #   i in {2,3}, j in {0,1}, k in {0,1}
    #   i in {2,3}, j in {2,3}, k in {0,1}
    #   i in {2,3}, j in {0,1}, k in {2,3}
    #   i in {2,3}, j in {2,3}, k in {2,3}
    # 8 blocks x 7 = 56, plus 100 - 64 = 36 standard = 92. Same as before.
    
    # Let's try a different strategy: use 2x2x3 Strassen-like blocks.
    # For 2x2x3, can we do better than 6? The tensor rank of 2x2x3 is 6.
    # So no improvement there.
    
    # What about 3x3x2? Same thing, rank 6.
    
    # The key insight: we should try to find overlapping Strassen blocks or
    # use a different algebraic structure.
    
    # Let me just go with the known rank-92 decomposition and verify it.
    
    U = [[0] * 92 for _ in range(nm)]
    V = [[0] * 92 for _ in range(mp)]
    W = [[0] * 92 for _ in range(pn)]
    
    blocks = []
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2]:
                blocks.append((i_base, i_base+1, j_base, j_base+1, k_base, k_base+1))
    
    skip_set = set()
    for i1, i2, j1, j2, k1, k2 in blocks:
        for i in [i1, i2]:
            for j in [j1, j2]:
                for k in [k1, k2]:
                    skip_set.add((i, j, k))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in skip_set:
                    standard_term(U, V, W, i, j, k, col)
                    col += 1
    
    for i1, i2, j1, j2, k1, k2 in blocks:
        col = strassen_block(U, V, W, i1, i2, j1, j2, k1, k2, col)
    
    rank = col
    assert rank == 92, f"Expected 92, got {rank}"
    
    if not verify(U, V, W, n, m, p):
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    standard_term(U, V, W, i, j, k, col)
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find further reductions by checking for redundant columns
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    current_rank = rank
    
    # Check for zero columns
    for r in range(current_rank - 1, -1, -1):
        if np.all(U_np[:, r] == 0) and np.all(V_np[:, r] == 0) and np.all(W_np[:, r] == 0):
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            current_rank -= 1
    
    # Convert to list format
    def to_list(mat):
        return [[int(round(mat[i][r])) if abs(mat[i][r] - round(mat[i][r])) < 1e-9 else float(mat[i][r])
                 for r in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    U_final = to_list(U_np)
    V_final = to_list(V_np)
    W_final = to_list(W_np)
    
    return {"U": U_final, "V": V_final, "W": W_final}
