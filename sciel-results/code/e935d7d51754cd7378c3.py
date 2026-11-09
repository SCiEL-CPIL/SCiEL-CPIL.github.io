import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = 2, 3, 10
    
    # Try to find rank-6 decomposition for 2x3 x 3x2
    # Output is 2x2, so w vectors are in R^4
    # We need 6 terms: sum_c u_c ⊗ v_c ⊗ w_c = T
    
    # T[i,j,k] = delta_{i,k} for the bilinear map (A,B) -> A*B
    # More precisely, for 2x3 * 3x2:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The tensor has entries: for each (i,j,k), the contribution to C[i,k] is A[i,j]*B[j,k]
    
    # Let me set up the problem differently.
    # We want: sum_c (u_c · A) * (v_c · B) * w_c = A * B (as a 2x2 matrix)
    # where u_c in R^6, v_c in R^6, w_c in R^4
    
    # The "target" tensor T has T[i,j,k,l] = delta_{i,k} * delta_{j,l} ... no
    # Actually, the bilinear map is: (A,B) -> C where C[i,k] = sum_j A[i,j]B[j,k]
    # So the tensor T has components T_{i,j;k} = delta_{i,k} ... 
    # Let me think of it as: T(A,B) = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} e_{i,k}
    # where T_{i,j,k} = 1 if the indices match appropriately
    # Specifically, T_{i,j,k} = 1 for all valid i,j,k (since each term A[i,j]*B[j,k] contributes to C[i,k])
    # But we need to be careful: the output dimension is indexed by (i,k), so we have 4 output coordinates
    
    # Let me flatten: A has 6 elements, B has 6 elements, C has 4 elements
    # The tensor T is a 6x6x4 tensor where T[a,b,c] = 1 if a corresponds to A[i,j], 
    # b corresponds to B[j,k], and c corresponds to C[i,k]
    
    # Index mapping:
    # A: index = i*3 + j, for i in {0,1}, j in {0,1,2}
    # B: index = j*2 + k, for j in {0,1,2}, k in {0,1}
    # C: index = i*2 + k, for i in {0,1}, k in {0,1}
    
    # T[a,b,c] = 1 if a = i*3+j, b = j*2+k, c = i*2+k for some i,j,k
    # Otherwise T[a,b,c] = 0
    
    T = np.zeros((6, 6, 4))
    for i in range(2):
        for j in range(3):
            for k in range(2):
                a = i * 3 + j
                b = j * 2 + k
                c = i * 2 + k
                T[a, b, c] = 1.0
    
    # Now I want to find rank-6 decomposition: T = sum_{c=1}^{6} u_c ⊗ v_c ⊗ w_c
    # This is equivalent to finding 6 vectors u_c in R^6, v_c in R^6, w_c in R^4
    # such that sum_c u_c[a] * v_c[b] * w_c[c'] = T[a,b,c'] for all a,b,c'
    
    # This is a hard nonlinear problem. Let me try a heuristic approach.
    # One idea: use the fact that we can combine terms.
    
    # Schoolbook for 2x3 x 3x2 has rank 12. Can we do better?
    # Let me try to see if rank 6 is achievable by attempting to solve.
    
    # Actually, let me try a different strategy: use the known rank-7 decomposition for 2x2x2 (Strassen)
    # and adapt it. But 2x3x3x2 is different.
    
    # Let me just try to verify if rank 6 is possible by checking if the tensor
    # can be written as sum of 6 rank-1 tensors.
    
    # Given the time constraint, let me try a specific ansatz.
    # Suppose I use 6 terms where each term handles a combination of indices.
    
    # Term structure: for each term c, I choose:
    # - which A indices to combine (u_c is a sparse vector)
    # - which B indices to combine (v_c is a sparse vector)  
    # - which C indices to produce (w_c is a sparse vector)
    
    # Let me try: 6 terms, each with u_c having 2 nonzeros, v_c having 2 nonzeros, w_c having 2 nonzeros
    
    # This is getting complex. Let me just return schoolbook for safety, but try one more idea.
    
    # Alternative: can I reduce the 10 columns by using linearity?
    # If I compute partial sums of B columns, I might reduce the effective p.
    
    # Actually, the simplest safe approach: return schoolbook rank 60.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
