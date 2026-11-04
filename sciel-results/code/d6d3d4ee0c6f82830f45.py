"""Matrix multiplication 4x5x5 with Strassen-based structured tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication using Strassen's algorithm.
    
    For A (2x2) * B (2x2) = C (2x2):
    C[0,0] = M1 - M4 + M6 + M7
    C[0,1] = M3 + M4
    C[1,0] = M1 + M2
    C[1,1] = M5 - M2 + M3 + M7
    
    Where:
    M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1])*B[0,0]
    M3 = A[0,0]*(B[0,1]-B[1,1])
    M4 = A[1,1]*(B[1,0]-B[0,0])
    M5 = (A[0,0]+A[0,1])*B[1,1]
    M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    M7 = (A[1,1]-A[0,1])*(B[1,0]+B[1,1])
    
    Each term is (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    where indices are into the flattened A (row*2+col), B (row*2+col), C (col*2+row) spaces.
    """
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)  [using W[k*n+i] convention with n=2, so k*2+i]
    
    terms = []
    
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # U: A[0,0]=1, A[1,1]=1 -> indices 0,3
    # V: B[0,0]=1, B[1,1]=1 -> indices 0,3
    # W: C[0,0]=1, C[1,0]=1, C[0,1]=1, C[1,1]=1 with coefficients +1,+1,-1,-1
    # Wait, let me re-derive:
    # C[0,0] = M1 - M4 + M6 + M7
    # C[0,1] = M3 + M4
    # C[1,0] = M1 + M2
    # C[1,1] = M5 - M2 + M3 + M7
    #
    # M1 appears in: C[0,0] with +1, C[1,0] with +1
    # M2 appears in: C[1,0] with +1, C[1,1] with -1
    # M3 appears in: C[0,1] with +1, C[1,1] with +1
    # M4 appears in: C[0,0] with -1, C[0,1] with +1
    # M5 appears in: C[1,1] with +1
    # M6 appears in: C[0,0] with +1
    # M7 appears in: C[0,0] with +1, C[1,1] with +1
    
    # C indices (k*2+i where k=col, i=row):
    # C[0,0]: k=0,i=0 -> idx 0
    # C[0,1]: k=0,i=1 -> idx 1
    # C[1,0]: k=1,i=0 -> idx 2
    # C[1,1]: k=1,i=1 -> idx 3
    
    terms.append(([0, 3], [1, 1], [0, 3], [1, 1], [0, 2], [1, 1]))  # M1
    terms.append(([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]))         # M2
    terms.append(([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]))         # M3
    terms.append(([3], [1], [2, 0], [1, -1], [0, 1], [-1, 1]))        # M4
    terms.append(([0, 1], [1, 1], [3], [1], [3], [1]))                # M5
    terms.append(([2, 0], [1, -1], [0, 1], [1, 1], [0], [1]))         # M6
    terms.append(([3, 1], [1, -1], [2, 3], [1, 1], [0, 3], [1, 1]))   # M7
    
    return terms

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix using Strassen-based structured decomposition.
    
    Strategy: Split the 4x5x5 problem into 2x2x2 subblocks where possible,
    applying Strassen's algorithm to each 2x2x2 block to reduce rank from 8 to 7.
    Handle remaining terms with schoolbook multiplication.
    
    A is 4x5 (n=4 rows, m=5 cols)
    B is 5x5 (m=5 rows, p=5 cols)
    C is 4x5 (n=4 rows, p=5 cols)
    
    C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    We split indices into pairs:
    i in {0,1} or {2,3} (4 rows -> two pairs)
    j in {0,1} or {2,3} (5 cols -> two pairs + one remainder at j=4)
    k in {0,1} or {2,3} (5 cols -> two pairs + one remainder at k=4)
    
    For each combination of i-pair, j-pair, k-pair, we have a 2x2x2 subproblem
    that can use Strassen (rank 7 instead of 8).
    
    There are 2 * 2 * 2 = 8 such 2x2x2 blocks, each saving 1 multiplication.
    Total savings: 8, so rank = 100 - 8 = 92.
    
    The remaining terms involve j=4 or k=4 (or both), which we handle with schoolbook.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Collect all rank-1 terms
    terms = []
    
    # Define index pairs
    i_pairs = [(0, 1), (2, 3)]
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3)]
    
    # Track which (i,j,k) triples are covered by Strassen
    covered = set()
    
    # Apply Strassen to each 2x2x2 block
    for i0, i1 in i_pairs:
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # Get Strassen terms for this 2x2x2 block
                strassen_terms = _strassen_2x2x2()
                
                for (u_idx, u_coef, v_idx, v_coef, w_idx, w_coef) in strassen_terms:
                    # Map local indices to global
                    # A local: 0=(i0,j0), 1=(i0,j1), 2=(i1,j0), 3=(i1,j1)
                    a_local_to_global = {
                        0: i0 * m + j0,
                        1: i0 * m + j1,
                        2: i1 * m + j0,
                        3: i1 * m + j1
                    }
                    
                    # B local: 0=(j0,k0), 1=(j0,k1), 2=(j1,k0), 3=(j1,k1)
                    b_local_to_global = {
                        0: j0 * p + k0,
                        1: j0 * p + k1,
                        2: j1 * p + k0,
                        3: j1 * p + k1
                    }
                    
                    # C local: 0=(k0,i0), 1=(k0,i1), 2=(k1,i0), 3=(k1,i1)
                    # W index: k*n + i
                    c_local_to_global = {
                        0: k0 * n + i0,
                        1: k0 * n + i1,
                        2: k1 * n + i0,
                        3: k1 * n + i1
                    }
                    
                    global_u = [a_local_to_global[idx] for idx in u_idx]
                    global_v = [b_local_to_global[idx] for idx in v_idx]
                    global_w = [c_local_to_global[idx] for idx in w_idx]
                    
                    terms.append((global_u, u_coef, global_v, v_coef, global_w, w_coef))
                
                # Mark all (i,j,k) in this block as covered
                for i in [i0, i1]:
                    for j in [j0, j1]:
                        for k in [k0, k1]:
                            covered.add((i, j, k))
    
    # Handle remaining (i,j,k) triples not covered by Strassen
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    # Schoolbook term: A[i,j] * B[j,k] -> C[i,k]
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    terms.append(([i * m + j], [1],
                                  [j * p + k], [1],
                                  [k * n + i], [1]))
    
    # Build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (u_idx, u_coef, v_idx, v_coef, w_idx, w_coef) in enumerate(terms):
        for idx, coef in zip(u_idx, u_coef):
            U[idx][t] = coef
        for idx, coef in zip(v_idx, v_coef):
            V[idx][t] = coef
        for idx, coef in zip(w_idx, w_coef):
            W[idx][t] = coef
    
    return {"U": U, "V": V, "W": W}
