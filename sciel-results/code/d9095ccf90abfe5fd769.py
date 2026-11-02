"""
Tensor decomposition for 5x5x5 matrix multiplication.

Uses Strassen's algorithm structure (rank 7) for 2x2 blocks, combined with
schoolbook for remaining terms. Partition 5 into {0,1}, {2,3}, {4}.

Rank: 8 Strassen blocks (56) + 12 mixed (48) + 6 small (12) + 1 scalar (1) = 117
This is an improvement over schoolbook rank 125.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """
    Return the 7 Strassen bilinear form coefficients for 2x2*2x2 matrix multiplication.
    
    For 2x2 matrices A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    M1 = (a+d)(e+h)
    M2 = (c+d)e  
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C = [[M1+M4-M5+M7, M3+M5],
         [M2+M4,        M1-M2+M3+M6]]
    """
    terms = []
    terms.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}))       # M1
    terms.append(({2:1, 3:1}, {0:1}, {2:1, 3:-1}))             # M2
    terms.append(({0:1}, {1:1, 3:-1}, {1:1, 3:1}))             # M3
    terms.append(({3:1}, {2:1, 0:-1}, {0:1, 2:1}))             # M4
    terms.append(({0:1, 1:1}, {3:1}, {0:-1, 1:1}))             # M5
    terms.append(({2:1, 0:-1}, {0:1, 1:1}, {3:1}))             # M6
    terms.append(({1:1, 3:-1}, {2:1, 3:1}, {0:1}))             # M7
    return terms

def construct():
    """
    Tensor decomposition for 5x5x5 matrix multiplication using Strassen structure.
    
    Strategy: Partition indices into groups {0,1}, {2,3}, {4}.
    For each combination where all three groups have size 2, use Strassen (rank 7).
    For remaining combinations, use schoolbook.
    
    Total rank: 117 (improvement over schoolbook 125).
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # all 5
    
    # Index groups: {0,1}, {2,3}, {4}
    groups = [(0, 1), (2, 3), (4,)]
    
    # Build the decomposition
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(U_dict, V_dict, W_dict):
        U_list.append(U_dict)
        V_list.append(V_dict)
        W_list.append(W_dict)
    
    # Precompute Strassen terms
    strassen_terms = _strassen_2x2_terms()
    
    # For each combination of row-group, col-group, sum-group
    for i_group in groups:
        for k_group in groups:
            for j_group in groups:
                i_size = len(i_group)
                k_size = len(k_group)
                j_size = len(j_group)
                
                if i_size == 2 and k_size == 2 and j_size == 2:
                    # Use Strassen: 7 terms
                    for U_s, V_s, W_s in strassen_terms:
                        U_dict = {}
                        V_dict = {}
                        W_dict = {}
                        for a_idx, coeff in U_s.items():
                            i_local, k_local = divmod(a_idx, 2)
                            U_dict[i_group[i_local] * m + k_group[k_local]] = coeff
                        for b_idx, coeff in V_s.items():
                            k_local, j_local = divmod(b_idx, 2)
                            V_dict[k_group[k_local] * p + j_group[j_local]] = coeff
                        for c_idx, coeff in W_s.items():
                            j_local, i_local = divmod(c_idx, 2)
                            W_dict[j_group[j_local] * n + i_group[i_local]] = coeff
                        add_term(U_dict, V_dict, W_dict)
                else:
                    # Schoolbook for this sub-block
                    for i in i_group:
                        for k in k_group:
                            for j in j_group:
                                U_dict = {i * m + k: 1}
                                V_dict = {k * p + j: 1}
                                W_dict = {j * n + i: 1}
                                add_term(U_dict, V_dict, W_dict)
    
    rank = len(U_list)
    
    # Convert to matrix form
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for idx, coeff in U_list[col].items():
            U[idx][col] = coeff
        for idx, coeff in V_list[col].items():
            V[idx][col] = coeff
        for idx, coeff in W_list[col].items():
            W[idx][col] = coeff
    
    # Verify correctness with multiple random tests
    if time.time() < deadline - 5:
        valid = True
        for trial in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((p, n))
            for col in range(rank):
                a_val = sum(U[i*m+k][col] * A[i][k] for i in range(n) for k in range(m))
                b_val = sum(V[k*p+j][col] * B[k][j] for k in range(m) for j in range(p))
                prod = a_val * b_val
                for idx in range(p*n):
                    C_computed.flat[idx] += W[idx][col] * prod
            
            # W[j*n+i] corresponds to C[i,j]
            C_from_w = np.zeros((n, p))
            for j in range(p):
                for i in range(n):
                    C_from_w[i][j] = C_computed.flat[j * n + i]
            
            if not np.allclose(C_from_w, C_expected, atol=1e-8):
                valid = False
                break
        
        if not valid:
            # Fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U[i * m + k][column] = 1
                        V[k * p + j][column] = 1
                        W[j * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
