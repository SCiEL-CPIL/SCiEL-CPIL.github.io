import numpy as np
import time
from itertools import combinations

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication.
    
    This uses the recursive Strassen algorithm.
    The rank of 4x4 matrix multiplication is currently known to be at most 49.
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (16, 49), (16, 49), (16, 49)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 4
    N_A = n * m  # 16
    N_B = m * p  # 16
    N_C = p * n  # 16
    
    rank = 49
    U = np.zeros((N_A, rank))
    V = np.zeros((N_B, rank))
    W = np.zeros((N_C, rank))
    
    # Index mapping helpers
    def a_idx(r, c): return r * m + c
    def b_idx(r, c): return r * p + c
    def c_idx(r, c): return c * n + r  # Trace convention: W[k*n+i, r]
    
    # Define the 7 top-level Strassen terms for 4x4 (operating on 2x2 blocks)
    top_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # t=0
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),          # t=1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),          # t=2
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),          # t=3
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),          # t=4
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),          # t=5
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})           # t=6
    ]
    
    # Bottom-level Strassen terms
    bottom_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),          # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),          # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),          # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),          # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),          # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})           # M7
    ]
    
    # Build the 49-term decomposition
    for t in range(7):
        A_top, B_top, C_top = top_terms[t]
        
        A_v = [[{} for _ in range(2)] for _ in range(2)]
        B_v = [[{} for _ in range(2)] for _ in range(2)]
        
        for (br, bc), coeff in A_top.items():
            for r in range(2):
                for c in range(2):
                    a_r = 2*br + r
                    a_c = 2*bc + c
                    A_v[r][c][(a_r, a_c)] = A_v[r][c].get((a_r, a_c), 0) + coeff
        
        for (br, bc), coeff in B_top.items():
            for r in range(2):
                for c in range(2):
                    b_r = 2*br + r
                    b_c = 2*bc + c
                    B_v[r][c][(b_r, b_c)] = B_v[r][c].get((b_r, b_c), 0) + coeff
        
        for k in range(7):
            X_d, Y_d, Z_d = bottom_terms[k]
            
            X = {}
            for (r, c), coeff in X_d.items():
                for (a_r, a_c), a_coeff in A_v[r][c].items():
                    X[(a_r, a_c)] = X.get((a_r, a_c), 0) + coeff * a_coeff
            
            Y = {}
            for (r, c), coeff in Y_d.items():
                for (b_r, b_c), b_coeff in B_v[r][c].items():
                    Y[(b_r, b_c)] = Y.get((b_r, b_c), 0) + coeff * b_coeff
            
            Z = {}
            for (br, bc), c_block_coeff in C_top.items():
                for (r, c), z_coeff in Z_d.items():
                    c_r = 2*br + r
                    c_c = 2*bc + c
                    Z[(c_r, c_c)] = Z.get((c_r, c_c), 0) + c_block_coeff * z_coeff
            
            rank_idx = t * 7 + k
            
            for (a_r, a_c), coeff in X.items():
                U[a_idx(a_r, a_c), rank_idx] += coeff
            
            for (b_r, b_c), coeff in Y.items():
                V[b_idx(b_r, b_c), rank_idx] += coeff
            
            for (c_r, c_c), coeff in Z.items():
                W[c_idx(c_r, c_c), rank_idx] += coeff
    
    # Verify the decomposition
    T = np.zeros((N_A, N_B, N_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_i = i * m + j
                b_i = j * p + k
                c_i = k * n + i
                T[a_i, b_i, c_i] = 1.0
    
    T_recon = np.einsum('as,bs,cs->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    
    # Check if any rank-1 term is redundant by trying to eliminate each one
    # A term is redundant if it can be expressed as a linear combination of the others
    # in the tensor space spanned by the remaining terms
    
    best_rank = 49
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing each term and see if we can still represent the tensor
    # This is a heuristic: check if the removed term's contribution is in the span
    # of the other terms' contributions to the tensor
    
    # First, let's check for obvious linear dependencies among the U, V, W columns
    # If two terms have proportional U, V, W vectors, one might be redundant
    
    # Build the tensor from each rank-1 term
    term_tensors = []
    for s in range(49):
        term_t = np.outer(np.outer(U[:, s], V[:, s]), W[:, s]).reshape(N_A, N_B, N_C)
        term_tensors.append(term_t)
    
    # Check if any term is a linear combination of others
    # This is expensive, so we'll do a quick check first
    
    # Quick check: are any two terms identical up to scaling?
    for i in range(49):
        if time.time() > deadline:
            break
        for j in range(i+1, 49):
            # Check if term i is proportional to term j
            ni = np.linalg.norm(term_tensors[i])
            nj = np.linalg.norm(term_tensors[j])
            if ni > 1e-10 and nj > 1e-10:
                dot = np.sum(term_tensors[i] * term_tensors[j])
                if abs(dot) > 0.99 * ni * nj:
                    # Terms are nearly identical, try removing one
                    pass
    
    # More systematic approach: try to find a subset of terms that spans the tensor
    # This is a set cover / basis problem, which is NP-hard in general
    # For now, we'll just return the 49-term decomposition
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
