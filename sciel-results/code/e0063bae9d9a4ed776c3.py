"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (18, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    rank = 32
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Define the 16 terms for 2x3x3 block
    # Local indices:
    # U: i in {0,1}, j in {0,1,2} -> idx = i*3+j
    # V: j in {0,1,2}, k in {0,1,2} -> idx = j*3+k
    # W: k in {0,1,2}, i in {0,1} -> idx = k*2+i
    
    terms = [
        # 1-9: Cover C_0k
        {'u': {0: 1}, 'v': {0: 1}, 'w': {0: 1}},      # A00 B00 -> C00
        {'u': {0: 1}, 'v': {1: 1}, 'w': {1: 1}},      # A00 B01 -> C01
        {'u': {0: 1}, 'v': {2: 1}, 'w': {2: 1}},      # A00 B02 -> C02
        {'u': {1: 1}, 'v': {3: 1}, 'w': {0: 1}},      # A01 B10 -> C00
        {'u': {1: 1}, 'v': {4: 1}, 'w': {1: 1}},      # A01 B11 -> C01
        {'u': {1: 1}, 'v': {5: 1}, 'w': {2: 1}},      # A01 B12 -> C02
        {'u': {2: 1}, 'v': {6: 1}, 'w': {0: 1}},      # A02 B20 -> C00
        {'u': {2: 1}, 'v': {7: 1}, 'w': {1: 1}},      # A02 B21 -> C01
        {'u': {2: 1}, 'v': {8: 1}, 'w': {2: 1}},      # A02 B22 -> C02
        # 10-12: Cover A10 B0k parts of C_1k
        {'u': {3: 1}, 'v': {0: 1}, 'w': {3: 1}},      # A10 B00 -> C10
        {'u': {3: 1}, 'v': {1: 1}, 'w': {4: 1}},      # A10 B01 -> C11
        {'u': {3: 1}, 'v': {2: 1}, 'w': {5: 1}},      # A10 B02 -> C12
        # 13-16: Cover remaining A11, A12 parts
        {'u': {4: 1}, 'v': {3: 1, 4: 1}, 'w': {3: 1, 4: 1}}, # A11 (B10+B11) -> C10+C11
        {'u': {4: 1}, 'v': {5: 1}, 'w': {5: 1}},      # A11 B12 -> C12
        {'u': {5: 1}, 'v': {6: 1, 7: 1}, 'w': {3: 1, 4: 1}}, # A12 (B20+B21) -> C10+C11
        {'u': {5: 1}, 'v': {8: 1}, 'w': {5: 1}},      # A12 B22 -> C12
    ]
    
    for block in range(2):
        for t_idx, term in enumerate(terms):
            col = block * 16 + t_idx
            
            for idx, val in term['u'].items():
                U[idx, col] = val
            
            for l_idx, val in term['v'].items():
                j_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = j_local * p + (block * 3 + k_local)
                V[g_idx, col] = val
            
            for l_idx, val in term['w'].items():
                i_local = l_idx % 2
                k_local = l_idx // 2
                g_idx = (block * 3 + k_local) * n + i_local
                W[g_idx, col] = val

    # Verify solution
    def verify_solution(sol):
        U = np.array(sol['U'])
        V = np.array(sol['V'])
        W = np.array(sol['W'])
        
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    T[i, j, k] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
        
        return np.allclose(T, 1.0, atol=1e-9)

    best_solution = {'U': U, 'V': V, 'W': W}
    best_rank = rank

    if not verify_solution(best_solution):
        # Fallback to trivial solution if needed
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        best_solution = {'U': U, 'V': V, 'W': W}
        best_rank = rank

    return {
        'U': best_solution['U'].tolist(),
        'V': best_solution['V'].tolist(),
        'W': best_solution['W'].tolist()
    }
# EVOLVE-BLOCK-END
