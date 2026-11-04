"""Matrix multiplication 2,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    deadline = time.time() + 140  # Leave margin before 150s kill
    
    def build_rank28():
        R = 28
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for i in u_indices:
                U[i][col] += 1
            for j in v_indices:
                V[j][col] += 1
            for k in w_indices:
                W[k][col] += 1
            col += 1
            
        def add_strassen_term(u_map, v_map, w_map):
            nonlocal col
            for (i, j), c in u_map.items():
                if c != 0:
                    U[i*m+j][col] += c
            for (j, k), c in v_map.items():
                if c != 0:
                    V[j*p+k][col] += c
            for (k, i), c in w_map.items():
                if c != 0:
                    W[k*n+i][col] += c
            col += 1

        # 1. A2 B2 (j=2): rank 10
        for i in range(2):
            for k in range(5):
                add_term([i*m+2], [2*p+k], [k*n+i])
                
        # 2. A1 B1a (j=0,1; k=0,1): Strassen rank 7
        add_strassen_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
        add_strassen_term({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1})
        add_strassen_term({(0,0):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (1,1):1})
        add_strassen_term({(1,1):1}, {(1,1):1, (0,1):-1}, {(0,0):1, (0,1):1})
        add_strassen_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})
        add_strassen_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
        add_strassen_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        
        # 3. A1 B1b (j=0,1; k=2,3): Strassen rank 7
        add_strassen_term({(0,0):1, (1,1):1}, {(0,2):1, (1,3):1}, {(2,0):1, (3,1):1})
        add_strassen_term({(1,0):1, (1,1):1}, {(0,2):1}, {(2,1):1, (3,1):-1})
        add_strassen_term({(0,0):1}, {(1,2):1, (0,2):-1}, {(3,0):1, (3,1):1})
        add_strassen_term({(1,1):1}, {(1,3):1, (0,3):-1}, {(2,0):1, (2,1):1})
        add_strassen_term({(0,0):1, (0,1):1}, {(1,3):1}, {(2,0):-1, (3,0):1})
        add_strassen_term({(1,0):1, (0,0):-1}, {(0,2):1, (0,3):1}, {(3,1):1})
        add_strassen_term({(0,1):1, (1,1):-1}, {(1,2):1, (1,3):1}, {(2,0):1})
        
        # 4. A1 B1c (j=0,1; k=4): Schoolbook rank 4
        add_term([0], [4], [8])
        add_term([1], [9], [8])
        add_term([2], [4], [9])
        add_term([3], [9], [9])
        
        return U, V, W, R
    
    def verify(U_mat, V_mat, W_mat, rank):
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        
        # Check non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U_np[a, r] * V_np[b, r] * W_np[c, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check zero entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    jj = b // p
                    k = b % p
                    kk = c // n
                    ii = c % n
                    if not (j == jj and i == ii and k == kk):
                        val = 0.0
                        for r in range(rank):
                            val += U_np[a, r] * V_np[b, r] * W_np[c, r]
                        if abs(val) > 1e-9:
                            return False
        return True
    
    # Build the rank-28 decomposition
    U, V, W, R = build_rank28()
    
    best_rank = R
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    if not verify(U, V, W, R):
        # Fallback to schoolbook rank 30
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        R = rank_schoolbook
        best_rank = R
        best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Time check
    if time.time() >= deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
