import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Rank 189 decomposition using 3x3 blocks of 2x2 matrices
    # with Strassen's algorithm for each 2x2 block multiplication
    rank = 189
    
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    # Strassen definitions for 2x2 matrices
    # M1 = (P00 + P11)(Q00 + Q11)
    # M2 = (P10 + P11)Q00
    # M3 = P00(Q01 - Q11)
    # M4 = P11(Q10 - Q00)
    # M5 = (P00 + P01)Q11
    # M6 = (P10 - P00)(Q00 + Q01)
    # M7 = (P01 - P11)(Q10 + Q11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for U (A), V (B), W (C)
    # A is 6x6, B is 6x6, C is 6x6
    # Block indices I, J, K in 0..2
    # Local indices u, w, v in 0..1
    # Global i = 2*I + u, j = 2*J + w, k = 2*K + v
    
    # U index a = i*m + j = 6*i + j
    # V index b = j*p + k = 6*j + k
    # W index c = k*n + i = 6*k + i
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for r in range(7):
                    # Term r (0-based)
                    
                    # Helper to set U, V, W values
                    # U entries for P (A block I,J)
                    # V entries for Q (B block J,K)
                    # W entries for C (C block I,K)
                    
                    if r == 0:
                        # M1
                        # U: P00, P11
                        u_idx_1 = 6*(2*I+0) + (2*J+0)
                        u_idx_2 = 6*(2*I+1) + (2*J+1)
                        U[u_idx_1, col] = 1
                        U[u_idx_2, col] = 1
                        
                        # V: Q00, Q11
                        v_idx_1 = 6*(2*J+0) + (2*K+0)
                        v_idx_2 = 6*(2*J+1) + (2*K+1)
                        V[v_idx_1, col] = 1
                        V[v_idx_2, col] = 1
                        
                        # W: C00 (1), C11 (1)
                        w_idx_1 = 6*(2*K+0) + (2*I+0)
                        w_idx_2 = 6*(2*K+1) + (2*I+1)
                        W[w_idx_1, col] = 1
                        W[w_idx_2, col] = 1
                        
                    elif r == 1:
                        # M2
                        # U: P10, P11
                        u_idx_1 = 6*(2*I+1) + (2*J+0)
                        u_idx_2 = 6*(2*I+1) + (2*J+1)
                        U[u_idx_1, col] = 1
                        U[u_idx_2, col] = 1
                        
                        # V: Q00
                        v_idx_1 = 6*(2*J+0) + (2*K+0)
                        V[v_idx_1, col] = 1
                        
                        # W: C10 (1), C11 (-1)
                        w_idx_1 = 6*(2*K+0) + (2*I+1)
                        w_idx_2 = 6*(2*K+1) + (2*I+1)
                        W[w_idx_1, col] = 1
                        W[w_idx_2, col] = -1
                        
                    elif r == 2:
                        # M3
                        # U: P00
                        u_idx_1 = 6*(2*I+0) + (2*J+0)
                        U[u_idx_1, col] = 1
                        
                        # V: Q01, Q11
                        v_idx_1 = 6*(2*J+0) + (2*K+1)
                        v_idx_2 = 6*(2*J+1) + (2*K+1)
                        V[v_idx_1, col] = 1
                        V[v_idx_2, col] = -1
                        
                        # W: C01 (1), C11 (1)
                        w_idx_1 = 6*(2*K+1) + (2*I+0)
                        w_idx_2 = 6*(2*K+1) + (2*I+1)
                        W[w_idx_1, col] = 1
                        W[w_idx_2, col] = 1
                        
                    elif r == 3:
                        # M4
                        # U: P11
                        u_idx_1 = 6*(2*I+1) + (2*J+1)
                        U[u_idx_1, col] = 1
                        
                        # V: Q10, Q00
                        v_idx_1 = 6*(2*J+1) + (2*K+0)
                        v_idx_2 = 6*(2*J+0) + (2*K+0)
                        V[v_idx_1, col] = 1
                        V[v_idx_2, col] = -1
                        
                        # W: C00 (1), C10 (1)
                        w_idx_1 = 6*(2*K+0) + (2*I+0)
                        w_idx_2 = 6*(2*K+0) + (2*I+1)
                        W[w_idx_1, col] = 1
                        W[w_idx_2, col] = 1
                        
                    elif r == 4:
                        # M5
                        # U: P00, P01
                        u_idx_1 = 6*(2*I+0) + (2*J+0)
                        u_idx_2 = 6*(2*I+0) + (2*J+1)
                        U[u_idx_1, col] = 1
                        U[u_idx_2, col] = 1
                        
                        # V: Q11
                        v_idx_1 = 6*(2*J+1) + (2*K+1)
                        V[v_idx_1, col] = 1
                        
                        # W: C00 (-1), C01 (1)
                        w_idx_1 = 6*(2*K+0) + (2*I+0)
                        w_idx_2 = 6*(2*K+1) + (2*I+0)
                        W[w_idx_1, col] = -1
                        W[w_idx_2, col] = 1
                        
                    elif r == 5:
                        # M6
                        # U: P10, P00
                        u_idx_1 = 6*(2*I+1) + (2*J+0)
                        u_idx_2 = 6*(2*I+0) + (2*J+0)
                        U[u_idx_1, col] = 1
                        U[u_idx_2, col] = -1
                        
                        # V: Q00, Q01
                        v_idx_1 = 6*(2*J+0) + (2*K+0)
                        v_idx_2 = 6*(2*J+0) + (2*K+1)
                        V[v_idx_1, col] = 1
                        V[v_idx_2, col] = 1
                        
                        # W: C11 (1)
                        w_idx_1 = 6*(2*K+1) + (2*I+1)
                        W[w_idx_1, col] = 1
                        
                    elif r == 6:
                        # M7
                        # U: P01, P11
                        u_idx_1 = 6*(2*I+0) + (2*J+1)
                        u_idx_2 = 6*(2*I+1) + (2*J+1)
                        U[u_idx_1, col] = 1
                        U[u_idx_2, col] = -1
                        
                        # V: Q10, Q11
                        v_idx_1 = 6*(2*J+1) + (2*K+0)
                        v_idx_2 = 6*(2*J+1) + (2*K+1)
                        V[v_idx_1, col] = 1
                        V[v_idx_2, col] = 1
                        
                        # W: C00 (1)
                        w_idx_1 = 6*(2*K+0) + (2*I+0)
                        W[w_idx_1, col] = 1
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
