"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit

    n, m, p = 2, 2, 11

    # Build the rank-39 Strassen-based construction
    rank = 39  # 5 Strassen blocks (7 each) + 1 schoolbook column (4)

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0

    # 5 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1

        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]

        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]

        # m1 = (a+d)(e+h)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (c+d)e
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a(f-h)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = d(g-e)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a+b)h
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

        if time.time() > deadline:
            break

    # Last column k=10: schoolbook
    if r < rank:
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = 11 + k2 # B[1, 10]
        w_c02 = 2 * k2      # C[0, 10]
        w_c12 = 2 * k2 + 1  # C[1, 10]

        # a * B[0, 10]
        U[0, r] += 1
        V[v_e2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # b * B[1, 10]
        U[1, r] += 1
        V[v_g2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # c * B[0, 10]
        U[2, r] += 1
        V[v_e2, r] += 1
        W[w_c12, r] += 1
        r += 1

        # d * B[1, 10]
        U[3, r] += 1
        V[v_g2, r] += 1
        W[w_c12, r] += 1
        r += 1

    # Aggressive rank reduction: iteratively try to remove terms
    if time.time() < deadline - 20:
        U_f = U.astype(np.float64)
        V_f = V.astype(np.float64)
        W_f = W.astype(np.float64)
        
        # Build the target tensor
        target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    target[alpha, beta, gamma] = 1.0
        
        # Verify current decomposition
        current = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r_idx in range(rank):
            current += np.einsum('a,b,c->abc', U_f[:, r_idx], V_f[:, r_idx], W_f[:, r_idx])
        
        max_err = np.max(np.abs(current - target))
        if max_err <= 1e-10:
            # Iterative greedy rank reduction
            current_rank = rank
            current_U = U.copy()
            current_V = V.copy()
            current_W = W.copy()
            
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for r_remove in range(current_rank):
                    if time.time() > deadline - 10:
                        break
                    
                    # Compute residual by summing all terms except r_remove
                    residual = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                    for r_keep in range(current_rank):
                        if r_keep == r_remove:
                            continue
                        residual += np.einsum('a,b,c->abc', 
                                              current_U_f[:, r_keep] if hasattr(current_U, '_f') else current_U[:, r_keep].astype(np.float64),
                                              current_V[:, r_keep].astype(np.float64),
                                              current_W[:, r_keep].astype(np.float64))
                    
                    # Check if residual equals target
                    if np.max(np.abs(residual - target)) < 1e-10:
                        # Remove this term
                        current_rank -= 1
                        current_U = np.delete(current_U, r_remove, axis=1)
                        current_V = np.delete(current_V, r_remove, axis=1)
                        current_W = np.delete(current_W, r_remove, axis=1)
                        improved = True
                        break
            
            if current_rank < rank:
                U = current_U
                V = current_V
                W = current_W
                rank = current_rank

    # Convert back to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
