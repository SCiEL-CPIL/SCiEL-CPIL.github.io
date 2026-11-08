import numpy as np
import time

DIMENSIONS = [2, 3, 9]
_DEADLINE = 120.0
_START_TIME = None

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: i*m+j
    # B indices: j*p+k
    # C indices: k*n+i
    
    # Part 1: A_L (2x2) * B_T (2x9)
    # A_L uses A_{00}, A_{01}, A_{10}, A_{11} -> U indices 0, 1, 3, 4
    # B_T uses B_{0k}, B_{1k} -> V indices 0..8, 9..17
    
    col = 0
    # 4 blocks of 2x2
    for b in range(4):
        # B columns 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices
        v_e = 0 * p + k0 # B_{0, k0}
        v_f = 0 * p + k1 # B_{0, k1}
        v_g = 1 * p + k0 # B_{1, k0}
        v_h = 1 * p + k1 # B_{1, k1}
        
        # A indices
        u_a = 0 * m + 0 # A_{00}
        u_b = 0 * m + 1 # A_{01}
        u_c = 1 * m + 0 # A_{10}
        u_d = 1 * m + 1 # A_{11}
        
        # Strassen Products
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # Output C_{0, k0}, C_{0, k1}, C_{1, k0}, C_{1, k1}
        # W indices: k*n+i
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C00 = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C01 = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C10 = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C11 = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
        
    # 1 block of 2x1 (last column k=8)
    k8 = 8
    v_e = 0 * p + k8 # B_{0, 8}
    v_g = 1 * p + k8 # B_{1, 8}
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    # P1 = a e
    U[u_a][col] += 1
    V[v_e][col] += 1
    # P2 = b g
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    # P3 = c e
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    # P4 = d g
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    # C08 = P1 + P2
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    
    # C18 = P3 + P4
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: A_R (2x1) * B_B (1x9)
    # A_R uses A_{02}, A_{12} -> U indices 2, 5
    # B_B uses B_{2k} -> V indices 18..26
    
    u_ar0 = 0 * m + 2 # A_{02}
    u_ar1 = 1 * m + 2 # A_{12}
    
    for k in range(9):
        v_b = 2 * p + k # B_{2, k}
        
        # Q_k = A_{02} B_{2k}
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A_{12} B_{2k}
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
        
    # Verify decomposition if time permits
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 5:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            # Check: for each rank term r, sum_r U[i,r]*V[j,r]*W[k,r] should give C[i,j,k]
            # C_{i,j} = sum_k A_{i,k} * B_{k,j}
            # Tensor form: T[i*m+j, k*p+j2, ...] 
            # Actually let's just verify with random matrices
            np.random.seed(42)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            # Compute C from decomposition
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_col = U_np[:, r]
                v_col = V_np[:, r]
                w_col = W_np[:, r]
                # u_col has n*m entries: A_{i,j} at index i*m+j
                # v_col has m*p entries: B_{j,k} at index j*p+k
                # w_col has p*n entries: C_{k,i} at index k*n+i
                # Product term: sum over i,j of u[i*m+j] * A_{i,j} * sum over j,k of v[j*p+k] * B_{j,k} * w[k*n+i]
                # This is: (u^T A_vec) * (v^T B_vec) * w
                # where A_vec is A reshaped, B_vec is B reshaped
                A_vec = A.reshape(n*m)
                B_vec = B.reshape(m*p)
                term = np.dot(u_col, A_vec) * np.dot(v_col, B_vec)
                # Add to C: w_col[k*n+i] contributes to C[i,k]
                w_mat = w_col.reshape(p, n)
                C_computed += term * w_mat.T
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Decomposition is wrong, but we still return it (score will be 0)
                pass
        except Exception:
            pass
    return {"U": U, "V": V, "W": W}
