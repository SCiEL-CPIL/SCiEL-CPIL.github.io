import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s margin
    
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    # Part 1: A1 B11 (4x4x4) using 8 Strassen 2x2x2 blocks
    # A1 is A[0:4, 0:4], B11 is B[0:4, 0:4], C11 is C[0:4, 0:4]
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                if time.time() - start_time > DEADLINE:
                    break
                # Strassen 2x2x2
                # Local indices il, jl, kl in 0..1
                # Global i = 2*bi + il
                # Global j = 2*bj + jl
                # Global k = 2*bk + kl
                
                def gI(il, jl): return (2*bi + il) * m + (2*bj + jl)
                def gJ(jl, kl): return (2*bj + jl) * p + (2*bk + kl)
                def gK(kl, il): return (2*bk + kl) * n + (2*bi + il)
                
                # M1
                add_term(
                    [(gI(0,0), 1), (gI(1,1), 1)],
                    [(gJ(0,0), 1), (gJ(1,1), 1)],
                    [(gK(0,0), 1), (gK(1,1), 1)]
                )
                # M2
                add_term(
                    [(gI(1,0), 1), (gI(1,1), 1)],
                    [(gJ(0,0), 1)],
                    [(gK(0,1), 1), (gK(1,1), -1)]
                )
                # M3
                add_term(
                    [(gI(0,0), 1)],
                    [(gJ(0,1), 1), (gJ(1,1), -1)],
                    [(gK(1,0), 1), (gK(1,1), 1)]
                )
                # M4
                add_term(
                    [(gI(1,1), 1)],
                    [(gJ(1,0), 1), (gJ(0,0), -1)],
                    [(gK(0,0), 1), (gK(0,1), 1)]
                )
                # M5
                add_term(
                    [(gI(0,0), 1), (gI(0,1), 1)],
                    [(gJ(1,1), 1)],
                    [(gK(0,0), -1), (gK(1,0), 1)]
                )
                # M6
                add_term(
                    [(gI(1,0), 1), (gI(0,0), -1)],
                    [(gJ(0,0), 1), (gJ(0,1), 1)],
                    [(gK(1,1), 1)]
                )
                # M7
                add_term(
                    [(gI(0,1), 1), (gI(1,1), -1)],
                    [(gJ(1,0), 1), (gJ(1,1), 1)],
                    [(gK(0,0), 1)]
                )

    # Part 2: A1 B12 (4x4x1)
    # C[i, 4] += sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            add_term(
                [(i * m + j, 1)],
                [(j * p + 4, 1)],
                [(4 * n + i, 1)]
            )

    # Part 3: A2 B21 (4x1x4)
    # C[i, k] += A[i, 4] B[4, k]
    for i in range(4):
        for k in range(4):
            add_term(
                [(i * m + 4, 1)],
                [(4 * p + k, 1)],
                [(k * n + i, 1)]
            )

    # Part 4: A2 B22 (4x1x1)
    # C[i, 4] += A[i, 4] B[4, 4]
    for i in range(4):
        add_term(
            [(i * m + 4, 1)],
            [(4 * p + 4, 1)],
            [(4 * n + i, 1)]
        )

    if not U_cols:
        return {"U": [[0]*100 for _ in range(20)], "V": [[0]*100 for _ in range(25)], "W": [[0]*100 for _ in range(20)]}
        
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Optional: verify decomposition correctness (skip if time is tight)
    if time.time() - start_time < DEADLINE - 10:
        try:
            # Create test matrices
            A_test = np.random.rand(n, m)
            B_test = np.random.rand(m, p)
            C_expected = A_test @ B_test
            
            # Compute using decomposition
            C_computed = np.zeros((n, p))
            for r in range(U.shape[1]):
                u = U[:, r]
                v = V[:, r]
                w = W[:, r]
                
                # u is (n*m,), reshape to (n,m)
                u_mat = u.reshape(n, m)
                # v is (m*p,), reshape to (m,p)
                v_mat = v.reshape(m, p)
                # w is (p*n,), reshape to (p,n) then transpose to (n,p)
                w_mat = w.reshape(p, n).T
                
                C_computed += u_mat @ v_mat * w_mat
            
            # Check if close
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # If verification fails, return schoolbook as fallback
                U_school = np.zeros((n*m, n*m*p))
                V_school = np.zeros((m*p, n*m*p))
                W_school = np.zeros((p*n, n*m*p))
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_school[i*m+j, idx] = 1
                            V_school[j*p+k, idx] = 1
                            W_school[k*n+i, idx] = 1
                            idx += 1
                
                return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        except Exception:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
