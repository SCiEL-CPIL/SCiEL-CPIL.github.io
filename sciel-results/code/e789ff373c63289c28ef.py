import numpy as np
import time

start_time = time.time()
TIME_LIMIT = 110.0

def construct():
    n, m, p = 2, 3, 9
    
    # Rank 48 decomposition using Strassen for 2x2 blocks and direct for the rest
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 27 rows
    W = [[0] * rank for _ in range(p * n)]  # 18 rows
    
    # A indices (2x3):
    # 0: A00, 1: A01, 2: A02
    # 3: A10, 4: A11, 5: A12
    
    col_idx = 0
    
    # Part 1: A[:, 0:2] * B[0:2, :]
    # For each pair of columns k0, k1 in B, use Strassen (7 mults)
    for b in range(4):  # pairs (0,1), (2,3), (4,5), (6,7)
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices for B rows 0,1 and cols k0,k1
        v_e = 0 * p + k0  # B_{0, k0}
        v_f = 0 * p + k1  # B_{0, k1}
        v_g = 1 * p + k0  # B_{1, k0}
        v_h = 1 * p + k1  # B_{1, k1}
        
        # U indices for A cols 0,1
        u_a = 0  # A_{00}
        u_b = 1  # A_{01}
        u_c = 3  # A_{10}
        u_d = 4  # A_{11}
        
        # Strassen M1..M7
        # M1 = (a+d)(e+h)
        U[u_a][col_idx] += 1
        U[u_d][col_idx] += 1
        V[v_e][col_idx] += 1
        V[v_h][col_idx] += 1
        
        # M2 = (c+d)e
        U[u_c][col_idx+1] += 1
        U[u_d][col_idx+1] += 1
        V[v_e][col_idx+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col_idx+2] += 1
        V[v_f][col_idx+2] += 1
        V[v_h][col_idx+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col_idx+3] += 1
        V[v_g][col_idx+3] += 1
        V[v_e][col_idx+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col_idx+4] += 1
        U[u_b][col_idx+4] += 1
        V[v_h][col_idx+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col_idx+5] += 1
        U[u_a][col_idx+5] -= 1
        V[v_e][col_idx+5] += 1
        V[v_f][col_idx+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col_idx+6] += 1
        U[u_d][col_idx+6] -= 1
        V[v_g][col_idx+6] += 1
        V[v_h][col_idx+6] += 1
        
        # W indices for C
        w_00 = k0 * n + 0  # C_{0, k0}
        w_01 = k1 * n + 0  # C_{0, k1}
        w_10 = k0 * n + 1  # C_{1, k0}
        w_11 = k1 * n + 1  # C_{1, k1}
        
        # C00 = M1 + M4 - M5 + M7
        W[w_00][col_idx] += 1
        W[w_00][col_idx+3] += 1
        W[w_00][col_idx+4] -= 1
        W[w_00][col_idx+6] += 1
        
        # C01 = M3 + M5
        W[w_01][col_idx+2] += 1
        W[w_01][col_idx+4] += 1
        
        # C10 = M2 + M4
        W[w_10][col_idx+1] += 1
        W[w_10][col_idx+3] += 1
        
        # C11 = M1 - M2 + M3 + M6
        W[w_11][col_idx] += 1
        W[w_11][col_idx+1] -= 1
        W[w_11][col_idx+2] += 1
        W[w_11][col_idx+5] += 1
        
        col_idx += 7
    
    # Handle column k=8 for A[:, 0:2] * B[0:2, 8] using 4 direct mults
    k8 = 8
    v_e8 = 0 * p + k8  # B_{0, 8}
    v_g8 = 1 * p + k8  # B_{1, 8}
    
    u_a = 0  # A00
    u_b = 1  # A01
    u_c = 3  # A10
    u_d = 4  # A11
    
    # M1 = A00 * B08
    U[u_a][col_idx] += 1
    V[v_e8][col_idx] += 1
    W[k8 * n + 0][col_idx] += 1
    
    # M2 = A01 * B18
    U[u_b][col_idx+1] += 1
    V[v_g8][col_idx+1] += 1
    W[k8 * n + 0][col_idx+1] += 1
    
    # M3 = A10 * B08
    U[u_c][col_idx+2] += 1
    V[v_e8][col_idx+2] += 1
    W[k8 * n + 1][col_idx+2] += 1
    
    # M4 = A11 * B18
    U[u_d][col_idx+3] += 1
    V[v_g8][col_idx+3] += 1
    W[k8 * n + 1][col_idx+3] += 1
    
    col_idx += 4
    
    # Part 2: A[:, 2:3] * B[2, :] using 2 direct mults per column
    for k in range(9):
        v_b = 2 * p + k  # B_{2, k}
        
        # M1 = A02 * B2k
        U[2][col_idx] += 1
        V[v_b][col_idx] += 1
        W[k * n + 0][col_idx] += 1
        
        # M2 = A12 * B2k
        U[5][col_idx+1] += 1
        V[v_b][col_idx+1] += 1
        W[k * n + 1][col_idx+1] += 1
        
        col_idx += 2
    
    return {"U": U, "V": V, "W": W}

# Verification
if __name__ == "__main__":
    res = construct()
    n, m, p = 2, 3, 9
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    C_flat = np.zeros(n * p)
    for s in range(res["U"][0].__len__()):
        us = np.array(res["U"][:, s])
        vs = np.array(res["V"][:, s])
        ws = np.array(res["W"][:, s])
        val = np.dot(us, A_flat) * np.dot(vs, B_flat)
        C_flat += val * ws
    
    C_actual = C_flat.reshape((p, n)).T
    
    if not np.allclose(C_actual, C_expected, atol=1e-6):
        print("Mismatch detected")
        print("Max diff:", np.max(np.abs(C_actual - C_expected)))
    else:
        print("Validation Passed, rank =", len(res["U"][0]))
