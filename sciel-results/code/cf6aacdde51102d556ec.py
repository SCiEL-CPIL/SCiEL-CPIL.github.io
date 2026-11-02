import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Target rank for 2x2x1 is 3. Total rank 33.
    # Fallback rank 44.
    
    # Define the tensor for one column (2x2x1)
    # A is 2x2, B is 2x1, C is 2x1
    # T[a, b, c] = 1 if A_a B_b contributes to C_c
    # a: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # b: 0, 1
    # c: 0, 1
    T = np.zeros((4, 2, 2))
    T[0, 0, 0] = 1.0 # A00 B0 -> C0
    T[1, 1, 0] = 1.0 # A01 B1 -> C0
    T[2, 0, 1] = 1.0 # A10 B0 -> C1
    T[3, 1, 1] = 1.0 # A11 B1 -> C1
    
    rank_single = 3
    U_single = None
    V_single = None
    W_single = None
    
    # Try to find rank-3 decomposition
    # Parameters: U (4x3), V (2x3), W (2x3)
    # Flatten to vector
    x0 = np.random.rand(4*3 + 2*3 + 2*3) * 2 - 1
    
    def objective(x):
        U = x[0:12].reshape(4, 3)
        V = x[12:18].reshape(2, 3)
        W = x[18:24].reshape(2, 3)
        
        # Reconstruct tensor
        # T_recon[a, b, c] = sum_s U[a, s] V[b, s] W[c, s]
        T_recon = np.einsum('as,bs,cs->abc', U, V, W)
        return np.sum((T - T_recon)**2)
    
    best_x = None
    best_val = np.inf
    
    # Run a few random restarts
    for i in range(10):
        if time.time() > deadline - 10:
            break
        x0 = np.random.rand(24) * 4 - 2
        res = minimize(objective, x0, method='BFGS', options={'maxiter': 1000})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
        if best_val < 1e-10:
            break
            
    if best_val < 1e-10:
        U_single = best_x[0:12].reshape(4, 3)
        V_single = best_x[12:18].reshape(2, 3)
        W_single = best_x[18:24].reshape(2, 3)
        # Clean up small values
        U_single[np.abs(U_single) < 1e-9] = 0
        V_single[np.abs(V_single) < 1e-9] = 0
        W_single[np.abs(W_single) < 1e-9] = 0
        # Round to nearest integer or half-integer if close
        # The problem allows integer/half-integer
        for mat in [U_single, V_single, W_single]:
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    val = mat[r, c]
                    # Check if close to integer or half-integer
                    nearest_int = round(val)
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_int) < 1e-6:
                        mat[r, c] = nearest_int
                    elif abs(val - nearest_half) < 1e-6:
                        mat[r, c] = nearest_half
    else:
        # Fallback to rank 4
        U_single = np.eye(4)
        V_single = np.eye(2)
        # W needs to map (A*B) to C
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # If U=I, V=I, then M_s = A_s B_s? No.
        # If U=I (4x4), V=I (2x2), W?
        # M_0 = A00 B0, M_1 = A01 B0, M_2 = A10 B0, M_3 = A11 B0
        # M_4 = A00 B1, M_5 = A01 B1, M_6 = A10 B1, M_7 = A11 B1
        # This is rank 8.
        # Schoolbook rank 4:
        # M0 = A00 B0
        # M1 = A01 B1
        # M2 = A10 B0
        # M3 = A11 B1
        # U:
        # col 0: (1,0,0,0)
        # col 1: (0,1,0,0)
        # col 2: (0,0,1,0)
        # col 3: (0,0,0,1)
        # V:
        # col 0: (1,0)
        # col 1: (0,1)
        # col 2: (1,0)
        # col 3: (0,1)
        # W:
        # C0 = M0 + M1
        # C1 = M2 + M3
        # W[0, 0] = 1, W[0, 1] = 1
        # W[1, 2] = 1, W[1, 3] = 1
        U_single = np.zeros((4, 4))
        V_single = np.zeros((2, 4))
        W_single = np.zeros((2, 4))
        U_single[0, 0] = 1
        U_single[1, 1] = 1
        U_single[2, 2] = 1
        U_single[3, 3] = 1
        V_single[0, 0] = 1
        V_single[1, 1] = 1
        V_single[0, 2] = 1
        V_single[1, 3] = 1
        W_single[0, 0] = 1
        W_single[0, 1] = 1
        W_single[1, 2] = 1
        W_single[1, 3] = 1
        rank_single = 4

    total_rank = rank_single * p
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for j in range(p):
        offset = j * rank_single
        # U is same for all j
        U[:, offset:offset+rank_single] = U_single
        
        # V: rows j*m to j*m+m? No.
        # V index: k*p + j?
        # Seed: V[j*p+k]. Here j is row of B (0..1), k is col of B (0..10).
        # So for col j of B, the rows are j*11? No.
        # B is 2x11.
        # Row 0 of B: indices 0, 11, 22...?
        # Seed: V[j*p+k].
        # If j=0 (row 0 of B), k=0..10. Indices 0, 1, ..., 10.
        # If j=1 (row 1 of B), k=0..10. Indices 11, 12, ..., 21.
        # So for a specific column k (of B), the rows are k and k+11.
        # In my loop, j is the column index of B (0..10).
        # So I need to put V_single into rows j and j+11.
        V[j, offset:offset+rank_single] = V_single[0]
        V[j+p, offset:offset+rank_single] = V_single[1]
        
        # W: index k*n + i.
        # k is col of C (0..10), i is row of C (0..1).
        # So for col j of C, rows are j*2 and j*2+1.
        W[j*n, offset:offset+rank_single] = W_single[0]
        W[j*n+1, offset:offset+rank_single] = W_single[1]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
