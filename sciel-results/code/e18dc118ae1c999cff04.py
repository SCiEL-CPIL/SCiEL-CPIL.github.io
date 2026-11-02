import time
import numpy as np
import itertools

DIMENSIONS = [3, 5, 7]

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_solution(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
    # Check that entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for v in row:
                if v is not None:
                    v2 = v * 2
                    if abs(v2 - round(v2)) > 1e-9:
                        return False
    # Verify the tensor equation: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-8:
                    return False
    return True

def _verify_solution_fast(U, V, W):
    """Fast verification using numpy."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # We can vectorize: for fixed (i,j), the equation over k is:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
    
    # Let's just do it with numpy broadcasting for speed
    # U[i*m+j] is shape (rank,)
    # V[j*p+k] is shape (rank,)
    # W[k*n+i] is shape (rank,)
    
    for i in range(n):
        for j in range(m):
            u_row = Ua[i*m+j]  # (rank,)
            for k in range(p):
                v_row = Va[j*p+k]  # (rank,)
                w_row = Wa[k*n+i]  # (rank,)
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1) > 1e-8:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin within 150s
    
    # Schoolbook decomposition (Rank 105)
    best_solution = _build_schoolbook()
    best_rank = n * m * p  # 105
    
    # Theoretical lower bound
    lower_bound = max(n * m, m * p, p * n)  # 35
    
    # Strategy: search from rank 104 down to lower_bound
    # For each rank, try random sparse U, V with entries in {-1, 0, 1}
    # Then solve for W using least squares and round to half-integers
    
    # Precompute index mappings for efficiency
    # For equation (i, j, k): eq_idx = i*m*p + j*p + k
    # Unknown (k, i, r): unk_idx = (k*n + i) * R + r
    
    for R in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline - 10:
            break
        
        # Adaptive number of trials
        # Higher ranks (closer to 105) get more trials
        if R >= 100:
            trials = 50
        elif R >= 90:
            trials = 30
        elif R >= 70:
            trials = 20
        else:
            trials = 10
        
        for t in range(trials):
            if time.time() > deadline - 10:
                break
            
            # Generate random U and V with entries in {-1, 0, 1}
            # Use sparse structure: each row has few non-zeros
            U_rand = np.random.randint(-1, 2, size=(n * m, R))
            V_rand = np.random.randint(-1, 2, size=(m * p, R))
            
            # Construct the linear system for W
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is linear in W entries
            # A is (n*m*p) x (p*n*R)
            # b is (n*m*p) all ones
            
            n_eq = n * m * p
            n_unk = p * n * R
            A = np.zeros((n_eq, n_unk))
            
            # Fill A efficiently
            for i in range(n):
                for j in range(m):
                    u_row = U_rand[i * m + j]  # (R,)
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        v_row = V_rand[j * p + k]  # (R,)
                        coeff = u_row * v_row  # (R,)
                        # Non-zero entries of coeff
                        nz = np.nonzero(coeff)[0]
                        if len(nz) > 0:
                            for r in nz:
                                unk_idx = (k * n + i) * R + r
                                A[eq_idx, unk_idx] = coeff[r]
            
            # Solve A @ w = b
            try:
                b = np.ones(n_eq)
                w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Round to nearest half-integer
                w_rounded = np.round(w_sol * 2) / 2
                
                # Verify: A @ w_rounded should equal b
                if np.allclose(A @ w_rounded, b, atol=1e-8):
                    # Construct W from w_rounded
                    W_list = [[0] * R for _ in range(p * n)]
                    for idx in range(n_unk):
                        ki = idx // R
                        r = idx % R
                        k_val = ki // n
                        i_val = ki % n
                        W_list[k_val * n + i_val][r] = w_rounded[idx]
                    
                    # Final verification
                    if _verify_solution_fast(U_rand.tolist(), V_rand.tolist(), W_list):
                        return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_list}
                        
            except Exception:
                continue
    
    return best_solution
# EVOLVE-BLOCK-END
