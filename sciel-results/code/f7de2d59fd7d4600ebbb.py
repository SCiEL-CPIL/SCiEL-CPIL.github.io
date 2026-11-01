"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (150s kill limit)
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # First, establish the schoolbook rank-36 decomposition as a fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Build the target tensor T
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_full_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Full verification: check all entries of the tensor."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check all entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = np.dot(U_np[a, :], V_np[b, :] * W_np[c, :])
                    expected = 1.0 if (a % m == b // p and a // m == c % n and b % p == c // n) else 0.0
                    if abs(val - expected) > tol:
                        return False
        return True
    
    def compute_error_fast(U, V, W, T):
        """Compute the Frobenius norm of the error efficiently using einsum."""
        val = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((val - T) ** 2)
        return np.sqrt(error)
    
    def solve_for_W(U, V, T, n, m, p):
        """Given U and V, solve for W using least squares."""
        r = U.shape[1]
        W = np.zeros((p*n, r))
        
        for c in range(p*n):
            A = np.zeros((n*m * m*p, r))
            rhs = T[:, :, c].flatten()
            
            idx = 0
            for a in range(n*m):
                for b in range(m*p):
                    A[idx, :] = U[a, :] * V[b, :]
                    idx += 1
            
            W_c, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
            W[c, :] = W_c
        
        return W
    
    def solve_for_U(V, W, T, n, m, p):
        """Given V and W, solve for U using least squares."""
        r = V.shape[1]
        U = np.zeros((n*m, r))
        
        for a in range(n*m):
            A = np.zeros((m*p * p*n, r))
            rhs = T[a, :, :].flatten()
            
            idx = 0
            for b in range(m*p):
                for c in range(p*n):
                    A[idx, :] = V[b, :] * W[c, :]
                    idx += 1
            
            U_a, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
            U[a, :] = U_a
        
        return U
    
    def solve_for_V(U, W, T, n, m, p):
        """Given U and W, solve for V using least squares."""
        r = U.shape[1]
        V = np.zeros((m*p, r))
        
        for b in range(m*p):
            A = np.zeros((n*m * p*n, r))
            rhs = T[:, b, :].flatten()
            
            idx = 0
            for a in range(n*m):
                for c in range(p*n):
                    A[idx, :] = U[a, :] * W[c, :]
                    idx += 1
            
            V_b, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
            V[b, :] = V_b
        
        return V
    
    # Try to find lower-rank decompositions using alternating least squares
    for target_rank in range(24, 17, -1):
        if time.time() > deadline - 15:
            break
        
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            prev_error = float('inf')
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                W = solve_for_W(U, V, T, n, m, p)
                U = solve_for_U(V, W, T, n, m, p)
                V = solve_for_V(U, W, T, n, m, p)
                
                if iteration % 5 == 0:
                    err = compute_error_fast(U, V, W, T)
                    if err < 1e-10:
                        break
                    if abs(prev_error - err) < 1e-12:
                        break
                    prev_error = err
            
            err = compute_error_fast(U, V, W, T)
            if err < 1e-8:
                if verify_full_decomposition(U, V, W, n, m, p, tol=1e-6):
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
        
        if best_rank == target_rank:
            break
    
    def to_native_list(arr):
        return [[float(x) for x in row] for row in arr]

    U_arr = np.array(best_result["U"], dtype=float)
    V_arr = np.array(best_result["V"], dtype=float)
    W_arr = np.array(best_result["W"], dtype=float)
    
    def try_snap(arr, step=0.5, tol=1e-6):
        snapped = np.round(arr / step) * step
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    U_snapped = try_snap(U_arr)
    V_snapped = try_snap(V_arr)
    W_snapped = try_snap(W_arr)
    
    if U_snapped is not None and V_snapped is not None and W_snapped is not None:
        if verify_full_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-8):
            best_result["U"] = to_native_list(U_snapped)
            best_result["V"] = to_native_list(V_snapped)
            best_result["W"] = to_native_list(W_snapped)
            return best_result

    best_result["U"] = to_native_list(U_arr)
    best_result["V"] = to_native_list(V_arr)
    best_result["W"] = to_native_list(W_arr)
    
    return best_result
# EVOLVE-BLOCK-END
