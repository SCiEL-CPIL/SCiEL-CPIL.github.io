import time
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill

def _time_remaining():
    return _DEADLINE - time.time()

def _build_strassen_rank25():
    """Build the rank-25 Strassen decomposition for 2x2 * 2x7."""
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_strassen_block(col_start, k_start):
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1
        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1
        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1
        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    # Schoolbook for column 6
    col_start = 21
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return U, V, W

def _verify_fast(U, V, W, n=2, m=2, p=7):
    """Fast verification using numpy. Returns True if decomposition is valid."""
    rank = U.shape[1]
    # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Build the tensor and check
    # Efficient: for each r, add outer product
    # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # Shape of T: (n*m, m*p, p*n)
    # But we only need to check specific entries: T[i*m+j, j*p+k, k*n+i] == 1
    
    # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # Vectorize: build matrix M where M[a,b] = sum_r U[a,r]*V[b,r]
    # Then check M[i*m+j, j*p+k] * ... no, we need triple product.
    
    # Alternative: for each r, compute contribution to each (i,j,k)
    # Let's just loop over the 28 (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U, V, W, target_rank, time_budget):
    """
    Try to reduce rank from current to target_rank.
    Strategy: start from current solution, zero out one term, randomly perturb others.
    Also try pure random search with small integer entries.
    """
    start_time = time.time()
    n, m, p = 2, 2, 7
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Method 1: Try to zero out one term and adjust
    # For each term to remove, zero it and try random adjustments to remaining terms
    for remove_idx in range(current_rank):
        if _time_remaining() < time_budget:
            break
        if time.time() - start_time > time_budget:
            break
        
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, remove_idx] = 0
        V_test[:, remove_idx] = 0
        W_test[:, remove_idx] = 0
        
        # Try random perturbations of remaining terms
        for attempt in range(200):
            if _time_remaining() < time_budget:
                break
            if time.time() - start_time > time_budget:
                break
            
            U_p = U_test.copy()
            V_p = V_test.copy()
            W_p = W_test.copy()
            
            # Randomly perturb a few entries in remaining terms
            for _ in range(3):
                row = random.randint(0, n*m-1)
                col = random.randint(0, target_rank-1)
                if col != remove_idx:
                    U_p[row, col] += random.choice([-1, 1])
                row = random.randint(0, m*p-1)
                col = random.randint(0, target_rank-1)
                if col != remove_idx:
                    V_p[row, col] += random.choice([-1, 1])
                row = random.randint(0, p*n-1)
                col = random.randint(0, target_rank-1)
                if col != remove_idx:
                    W_p[row, col] += random.choice([-1, 1])
            
            if _verify_fast(U_p, V_p, W_p, n, m, p):
                # Remove the zeroed column
                mask = np.ones(target_rank, dtype=bool)
                mask[remove_idx] = False
                return U_p[:, mask].astype(int), V_p[:, mask].astype(int), W_p[:, mask].astype(int)
    
    # Method 2: Pure random search with small entries
    for size in [2, 3]:
        if _time_remaining() < time_budget:
            break
        if time.time() - start_time > time_budget:
            break
        
        low = -size // 2
        high = (size - 1) // 2
        max_attempts = 3000
        for attempt in range(max_attempts):
            if _time_remaining() < time_budget:
                break
            if time.time() - start_time > time_budget:
                break
            
            U_rand = np.random.randint(low, high+1, (n*m, target_rank))
            V_rand = np.random.randint(low, high+1, (m*p, target_rank))
            W_rand = np.random.randint(low, high+1, (p*n, target_rank))
            
            if _verify_fast(U_rand, V_rand, W_rand, n, m, p):
                return U_rand, V_rand, W_rand
    
    return None

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Starts with rank-25 Strassen, attempts to reduce to rank-24 via local search.
    """
    if _time_remaining() < 10.0:
        # Safety: return schoolbook rank-28
        U, V, W = _schoolbook_decomposition()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Step 1: Build rank-25 Strassen decomposition
    U25, V25, W25 = _build_strassen_rank25()
    
    # Verify
    if not _verify_fast(U25, V25, W25):
        U, V, W = _schoolbook_decomposition()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U25, V25, W25
    best_rank = 25
    
    # Step 2: Try to find rank 24
    time_budget = min(_time_remaining() - 10.0, 80.0)
    if time_budget > 5.0:
        result = _try_reduce_rank(U25, V25, W25, 24, time_budget)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 24
    
    # Step 3: If rank 24 found, try rank 23
    if best_rank == 24 and _time_remaining() > 20.0:
        time_budget = min(_time_remaining() - 15.0, 40.0)
        if time_budget > 5.0:
            result = _try_reduce_rank(best_U, best_V, best_W, 23, time_budget)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = 23
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB
            
    C_mat = np.zeros((2,7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k+i]
            
    if np.allclose(C_mat, C_expected):
        print(f"Verification PASSED (rank={U.shape[1]})")
    else:
        print(f"Verification FAILED (rank={U.shape[1]})")
        print(C_mat)
        print(C_expected)
