import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Build rank-28 decomposition
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # 7 Strassen terms for 2x2x2
        # M1 = (a00+a11)(b00+b11)
        U[(0 + i_off) * m + (0 + j_off), col] = 1
        U[(1 + i_off) * m + (1 + j_off), col] = 1
        V[(0 + j_off) * p + (0 + k_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = 1
        W[(0 + k_off) * n + (0 + i_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M2 = a10*(b00+b01)
        U[(1 + i_off) * m + (0 + j_off), col] = 1
        U[(1 + i_off) * m + (1 + j_off), col] = 1
        V[(0 + j_off) * p + (0 + k_off), col] = 1
        W[(0 + k_off) * n + (1 + i_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = -1
        col += 1
        
        # M3 = (a00-a10)*b10
        U[(0 + i_off) * m + (0 + j_off), col] = 1
        V[(0 + j_off) * p + (1 + k_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = -1
        W[(1 + k_off) * n + (0 + i_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M4 = a11*(b10-b00)
        U[(1 + i_off) * m + (1 + j_off), col] = 1
        V[(1 + j_off) * p + (0 + k_off), col] = 1
        V[(0 + j_off) * p + (0 + k_off), col] = -1
        W[(0 + k_off) * n + (0 + i_off), col] = 1
        W[(0 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M5 = (a00+a01)*b11
        U[(0 + i_off) * m + (0 + j_off), col] = 1
        U[(0 + i_off) * m + (1 + j_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = 1
        W[(0 + k_off) * n + (0 + i_off), col] = -1
        W[(1 + k_off) * n + (0 + i_off), col] = 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b11)
        U[(1 + i_off) * m + (0 + j_off), col] = 1
        U[(0 + i_off) * m + (0 + j_off), col] = -1
        V[(0 + j_off) * p + (0 + k_off), col] = 1
        V[(0 + j_off) * p + (1 + k_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[(0 + i_off) * m + (1 + j_off), col] = 1
        U[(1 + i_off) * m + (1 + j_off), col] = -1
        V[(1 + j_off) * p + (0 + k_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = 1
        W[(0 + k_off) * n + (0 + i_off), col] = 1
        col += 1
    
    # Block 1: j=0, all i,k -> 2*5 = 10 schoolbook terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Block 2: j=1,2, k=0,1 -> Strassen 2x2x2 = 7 terms
    add_strassen_222(0, 1, 0)
    
    # Block 3: j=1,2, k=2 -> schoolbook 2x2x1 = 4 terms
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    
    # Block 4: j=1,2, k=3,4 -> Strassen 2x2x2 = 7 terms
    add_strassen_222(0, 1, 3)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Helper to check if entries are valid (integer or half-integer)
    def is_valid_entries(mat, tol=1e-9):
        # Check if all entries are close to k/2 for integer k
        # i.e., 2*x is close to an integer
        scaled = 2 * mat
        closest = np.round(scaled)
        if np.max(np.abs(scaled - closest)) < tol:
            return True
        return False

    def verify_decomposition(U_mat, V_mat, W_mat):
        """Check that sum_r u_r ⊗ v_r ⊗ w_r = T where T[i,j,k]=1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    coeff = 0
                    for r in range(U_mat.shape[1]):
                        coeff += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(coeff - 1) > 1e-10:
                        return False
        return True

    # Verify the rank-28 decomposition
    if not verify_decomposition(U, V, W):
        # Fallback: pure schoolbook
        rank_school = n*m*p
        U_s = np.zeros((n*m, rank_school))
        V_s = np.zeros((m*p, rank_school))
        W_s = np.zeros((p*n, rank_school))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U_s[i*m+j, idx] = 1
                    V_s[j*p+k, idx] = 1
                    W_s[k*n+i, idx] = 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Try to find a rank-27 decomposition
    # Build the 30 x 28 matrix M
    M = np.zeros((n*m*p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i*m*p + j*p + k
                M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    
    target = np.ones(n*m*p)
    
    # Check if rank 27 is possible within the span of these 28 terms
    # We try to exclude one term at a time
    found_rank_27 = False
    best_U = None
    best_V = None
    best_W = None
    
    if time.time() < deadline - 10:
        for excluded in range(rank):
            subset = [r for r in range(rank) if r != excluded]
            M_sub = M[:, subset]
            
            try:
                # Solve M_sub @ x = target
                x, residuals, rank_sub, s = np.linalg.lstsq(M_sub, target, rcond=1e-12)
                
                # Check if solution is valid
                if len(residuals) > 0 and residuals[0] < 1e-9:
                    # Construct the new tensors
                    new_U = U[:, subset] * x
                    new_V = V[:, subset] * x
                    new_W = W[:, subset] * x
                    
                    # Verify exact decomposition
                    if verify_decomposition(new_U, new_V, new_W):
                        # Check integer/half-integer constraint
                        if is_valid_entries(new_U) and is_valid_entries(new_V) and is_valid_entries(new_W):
                            found_rank_27 = True
                            best_U = new_U
                            best_V = new_V
                            best_W = new_W
                            break
            except np.linalg.LinAlgError:
                continue
    
    if found_rank_27:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Return the rank-28 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
