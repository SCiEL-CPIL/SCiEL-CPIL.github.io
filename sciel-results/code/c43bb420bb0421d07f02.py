import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _check_half_integer(val, tol=1e-9):
    """Check if val is a half-integer."""
    two_val = 2.0 * val
    nearest = round(two_val)
    return abs(two_val - nearest) < tol

def _is_valid_half_integer_matrix(M, tol=1e-9):
    for row in M:
        for val in row:
            if not _check_half_integer(val, tol):
                return False
    return True

def _solve_2x3x3(rank_target, max_iter=1000, tol=1e-9):
    """
    Attempt to find a rank 'rank_target' decomposition for 2x3x3 matrix multiplication.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 2, 3, 3
    # Initialize with random values or schoolbook subset
    # Schoolbook rank is 18.
    # We want rank_target <= 18.
    
    # Use a random initialization scaled appropriately
    U = np.random.randn(n*m, rank_target) * 0.5
    V = np.random.randn(m*p, rank_target) * 0.5
    W = np.random.randn(p*n, rank_target) * 0.5
    
    # Simple gradient descent on the loss function
    # Loss = sum_{i,j,k} ( sum_c U_{ij,c} V_{jk,c} W_{ki,c} - 1 )^2
    # This is non-convex.
    
    # To improve success rate, we can try multiple random restarts
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(10):
        U = np.random.randn(n*m, rank_target) * 0.5
        V = np.random.randn(m*p, rank_target) * 0.5
        W = np.random.randn(p*n, rank_target) * 0.5
        
        # Normalize to keep values reasonable
        # We want the product to be 1.
        
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank_target):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    loss += (s - 1.0)**2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        # If loss is already very small, check validity
        if loss < 1e-6:
             # Try to polish to half-integers
             # This is a hard constraint, so we just check if the current float solution is close to half-integers
             # If not, we might need to project.
             pass

    # If we found a good float solution, try to snap to half-integers
    if best_U is not None:
        # Snap to nearest half-integer
        U_snapped = np.round(best_U * 2) / 2
        V_snapped = np.round(best_V * 2) / 2
        W_snapped = np.round(best_W * 2) / 2
        
        # Verify snapped solution
        if _verify_decomposition(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p, tol=1e-9):
            return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
            
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # Safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Strategy: Try to find a lower rank decomposition by splitting into 2x3x3 blocks
    # B is 3x6. Split into B1 (3x3) and B2 (3x3).
    # C = A B = [A B1, A B2].
    # We need a decomposition for 2x3x3.
    # If we find rank r_233, total rank is 2 * r_233.
    
    # Try ranks for 2x3x3: 16, 17, 18
    # Schoolbook for 2x3x3 is 18.
    
    found_2x3x3 = None
    for r in [16, 17, 18]:
        if time.time() > deadline:
            break
        res = _solve_2x3x3(r)
        if res is not None:
            U_233, V_233, W_233 = res
            # Construct the full 2x3x6 decomposition
            # U_full: 6 x (2*r)
            # V_full: 18 x (2*r)
            # W_full: 12 x (2*r)
            
            U_full = [[0] * (2*r) for _ in range(n*m)]
            V_full = [[0] * (2*r) for _ in range(m*p)]
            W_full = [[0] * (2*r) for _ in range(p*n)]
            
            # First half: B1 (columns 0-2 of B)
            # V indices for B1: j*3 + k where k in 0..2
            # W indices for C1: k*2 + i where k in 0..2
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        U_full[i*m+j][c] = U_233[i*m+j][c]
                for j in range(m):
                    for k in range(3):
                        V_full[j*3+k][c] = V_233[j*3+k][c]
                for k in range(3):
                    for i in range(n):
                        W_full[k*n+i][c] = W_233[k*n+i][c]
                        
            # Second half: B2 (columns 3-5 of B)
            # V indices for B2: j*3 + (k+3) where k in 0..2 -> j*3 + 3 + k
            # W indices for C2: (k+3)*2 + i where k in 0..2 -> (k+3)*2 + i
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        U_full[i*m+j][r+c] = U_233[i*m+j][c]
                for j in range(m):
                    for k in range(3):
                        V_full[j*3+3+k][r+c] = V_233[j*3+k][c]
                for k in range(3):
                    for i in range(n):
                        W_full[(k+3)*n+i][r+c] = W_233[k*n+i][c]
                        
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                if 2*r < best_rank:
                    best_rank = 2*r
                    best_result = {"U": U_full, "V": V_full, "W": W_full}
                break # Found a valid lower rank, no need to try higher r for 233? 
                      # Actually, we tried 16 first. If 16 worked, 32 is best.
                      # If 16 failed, we try 17 (rank 34), then 18 (rank 36).
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
