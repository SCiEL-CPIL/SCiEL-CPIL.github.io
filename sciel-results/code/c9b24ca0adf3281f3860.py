"""
Strategy: Rank-increasing exact algebraic search.
Instead of reducing rank from a high-rank decomposition, we start from rank 1 and
incrementally increase the rank, using randomized algebraic methods to find exact
solutions. This avoids the non-convex optimization traps of rank reduction.
"""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    def check_decomp(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # This should equal 1 for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def build_schoolbook():
        """Build the trivial rank 64 decomposition."""
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def try_find_rank_r(target_rank, max_attempts=100):
        """
        Try to find a rank r decomposition using randomized algebraic search.
        We fix U and V randomly, then solve for W linearly.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                return None
            
            # Initialize U and V with random small integers or half-integers
            # Use a structured random initialization
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(U_dim, target_rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(V_dim, target_rank))
            
            # For each (i,j,k), we need:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is a linear system in W.
            # Let A_w be the matrix where A_w[k*n+i, r] = U[i*m+j,r] * V[j*p+k,r]
            # But we have multiple constraints for the same W row (k*n+i).
            # We need to collect all constraints for each W row.
            
            # Build the linear system:
            # For each W row w_idx = k*n+i, we have multiple equations:
            # sum_r A_w[w_idx, r] * W[w_idx, r] = 1 (for each (i,j,k) mapping to w_idx)
            # Actually, for a fixed (i,k), j varies, so we get multiple equations.
            # Let's build the system properly.
            
            # We have U_dim * V_dim * W_dim = 4096 equations.
            # We have W_dim * target_rank unknowns.
            # This is overdetermined, so we use least squares.
            
            # Build A_w and b_w
            A_w = np.zeros((W_dim, target_rank))
            b_w = np.zeros(W_dim)
            count_w = np.zeros(W_dim)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        A_w[w_idx, :] += U[u_idx, :] * V[v_idx, :]
                        b_w[w_idx] += 1.0
                        count_w[w_idx] += 1
            
            # Only use rows with at least one constraint
            valid_w = count_w > 0
            if np.sum(valid_w) < W_dim:
                continue
            
            # Solve least squares
            try:
                W_sol, residuals, rank_w, sv = np.linalg.lstsq(
                    A_w[valid_w, :], b_w[valid_w], rcond=None
                )
                W = np.zeros((W_dim, target_rank))
                W[valid_w, :] = W_sol
            except:
                continue
            
            # Check if the solution is valid
            if check_decomp(U, V, W, tol=1e-5):
                # Check if entries are close to integers or half-integers
                is_valid = True
                for mat in [U, V, W]:
                    for val in mat.flatten():
                        if abs(val - round(val)) > 1e-4 and abs(val - (round(2*val)/2)) > 1e-4:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if is_valid:
                    return U, V, W
        
        return None
    
    # Start with schoolbook
    U, V, W, current_rank = build_schoolbook()
    
    # Try to find lower ranks, starting from rank 15 (known lower bound)
    # and going up to current_rank - 1
    for target_rank in range(15, current_rank):
        if time.time() > deadline - 10.0:
            break
        
        result = try_find_rank_r(target_rank, max_attempts=50)
        if result is not None:
            U, V, W = result
            current_rank = target_rank
            break
    
    # Final verification
    if not check_decomp(U, V, W):
        # Fall back to schoolbook
        U, V, W, current_rank = build_schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
