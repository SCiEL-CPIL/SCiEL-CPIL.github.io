"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Schoolbook baseline (always valid)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W form a valid decomposition."""
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def try_linear_solution(target_rank, num_trials=50):
        """
        Try to find a decomposition of target_rank by fixing U and V randomly,
        and solving for W using the linear system:
        For each i, k: M_ik^T w_ik = 1_m
        where M_ik[j, c] = U[i, j, c] * V[j, k, c]
        """
        if time.time() > deadline:
            return None
            
        nm = n * m
        mp = m * p
        pn = p * n
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization of U and V
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            
            # Reshape for easier access
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            
            W = np.zeros((pn, target_rank))
            valid = True
            
            # Solve for W for each (i, k)
            for i in range(n):
                for k in range(p):
                    # M_ik is (m, target_rank)
                    # M_ik[j, c] = U_r[i, j, c] * V_r[j, k, c]
                    M_ik = U_r[i, :, :] * V_r[:, k, :]
                    
                    # We want to solve M_ik^T w = 1_m
                    # A = M_ik^T (target_rank, m)
                    # b = ones(m)
                    A = M_ik.T
                    b = np.ones(m)
                    
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if the solution is good
                        # If rank < m, it might be underdetermined or inconsistent.
                        # lstsq returns min-norm solution.
                        # Check residual
                        if residuals is not None and len(residuals) > 0:
                            if residuals[0] > 1e-10:
                                # Solution not exact, but let's store it and check global later
                                # Or we can mark as invalid if we require exactness in construction
                                # However, due to floating point, residuals might be small but non-zero.
                                pass
                        W[k*n + i, :] = w
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
                
            # Verify the full decomposition
            if check_decomposition(U, V, W):
                return (U, V, W)
        
        return None

    # Try ranks starting from the theoretical lower bound max(n, m, p) = 5
    # The rank of matrix multiplication tensor M_{n,m,p} is at least max(n,m,p).
    # For 3x5x5, lower bound is 5.
    # We try small ranks first.
    for target_rank in range(5, 20):
        if time.time() > deadline:
            break
        
        # Try multiple trials for each rank
        result = try_linear_solution(target_rank, num_trials=100)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                # If we found a very low rank, we might want to try to lower it further,
                # but since we are iterating up, we can break if we find one?
                # No, we iterate from 5 upwards. So the first one found is the minimal?
                # Not necessarily, because 5 might fail and 6 succeeds.
                # So we continue to see if we can find a lower one? No, we started at 5.
                # So if 5 succeeds, it's the best.
                # If 5 fails, we try 6.
                # So we can break as soon as we find a valid one?
                # Yes, because we are going from low to high.
                break
    
    # Final check
    if not check_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
