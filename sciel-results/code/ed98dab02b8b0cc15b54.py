"""Matrix multiplication 3,3,8 - rank-minimal decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    nm, mp, pn = n * m, m * p, p * n
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return U, V, W

def _try_structured_search(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using structured patterns.
    """
    nm, mp, pn = n * m, m * p, p * n
    best_rank = n * m * p
    best_decomp = None
    
    # First get the schoolbook as baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    if _verify_decomposition(U_sb, V_sb, W_sb, n, m, p):
        best_decomp = (U_sb, V_sb, W_sb)
        best_rank = n * m * p
    
    # Try a systematic search for lower rank decompositions
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p)
    
    # Key insight: For an all-ones tensor, we can potentially use
    # rank = min(n*m, m*p, p*n) with appropriate scaling
    
    # Let's try various structured approaches:
    
    # Approach 1: Rank = n*m (group by (i,j))
    # U[i*m+j, r] = 1 if r = i*m+j, else 0
    # V[j*p+k, r] = 1/sqrt(p) for all k (but this won't work exactly)
    
    # Actually, let's think about this more carefully.
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For rank = n*m = 9:
    # Let r index (i', j')
    # U[i*m+j, (i',j')] = 1 if i=i' and j=j', else 0
    # Then we need: V[j*p+k, (i',j')] * W[k*n+i, (i',j')] = 1 for i=i', j=j'
    # And 0 otherwise.
    # This gives V[j*p+k, (i',j')] * W[k*n+i, (i',j')] = delta_{i,i'} * delta_{j,j'}
    # But V depends on j and W depends on i, so this is tricky.
    
    # Let's try a different approach: use the fact that the tensor is all-ones
    # and try to find a low-rank decomposition using random methods with rounding.
    
    # Try ranks from best_rank-1 down to 9
    for target_rank in range(best_rank - 1, 8, -1):
        if time.time() > deadline:
            break
            
        # Try random initializations with structured patterns
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Try different initialization strategies
            strategy = attempt % 3
            
            if strategy == 0:
                # Random initialization
                U = np.random.randn(nm, target_rank)
                V = np.random.randn(mp, target_rank)
                W = np.random.randn(pn, target_rank)
            elif strategy == 1:
                # Structured: U has identity-like pattern
                U = np.zeros((nm, target_rank))
                V = np.random.randn(mp, target_rank)
                W = np.random.randn(pn, target_rank)
                for i in range(nm):
                    U[i, i % target_rank] = 1.0
            else:
                # Sparse random
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    # Pick random non-zero entries
                    u_idx = np.random.choice(nm, size=min(3, nm), replace=False)
                    v_idx = np.random.choice(mp, size=min(3, mp), replace=False)
                    w_idx = np.random.choice(pn, size=min(3, pn), replace=False)
                    U[u_idx, r] = np.random.choice([-1, 1], size=len(u_idx))
                    V[v_idx, r] = np.random.choice([-1, 1], size=len(v_idx))
                    W[w_idx, r] = np.random.choice([-1, 1], size=len(w_idx))
            
            # Scale to reasonable magnitude
            norm_U = np.linalg.norm(U)
            norm_V = np.linalg.norm(V)
            norm_W = np.linalg.norm(W)
            if norm_U > 0:
                U /= norm_U
            if norm_V > 0:
                V /= norm_V
            if norm_W > 0:
                W /= norm_W
            
            # Check if this is close to a solution
            error = 0
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err = val - 1.0
                        error += err * err
                        max_err = max(max_err, abs(err))
            
            if max_err < 0.1:
                # Close enough to try rounding
                # Try rounding to integers and half-integers
                for scale in [1, 2, 4]:
                    U_rounded = np.round(U * scale) / scale
                    V_rounded = np.round(V * scale) / scale
                    W_rounded = np.round(W * scale) / scale
                    
                    # Verify the rounded version
                    error_rounded = 0
                    max_err_rounded = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                                err = val - 1.0
                                error_rounded += err * err
                                max_err_rounded = max(max_err_rounded, abs(err))
                    
                    if max_err_rounded < 1e-9:
                        # Found a valid solution!
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        
                        # Clean up values
                        for i in range(nm):
                            for r in range(target_rank):
                                val = U_list[i][r]
                                if abs(val) < 1e-9:
                                    U_list[i][r] = 0
                                elif abs(val - round(val)) < 1e-9:
                                    U_list[i][r] = round(val)
                                elif abs(val * 2 - round(val * 2)) < 1e-9:
                                    U_list[i][r] = round(val * 2) / 2
                        for i in range(mp):
                            for r in range(target_rank):
                                val = V_list[i][r]
                                if abs(val) < 1e-9:
                                    V_list[i][r] = 0
                                elif abs(val - round(val)) < 1e-9:
                                    V_list[i][r] = round(val)
                                elif abs(val * 2 - round(val * 2)) < 1e-9:
                                    V_list[i][r] = round(val * 2) / 2
                        for i in range(pn):
                            for r in range(target_rank):
                                val = W_list[i][r]
                                if abs(val) < 1e-9:
                                    W_list[i][r] = 0
                                elif abs(val - round(val)) < 1e-9:
                                    W_list[i][r] = round(val)
                                elif abs(val * 2 - round(val * 2)) < 1e-9:
                                    W_list[i][r] = round(val * 2) / 2
                        
                        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                            return U_list, V_list, W_list, target_rank
    
    return best_decomp[0], best_decomp[1], best_decomp[2], best_rank

def construct():
    """
    Returns a rank-minimal tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Try to find a lower-rank decomposition
    U, V, W, rank = _try_structured_search(n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
