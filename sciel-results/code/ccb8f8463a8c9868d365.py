"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave 10s margin for 150s limit

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Reshape: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # Compute sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Result should be 1 for all (i,j,k)
    # Efficient check using einsum or loops
    result = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                result[i, j, k] = s
    return np.allclose(result, 1.0, atol=1e-9)

def _try_round(U, V, W):
    """Try to round entries to half-integers and verify."""
    n, m, p = DIMENSIONS
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school

    # Known: for [2,4,7], try ranks from some lower bound up
    # The tensor rank is at least max(n*m, m*p, p*n) = max(8,28,14) = 28
    # Also at least n*m*p / min(n,m,p) ... not a tight bound
    # Let's try ranks from 28 to 55
    
    # We will use ALS (Alternating Least Squares) to find a decomposition.
    # Then we will try to round to half-integers.
    
    # Optimization: Precompute indices for speed?
    # For small dimensions, loops are fine.
    
    # Try ranks from 28 upwards.
    # If we find a valid half-integer decomposition, return it.
    # If time runs out, return the best valid one (which might be schoolbook if no rounding worked, 
    # but usually ALS finds a valid float one, and rounding might fail. 
    # However, the task requires EXACT integer/half-integer entries.
    # So we MUST find a half-integer solution.
    # If ALS converges to a float solution that doesn't round to a valid half-integer, we must continue searching.
    # Often, the true minimal rank decomposition has integer/half-integer entries.
    
    # Let's iterate ranks.
    for rank in range(28, rank_school + 1):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            rng = np.random.default_rng(rank * 100 + trial)
            U = rng.standard_normal((n * m, rank))
            V = rng.standard_normal((m * p, rank))
            W = rng.standard_normal((p * n, rank))
            
            T = np.ones((n, m, p), dtype=np.float64)
            
            # ALS iterations
            for it in range(200):
                if time.time() - _START_TIME > _TIME_LIMIT - 10:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # Fix i,j. Solve for U_row.
                        # A[c,k] = V[j*p+k,c] * W[k*n+i,c]
                        A = np.zeros((rank, p))
                        for k in range(p):
                            for c in range(rank):
                                A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                        # U_row @ A.T = T[i,j,:]
                        # A @ U_row.T = T[i,j,:].T
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, T[i,j,:].T, rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # Fix j,k. Solve for V_row.
                        # A[c,i] = U[i*m+j,c] * W[k*n+i,c]
                        A = np.zeros((rank, n))
                        for i in range(n):
                            for c in range(rank):
                                A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                        # V_row @ A.T = T[:,j,k]
                        # A @ V_row.T = T[:,j,k].T
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, T[:,j,k].T, rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # Fix k,i. Solve for W_row.
                        # A[c,j] = U[i*m+j,c] * V[j*p+k,c]
                        A = np.zeros((rank, m))
                        for j in range(m):
                            for c in range(rank):
                                A[c, j] = U[i*m+j, c] * V[j*p+k, c]
                        # W_row @ A.T = T[i,:,k]
                        # A @ W_row.T = T[i,:,k].T
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, T[i,:,k].T, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i,j,k] = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
                res = np.linalg.norm(recon - T)
                if res < 1e-8:
                    break
            
            # Try to round to half-integers
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Simple rounding
            rounded = _try_round(U_list, V_list, W_list)
            if rounded is not None:
                U_r, V_r, W_r = rounded
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U_r, "V": V_r, "W": U_r} # Typo check: W_r
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                return best_result
            
            # If rounding failed, we might try to refine or accept if it's close?
            # No, must be exact.
            # If we found a low rank float solution but can't round it, we can't use it.
            # However, often the minimal rank solution IS half-integer.
            # If we fail to round, we should probably try more trials or higher rank.
            
    return best_result
# EVOLVE-BLOCK-END
