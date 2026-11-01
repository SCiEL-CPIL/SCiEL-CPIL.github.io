import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _time_remaining():
    return _DEADLINE - time.time()

def _construct_rank_25():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses Strassen for 2x2 blocks and schoolbook for the last column.
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
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
        
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
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

def _construct_rank_28():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return U, V, W

def _get_target_tensor():
    n, m, p = 2, 2, 7
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _verify(U, V, W, T):
    # Reconstruct T from U, V, W
    # T_reconstructed = sum_r U[:,r] (x) V[:,r] (x) W[:,r]
    # Efficiently: T_reconstructed[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Using einsum or loops
    T_rec = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(T_rec, T, atol=1e-9)

def construct():
    if _time_remaining() < 10.0:
        U, V, W = _construct_rank_28()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    T = _get_target_tensor()
    
    # 1. Try Rank 25
    U25, V25, W25 = _construct_rank_25()
    if _verify(U25, V25, W25, T):
        # Check if we can improve to 24
        if _time_remaining() > 30.0:
            res = _search_rank_24(T, U25, V25, W25)
            if res is not None:
                U, V, W = res
                if _verify(U, V, W, T):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}
    
    # 2. Fallback to Rank 28
    U28, V28, W28 = _construct_rank_28()
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

def _search_rank_24(T, U_init, V_init, W_init):
    """
    Attempt to find a rank-24 decomposition using least squares optimization.
    """
    n, m, p = 2, 2, 7
    dim_u = n*m
    dim_v = m*p
    dim_w = p*n
    rank = 24
    
    # Flatten U, V, W into a single parameter vector
    # We initialize by taking the rank 25 solution and zeroing out one column (the last one, which is schoolbook)
    # and perturbing the others? Or just random init?
    # Random init is often better for escaping local minima, but we have a good structure.
    # Let's try initializing with U25, V25, W25 but with the last column zeroed and scaled?
    # Actually, just use random initialization with small values, or reuse the first 24 columns of the rank 25 solution.
    
    U0 = U_init[:, :rank].copy()
    V0 = V_init[:, :rank].copy()
    W0 = W_init[:, :rank].copy()
    
    # Add small noise to avoid exact local minima of the rank 25 subspace
    U0 += np.random.normal(0, 0.1, U0.shape)
    V0 += np.random.normal(0, 0.1, V0.shape)
    W0 += np.random.normal(0, 0.1, W0.shape)
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def residuals(x):
        U = x[:dim_u*rank].reshape(dim_u, rank)
        V = x[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = x[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            
        return (T_rec - T).flatten()
    
    try:
        # Use least_squares. Method 'lm' is for linear, but this is non-linear.
        # 'trf' is default and robust.
        result = least_squares(residuals, x0, method='trf', max_nfev=10000, xtol=1e-12, ftol=1e-12)
        
        if result.success or result.cost < 1e-6:
            x_opt = result.x
            U_opt = x_opt[:dim_u*rank].reshape(dim_u, rank)
            V_opt = x_opt[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W_opt = x_opt[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            
            # Try to round to half-integers
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            # Check if rounded version is valid
            if _verify(U_rounded, V_rounded, W_rounded, T):
                return U_rounded, V_rounded, W_rounded
            
            # If not, check if the unrounded version is "close enough" to integers?
            # The problem requires integer/half-integer entries.
            # So we must return the rounded version if it's valid.
            
    except Exception:
        pass
        
    return None

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    T = _get_target_tensor()
    if _verify(U, V, W, T):
        print(f"Valid decomposition, rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
