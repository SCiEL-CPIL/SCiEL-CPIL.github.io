import numpy as np
import time

def _search_slice_rank():
    # Target tensor for 2x3 x 3x1
    # A: 6, B: 3, C: 2
    T = np.zeros((6, 3, 2))
    T[0, 0, 0] = 1
    T[3, 0, 1] = 1
    T[1, 1, 0] = 1
    T[4, 1, 1] = 1
    T[2, 2, 0] = 1
    T[5, 2, 1] = 1
    
    # Try rank 5
    # U: 6x5, V: 3x5, W: 2x5
    # Coefficients in {-1, 0, 1}
    # This is a small search space if we use random restarts
    rng = np.random.default_rng(42)
    alphabet = np.array([-1, 0, 1])
    
    for _ in range(1000):
        U = rng.choice(alphabet, size=(6, 5))
        V = rng.choice(alphabet, size=(3, 5))
        W = rng.choice(alphabet, size=(2, 5))
        
        S = np.zeros((6, 3, 2))
        for r in range(5):
            S += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        
        if np.allclose(S, T):
            return U, V, W
            
    return None

def construct():
    # Try to find a low rank slice
    slice_decomp = _search_slice_rank()
    
    if slice_decomp:
        U_slice, V_slice, W_slice = slice_decomp
        # Full problem: 2x3 x 3x9
        # We have 9 slices (k=0..8)
        # U_full: 6 x (9 * rank_slice)
        # V_full: 27 x (9 * rank_slice)
        # W_full: 18 x (9 * rank_slice)
        
        rank_slice = U_slice.shape[1]
        rank_full = 9 * rank_slice
        
        U_full = np.zeros((6, rank_full))
        V_full = np.zeros((27, rank_full))
        W_full = np.zeros((18, rank_full))
        
        for k in range(9):
            # Slice k uses columns k*rank_slice to (k+1)*rank_slice
            start = k * rank_slice
            end = (k + 1) * rank_slice
            
            U_full[:, start:end] = U_slice
            
            # V_slice is 3x5. We need to place it in V_full which is 27x5.
            # V_full row index j*p + k. For fixed k, rows are k, 3+k, 6+k?
            # No, V index is j*p + k. p=9.
            # So for fixed k, the rows are 0*9+k, 1*9+k, 2*9+k => k, 9+k, 18+k.
            V_full[k, start:end] = V_slice[0]
            V_full[9+k, start:end] = V_slice[1]
            V_full[18+k, start:end] = V_slice[2]
            
            # W_slice is 2x5. We need to place it in W_full which is 18x5.
            # W index is k*n + i. n=2.
            # For fixed k, rows are k*2+0, k*2+1 => 2k, 2k+1.
            W_full[2*k, start:end] = W_slice[0]
            W_full[2*k+1, start:end] = W_slice[1]
            
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to schoolbook
        n, m, p = 2, 3, 9
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
