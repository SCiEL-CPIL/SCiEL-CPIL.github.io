import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        
        # Compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Result shape (n, m, p)
        # Note: W is indexed by (k, n, i) in the prompt's W[k*n+i]
        # So W_t[k, i, c] corresponds to W[k*n+i, c]
        # The prompt verify: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # So indices are i, j, k.
        # U[i, j, c] * V[j, k, c] * W[k, i, c]
        
        try:
            result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            # Check if all are 1.0
            if np.allclose(result, 1.0, atol=1e-7):
                return True
        except Exception:
            return False
        return False

    def snap_to_half(U, V, W):
        # Snap values to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None

    # Schoolbook fallback
    def make_schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    best_U, best_V, best_W, best_rank = make_schoolbook()

    np.random.seed(42)
    
    # Search for lower ranks
    # Start from 55 down to 30
    for target_rank in range(55, 29, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few different initialization strategies
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            
            # Strategy 1: Random integers -1, 0, 1
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            if verify(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
            
            # Strategy 2: Random floats, then snap
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to roughly unit norm
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            # Check direct
            if verify(U, V, W):
                # Try to snap
                U_s, V_s, W_s = snap_to_half(U, V, W)
                if verify(U_s, V_s, W_s):
                    U, V, W = U_s, V_s, W_s
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
            else:
                # Try snapping the random floats
                U_s, V_s, W_s = snap_to_half(U, V, W)
                if verify(U_s, V_s, W_s):
                    best_U, best_V, best_W, best_rank = U_s, V_s, W_s, target_rank
                    break

        if best_rank < target_rank:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
