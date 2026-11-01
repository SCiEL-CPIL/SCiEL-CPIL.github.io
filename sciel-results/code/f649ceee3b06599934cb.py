import time
import numpy as np

DIMENSIONS = [2, 3, 9]
_DEADLINE = 150.0

# EVOLVE-BLOCK-START
def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    def time_left():
        return _DEADLINE - (time.time() - start) - 5.0
    
    if time_left() <= 0:
        return {"U": [[0]], "V": [[0]], "W": [[0]]}
    
    result = _construct_schoolbook(n, m, p)
    U, V, W = result["U"], result["V"], result["W"]
    rank = U.shape[1]
    
    # Verify the 54 non-zero entries
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U[i * m + j]
                v_vec = V[j * p + k]
                w_vec = W[k * n + i]
                # Sum of u_vec[r] * v_vec[r] * w_vec[r]
                # Since they are sparse with single 1s, the product is 1 only if all have 1 at same r
                # In schoolbook, they are 1 at unique r for each (i,j,k)
                # So sum should be 1.
                prod = u_vec * v_vec * w_vec
                if np.sum(prod) != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        # Convert to list of lists if required, or keep numpy?
        # Interface says: factor matrices of shapes ... with integer/half-integer entries.
        # Usually numpy arrays are accepted. If not, .tolist()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": [[0]], "V": [[0]], "W": [[0]]}
# EVOLVE-BLOCK-END
