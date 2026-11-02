import time
import numpy as np
import random

DIMENSIONS = [2, 3, 3]
DEADLINE_MARGIN = 5  # seconds to spare

def _schoolbook():
    """Return the schoolbook (rank-18) decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W):
    """
    Validate that U, V, W form a valid decomposition.
    Uses numpy for speed.
    """
    n, m, p = DIMENSIONS
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Build target tensor T (6, 9, 6)
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    # Calculate decomposed tensor
    # T_calc[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
    # We can do this by iterating r or using einsum/broadcasting
    # For small R, a loop is fine, but let's try broadcasting
    # U[:, r] is (6,1), V[:, r] is (9,1), W[:, r] is (6,1)
    # Outer product is (6,9,6)
    
    T_calc = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_calc += np.einsum('u,v,w->uvw', U_np[:, r], V_np[:, r], W_np[:, r])
        
    # Check if T_calc == T_target
    if not np.allclose(T_calc, T_target, atol=1e-9):
        return False
    
    # Check if entries are integer/half-integer
    # The problem allows integer/half-integer entries.
    # We should check if 2*U, 2*V, 2*W are integers.
    if not np.allclose(U_np * 2, np.round(U_np * 2)):
        return False
    if not np.allclose(V_np * 2, np.round(V_np * 2)):
        return False
    if not np.allclose(W_np * 2, np.round(W_np * 2)):
        return False
        
    return True

def _try_rank(rank, start_time, deadline):
    """
    Try to find a rank-`rank` decomposition using random search.
    """
    n, m, p = DIMENSIONS
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Values to sample from: half-integers
    # We scale by 2 to work with integers internally, then divide by 2 at the end
    # But for search, we can just use floats and check at the end
    # Or we can search over integers in {-2, -1, 0, 1, 2} and divide by 2?
    # Let's search over integers in {-2, -1, 0, 1, 2} for 2*U, 2*V, 2*W
    # This ensures the final values are half-integers.
    
    values = [-2, -1, 0, 1, 2]
    
    # Precompute target tensor for fast comparison
    T_target = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    # We will store the scaled integers (2*U, etc.)
    # Let U_int be 2*U. Then U = U_int / 2.
    # T_calc = sum_r (U_int[u,r]/2) * (V_int[v,r]/2) * (W_int[w,r]/2)
    # T_calc = (1/8) * sum_r U_int[u,r] * V_int[v,r] * W_int[w,r]
    # We need T_calc = T_target
    # So sum_r U_int[u,r] * V_int[v,r] * W_int[w,r] = 8 * T_target
    
    T_target_scaled = 8.0 * T_target
    
    while time.time() < deadline:
        # Random initialization
        U_int = np.random.choice(values, size=(U_size, rank))
        V_int = np.random.choice(values, size=(V_size, rank))
        W_int = np.random.choice(values, size=(W_size, rank))
        
        # Calculate scaled tensor
        T_calc_scaled = np.zeros((U_size, V_size, W_size))
        for r in range(rank):
            T_calc_scaled += np.einsum('u,v,w->uvw', U_int[:, r], V_int[:, r], W_int[:, r])
            
        if np.allclose(T_calc_scaled, T_target_scaled, atol=1e-9):
            # Found a solution!
            # Convert back to floats (divide by 2)
            U_res = (U_int / 2.0).tolist()
            V_res = (V_int / 2.0).tolist()
            W_res = (W_int / 2.0).tolist()
            return {"U": U_res, "V": V_res, "W": W_res}
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s out of 150s
    
    # Try rank 17 first
    # If found, return it
    res = _try_rank(17, start_time, deadline)
    if res:
        return res
        
    # Fallback to schoolbook
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Valid:", _validate(res['U'], res['V'], res['W']))
