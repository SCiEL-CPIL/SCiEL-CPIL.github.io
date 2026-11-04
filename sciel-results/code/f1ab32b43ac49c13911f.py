import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 6]

def _make_schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    R = U.shape[1]
    # Check integer/half-integer
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
            return False
    # Check tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if not np.isclose(val, 1.0, atol=tol):
                    return False
    return True

def _snap_and_check(U, V, W, tol=1e-9):
    # Snap to nearest half-integer
    U2 = np.round(U * 2) / 2.0
    V2 = np.round(V * 2) / 2.0
    W2 = np.round(W * 2) / 2.0
    
    # Check if the snap was close enough
    if np.linalg.norm(U - U2) > 1e-6 or np.linalg.norm(V - V2) > 1e-6 or np.linalg.norm(W - W2) > 1e-6:
        return None, None, None
        
    if _verify(U2, V2, W2, tol):
        return U2, V2, W2
    return None, None, None

def _als_update(U, V, W, n, m, p, R, iterations=100):
    """
    Alternating Least Squares for Tensor Decomposition.
    Goal: minimize sum_{i,j,k} (sum_r U_ijr V_jkr W_ikr - 1)^2
    """
    # Precompute structures for efficiency if possible? 
    # For n=4, m=5, p=6, R ~ 119. 
    # Loops are small enough.
    
    for it in range(iterations):
        # Update U
        for i in range(n):
            for j in range(m):
                # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # b[k] = 1
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j, :] = u

        # Update V
        for j in range(m):
            for k in range(p):
                # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # b[i] = 1
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k, :] = v

        # Update W
        for k in range(p):
            for i in range(n):
                # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # b[j] = 1
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i, :] = w

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Schoolbook
    U_sb, V_sb, W_sb = _make_schoolbook()
    
    # 2. Try to find better rank using ALS
    # Schoolbook rank is 120.
    # We iterate R from 119 down to 100 (arbitrary lower bound for search).
    # Since exact integer solutions are hard to hit, we run ALS and then snap.
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Use multiple random seeds to increase chance of finding a valid solution
    # The ALS landscape might have local minima, but for matrix multiplication,
    # the symmetric solution is often unique or there are known symmetries.
    
    # Strategy: 
    # For each R, try several random initializations.
    # If time allows, go lower.
    
    seeds = [42, 123, 999, 777, 1, 2, 3]
    
    for R in range(119, 99, -1):
        if time.time() > deadline:
            break
            
        # Calculate time remaining
        time_left = deadline - time.time()
        
        # Decide how many random restarts to do based on time
        # Rough estimate: each ALS run takes ~0.5s for R=119?
        # Let's just try 1-2 restarts if time is tight.
        
        max_restarts = 2 if time_left < 30 else 5
        
        for seed in seeds[:max_restarts]:
            if time.time() > deadline:
                break
                
            np.random.seed(seed)
            
            # Initialization
            # Scale matters. Schoolbook has 1s.
            # Random normal with small std dev is common.
            # Or start with schoolbook and perturb? 
            # If R < 120, we can't take schoolbook directly.
            # Let's use random normal * 0.5
            
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # Run ALS
            # We can break early if error is very low, but ALS might oscillate.
            # Let's run a fixed number of iterations, say 200, but check time.
            
            iterations = 0
            while iterations < 100:
                if time.time() > deadline - 1.0:
                    break
                
                _als_update(U, V, W, n, m, p, R, iterations=1)
                iterations += 1
                
                # Quick check: if error is small, try snapping
                # Calculate error efficiently? 
                # err = sum (U . (V*W) - 1)^2
                # This is O(n*m*p*R). 4*5*6*120 ~ 14400 ops. Very fast.
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0)**2
                
                if err < 1e-6:
                    # Try to snap
                    U_s, V_s, W_s = _snap_and_check(U, V, W)
                    if U_s is not None:
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
                # Also check if it's close to integer/half-integer already?
                # If it's not close to 0 error, it's unlikely to snap.
                
                # If R is small, it might converge faster.
                # If R is large, it might converge slower.
            
            # After full ALS run, check one last time
            U_s, V_s, W_s = _snap_and_check(U, V, W)
            if U_s is not None:
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
    return best_result

# Fallback in case construct is not called or for direct testing
if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    n, m, p = DIMENSIONS
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if not np.isclose(val, 1.0):
                    ok = False
                    print(f"Fail at {i},{j},{k}: {val}")
                    break
            if not ok: break
        if not ok: break
    if ok:
        print("Verification Passed")
    else:
        print("Verification Failed")
