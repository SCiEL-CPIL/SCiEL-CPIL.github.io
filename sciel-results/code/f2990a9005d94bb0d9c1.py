import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Target Tensor Indices
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # T[a,b,c] = 1 for valid (i,j,k), else 0.
    # We only need to satisfy the 36 non-zero entries.
    
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets.append((a, b, c))
                
    num_targets = len(targets)
    
    def residuals(x, R):
        # x is flat vector of length R * (n*m + m*p + p*n)
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U = x[:size_U * R].reshape(size_U, R)
        V = x[size_U * R : size_U * R + size_V * R].reshape(size_V, R)
        W = x[size_U * R + size_V * R :].reshape(size_W, R)
        
        res = np.zeros(num_targets)
        for idx, (a, b, c) in enumerate(targets):
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            res[idx] = val - 1.0
        return res

    def check_solution(U, V, W, tol=1e-9):
        # Check if the solution is exact (within floating point precision)
        # And if entries are close to half-integers
        for idx, (a, b, c) in enumerate(targets):
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Check half-integer property
        for mat in [U, V, W]:
            # Distance to nearest half-integer
            # half_int = np.round(mat * 2) / 2
            # if np.max(np.abs(mat - half_int)) > 1e-5: return False
            # We don't strictly need to check this in the loop, 
            # but we round before returning.
            pass
        return True

    def try_rank(R, num_restarts=20):
        size_U = n * m
        size_V = m * p
        size_W = p * n
        total_size = R * (size_U + size_V + size_W)
        
        for r in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            # Scale matters. Strassen uses small integers.
            # Let's try uniform [-2, 2]
            x0 = np.random.uniform(-2, 2, total_size)
            
            try:
                # Use least_squares for better stability
                result = opt.least_squares(
                    residuals, 
                    x0, 
                    args=(R,), 
                    method='trf', 
                    ftol=1e-12, 
                    xtol=1e-12, 
                    gtol=1e-12,
                    max_nfev=10000
                )
                
                if result.success and result.cost < 1e-10:
                    # Reconstruct matrices
                    U = result.x[:size_U * R].reshape(size_U, R)
                    V = result.x[size_U * R : size_U * R + size_V * R].reshape(size_V, R)
                    W = result.x[size_U * R + size_V * R :].reshape(size_W, R)
                    
                    # Round to nearest half-integer
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    if check_solution(U_r, V_r, W_r):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        
            except Exception:
                pass
        return None

    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_solution = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook

    # Try to find better ranks
    # Known bounds suggest 23-32 is the range of interest.
    # We try from 32 downwards.
    # If we find one, we return it immediately as it's likely near-optimal 
    # or at least a significant improvement.
    
    # Note: 32 is the theoretical upper bound from 23+9.
    # It is highly probable that 32 is achievable.
    # 31, 30... are less certain.
    
    for R in range(32, 22, -1):
        if time.time() > deadline:
            break
        print(f"Trying rank {R}")
        sol = try_rank(R, num_restarts=10 if R > 25 else 5)
        if sol:
            best_solution = sol
            best_rank = R
            # We can stop here if we find a low rank, 
            # but continuing might find lower.
            # However, lower ranks are harder to find.
            # Given time constraints, returning the first found (highest R) 
            # is safer? No, we want MINIMAL rank.
            # So we should continue to lower R if possible.
            # But if we find 32, and then try 31 and fail, we keep 32.
            # If we find 26, we keep 26.
            pass

    return best_solution
