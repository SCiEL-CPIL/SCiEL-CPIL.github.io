import numpy as np
import scipy.optimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    num_a = n * m
    num_b = m * p
    num_c = p * n
    
    ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones.append((a, b, c))
                
    T_target = np.zeros((num_a, num_b, num_c))
    for a, b, c in ones:
        T_target[a, b, c] = 1.0

    def compute_tensor(U, V, W, R):
        T = np.zeros((num_a, num_b, num_c))
        for r in range(R):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return T

    def residuals(x, R):
        size_U = num_a * R
        size_V = num_b * R
        size_W = num_c * R
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:]
        
        U = U_flat.reshape(num_a, R)
        V = V_flat.reshape(num_b, R)
        W = W_flat.reshape(num_c, R)
        
        T = compute_tensor(U, V, W, R)
        return (T - T_target).flatten()

    # Schoolbook fallback
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(num_a)]
    V_sb = [[0] * rank_school for _ in range(num_b)]
    W_sb = [[0] * rank_school for _ in range(num_c)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
                
    best_rank = rank_school
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Try lower ranks
    # We will try up to 20 attempts per rank
    # To stay within time, we limit max_nfev
    for R in range(rank_school - 1, 15, -1):
        if R >= best_rank:
            continue
            
        found = False
        for attempt in range(20):
            # Random initialization
            # Use a mix of scales
            scale = 0.5 if attempt % 2 == 0 else 0.1
            x0 = np.random.randn(num_a * R + num_b * R + num_c * R) * scale
            
            try:
                res = scipy.optimize.least_squares(
                    residuals, x0, args=(R,), 
                    method='trf', 
                    max_nfev=1000, 
                    xtol=1e-8, ftol=1e-8, gtol=1e-8
                )
            except Exception:
                continue
                
            # Extract and round
            size_U = num_a * R
            size_V = num_b * R
            U_flat = res.x[:size_U]
            V_flat = res.x[size_U:size_U+size_V]
            W_flat = res.x[size_U+size_V:]
            
            U = U_flat.reshape(num_a, R)
            V = V_flat.reshape(num_b, R)
            W = W_flat.reshape(num_c, R)
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify
            T_calc = compute_tensor(U_r, V_r, W_r, R)
            if np.allclose(T_calc, T_target, atol=1e-9):
                best_rank = R
                best_U = U_r.tolist()
                best_V = V_r.tolist()
                best_W = W_r.tolist()
                found = True
                break # Found solution for this R, try lower R
                
        # If found, we continue to lower R. If not, we continue.
        
    return {"U": best_U, "V": best_V, "W": best_W}
