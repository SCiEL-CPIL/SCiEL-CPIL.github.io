"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Construct the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Helper to verify if a set of factors U, V, W (lists of vectors) exactly matches T
    def verify(U, V, W):
        rank = len(U)
        T_rec = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[r], V[r], W[r])
        return np.allclose(T_rec, T, atol=1e-9)

    # Helper to convert float factors to "integer/half-integer" (multiples of 0.5)
    # and verify again with the rounded values.
    def to_half_integer(U, V, W):
        U_hi = np.array(U)
        V_hi = np.array(V)
        W_hi = np.array(W)
        
        # Round to nearest 0.5
        U_hi = np.round(U_hi * 2) / 2.0
        V_hi = np.round(V_hi * 2) / 2.0
        W_hi = np.round(W_hi * 2) / 2.0
        
        # Check if rounding was effective (i.e. values are actually half-integers)
        # If not, the solution might be irrational or require higher precision, 
        # but for matrix multiplication tensors, minimal rank decompositions 
        # often have small integer/rational entries.
        
        if verify(U_hi.tolist(), V_hi.tolist(), W_hi.tolist()):
            return [U_hi.tolist(), V_hi.tolist(), W_hi.tolist()]
        return None

    # 1. Construct the known Rank 42 solution (Strassen per 2x2 block)
    # We partition k=0..11 into 6 pairs.
    # For each pair, use Strassen's 7 products.
    
    # Strassen factors for 2x2 x 2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> mapped to (k,i) in local 2x2
    
    # The 7 terms (u, v, w) as sparse dictionaries of index:coeff
    # Local indices:
    # U: i*m+j -> 0,1,2,3
    # V: j*p+k -> 0,1,2,3 (local p=2)
    # W: k*n+i -> 0,1,2,3 (local n=2)
    
    strassen_terms = [
        # M1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2
        ({2:1, 3:1}, {0:1}, {2:1}),
        # M3
        ({0:1}, {1:1, 3:-1}, {1:1}),
        # M4
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        # M5
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # M6
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # M7
        ({1:1, 3:-1}, {2:1, 3:1}, {0:-1, 3:1})
    ]
    
    U42 = []
    V42 = []
    W42 = []
    
    for t in range(6):
        for u_dict, v_dict, w_dict in strassen_terms:
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            for idx, coeff in u_dict.items():
                i, j = idx // m, idx % m
                u_vec[i*m + j] = coeff
                
            for idx, coeff in v_dict.items():
                j, k_loc = idx // 2, idx % 2
                k = 2*t + k_loc
                v_vec[j*p + k] = coeff
                
            for idx, coeff in w_dict.items():
                k_loc, i = idx // 2, idx % 2
                k = 2*t + k_loc
                w_vec[k*n + i] = coeff
                
            U42.append(u_vec)
            V42.append(v_vec)
            W42.append(w_vec)
            
    U42 = np.array(U42)
    V42 = np.array(V42)
    W42 = np.array(W42)
    
    # Verify Rank 42
    if verify(U42.tolist(), V42.tolist(), W42.tolist()):
        best_result = {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
        best_rank = 42
    else:
        # Fallback to schoolbook 48 if Strassen construction is flawed (shouldn't happen)
        U48 = []
        V48 = []
        W48 = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n*m); u[i*m+j] = 1
                    v = np.zeros(m*p); v[j*p+k] = 1
                    w = np.zeros(p*n); w[k*n+i] = 1
                    U48.append(u); V48.append(v); W48.append(w)
        best_result = {"U": np.array(U48).tolist(), "V": np.array(V48).tolist(), "W": np.array(W48).tolist()}
        best_rank = 48
        U42 = np.array(U48)
        V42 = np.array(V48)
        W42 = np.array(W48)

    # 2. Attempt to reduce rank using optimization
    # We try to find a decomposition of rank R < best_rank.
    # We start from the top R terms of the rank 42 solution (or random) and optimize.
    # Since the space of rank R decompositions is non-convex, we use multiple restarts.
    
    # Flatten factors for optimization
    # x = [U_0, U_1, ..., U_{R-1}, V_0, ..., V_{R-1}, W_0, ..., W_{R-1}]
    # Size: R * (n*m + m*p + p*n)
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_dim = dim_u + dim_v + dim_w
    
    def objective(x, R):
        U_flat = x[:R*dim_u]
        V_flat = x[R*dim_u:R*dim_u+R*dim_v]
        W_flat = x[R*dim_u+R*dim_v:]
        
        U_mat = U_flat.reshape(R, dim_u)
        V_mat = V_flat.reshape(R, dim_v)
        W_mat = W_flat.reshape(R, dim_w)
        
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(R):
            T_rec += np.einsum('i,j,k->ijk', U_mat[r], V_mat[r], W_mat[r])
            
        # Residual
        return (T_rec - T).flatten()

    # Try ranks from 41 down to 30 (or lower if time permits)
    # We prioritize trying ranks close to 42 first.
    
    min_rank_to_try = 30
    if time.time() > deadline - 30:
        min_rank_to_try = 41 # Only try 41 if time is tight
        
    found_better = False
    
    for R in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 10:
            break
            
        # Multiple restarts for each rank
        # We can try initializing from random, or from subsets of the known solution.
        # Using subsets of the known solution is a good heuristic if the lower rank
        # solution is a "projection" or "merging" of the higher rank one.
        
        success = False
        num_restarts = 20 if R > 40 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            if restart == 0:
                # Initialize with first R terms of the known solution
                x0 = np.concatenate([
                    U42[:R].flatten(),
                    V42[:R].flatten(),
                    W42[:R].flatten()
                ])
            else:
                # Random initialization
                x0 = np.random.randn(R * total_dim) * 0.1
                # Add a bit of the known solution to guide it?
                # x0 += np.random.randn(R * total_dim) * 0.01
                
            try:
                res = least_squares(objective, x0, args=(R,), method='lm', max_nfev=5000)
                
                if res.cost < 1e-10:
                    # Check exactness with rational rounding
                    U_opt = res.x[:R*dim_u].reshape(R, dim_u)
                    V_opt = res.x[R*dim_u:R*dim_u+R*dim_v].reshape(R, dim_v)
                    W_opt = res.x[R*dim_u+R*dim_v:].reshape(R, dim_w)
                    
                    hi_res = to_half_integer(U_opt, V_opt, W_opt)
                    if hi_res is not None:
                        best_result = {"U": hi_res[0], "V": hi_res[1], "W": hi_res[2]}
                        best_rank = R
                        found_better = True
                        success = True
                        # If we found a better rank, we can break out of restarts for this R
                        # and try to go even lower
                        break
            except Exception:
                continue
        
        if success:
            # Continue to try lower ranks
            continue
        else:
            # If we failed to find rank R, it's likely we can't find R-1 easily either
            # without more sophisticated search, so we stop.
            # However, sometimes R=41 fails but R=40 works? Unlikely without a jump.
            # We'll stop to save time.
            break

    # Final check: if we didn't find anything better, return the best known (42 or 48)
    # But wait, we should double check if the initial 42 is valid and return it.
    
    return best_result

# EVOLVE-BLOCK-END
