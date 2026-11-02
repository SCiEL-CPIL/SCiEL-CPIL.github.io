import numpy as np
from itertools import product, combinations
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank 18 as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

    # For this small problem, we'll do a systematic search
    # but with a time limit to ensure we return within 120s
    
    start_time = time.time()
    time_limit = 110  # seconds
    
    # Enumerate patterns with support size 1 or 2
    candidates = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                candidates.append(((a,), (b,), (c,)))
    
    # Add some size-2 patterns
    for su in combinations(range(n*m), 2):
        for sv in combinations(range(m*p), 2):
            for sw in combinations(range(p*n), 2):
                has_consistent = False
                for a in su:
                    i, j = divmod(a, m)
                    for b in sv:
                        j2, k = divmod(b, p)
                        if j == j2:
                            for c in sw:
                                k2, i2 = divmod(c, n)
                                if k == k2 and i == i2:
                                    has_consistent = True
                                    break
                            if has_consistent:
                                break
                    if has_consistent:
                        break
                if has_consistent:
                    candidates.append((su, sv, sw))
                    if time.time() - start_time > time_limit:
                        return fallback
    
    # Build vectors
    def pattern_to_vector(su, sv, sw):
        vec = np.zeros(n*m*p)
        for a in su:
            i, j = divmod(a, m)
            for b in sv:
                j2, k = divmod(b, p)
                if j == j2:
                    for c in sw:
                        k2, i2 = divmod(c, n)
                        if k == k2 and i == i2:
                            vec[i*m*p + j*p + k] += 1
        return vec
    
    pattern_vecs = [pattern_to_vector(su, sv, sw) for su, sv, sw in candidates]
    target = np.ones(n*m*p)
    
    # Try rank 15: find 15 patterns whose linear combination = target
    # Use a simple approach: try all subsets of size 15 from the first 25 candidates
    # This is C(25,15) = 3268760, which is manageable
    
    num_cands = min(25, len(candidates))
    
    for r in range(15, 10, -1):
        if time.time() - start_time > time_limit:
            return fallback
        
        # Try combinations
        from itertools import combinations as comb
        for combo in comb(range(num_cands), r):
            if time.time() - start_time > time_limit:
                return fallback
            
            A_sub = np.column_stack([pattern_vecs[i] for i in combo])
            try:
                x, res, rank_sub, s = np.linalg.lstsq(A_sub, target, rcond=None)
            except:
                continue
            
            # Check if x is close to half-integers
            x_rounded = np.round(x * 2) / 2
            if np.allclose(A_sub @ x_rounded, target, atol=1e-10):
                # Build factors
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                for r_idx, c_idx in enumerate(combo):
                    su, sv, sw = candidates[c_idx]
                    coeff = x_rounded[r_idx]
                    for a in su:
                        U[a][r_idx] = coeff
                    for b in sv:
                        V[b][r_idx] = coeff
                    for c in sw:
                        W[c][r_idx] = coeff
                
                # Verify
                U_arr = np.array(U)
                V_arr = np.array(V)
                W_arr = np.array(W)
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(r))
                            if abs(val - 1) > 1e-10:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    return {"U": U, "V": V, "W": W}
    
    return fallback
# EVOLVE-BLOCK-END
