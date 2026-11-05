"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 110.0   # hard wall-clock guard (evaluator kills at 150s)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Check that sum_r U[a,r]*V[b,r]*W[c,r] equals the matrix multiplication tensor."""
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    r = U.shape[1]
    nm = n * m
    mp = m * p
    pn = p * n
    
    # For each (a,b,c), check sum_r U[a,r]*V[b,r]*W[c,r] == delta
    # where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # The tensor T[i,j,k,i',j',k'] = delta_{j,j'} * delta_{i',i} ... 
    # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] for the three indices: A index (i,j), B index (j,k), C index (i,k)
    # T[A_idx, B_idx, C_idx] = 1 if B_idx's first index == A_idx's second index AND C_idx == (A_idx's first, B_idx's second)
    
    # Let's define:
    # A index a = i*m + j  (i in [0,n), j in [0,m))
    # B index b = j'*p + k  (j' in [0,m), k in [0,p))  
    # C index c = k'*n + i' (k' in [0,p), i' in [0,n))
    # T[a,b,c] = 1 if j == j' and i == i' and k == k'
    
    for a in range(nm):
        i = a // m
        j = a % m
        for b in range(mp):
            j2 = b // p
            k = b % p
            if j != j2:
                continue
            for c in range(pn):
                k2 = c // n
                i2 = c % n
                expected = 1 if (i == i2 and k == k2) else 0
                if expected == 0:
                    continue
                # Check sum_r U[a,r]*V[b,r]*W[c,r] == 1
                s = 0
                for rr in range(r):
                    uv = U[a, rr] * V[b, rr]
                    if uv != 0:
                        s += uv * W[c, rr]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy matrix multiplication."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    nm = n * m
    mp = m * p
    pn = p * n
    
    # For each c, compute sum_r U[:,r]*V[:,r]*W[c,r]
    # This is a tensor contraction. We can do:
    # For each c: result[a,b] = sum_r U[a,r]*V[b,r]*W[c,r]
    # = (U * W[c,:]) @ V.T  ... no, that's not right
    # Actually: sum_r U[a,r]*V[b,r]*W[c,r] = sum_r (U[a,r]*W[c,r]) * V[b,r]
    # = (U * W[c,:]) @ V.T  where U*W[c,:] means element-wise multiply columns
    
    # Let's do it differently: for each c, form M_c[a,b] = sum_r U[a,r]*V[b,r]*W[c,r]
    # = sum_r (U[a,r]*W[c,r]) * V[b,r]
    # Let UW_c = U * W[c,:] (element-wise, shape nm x r)
    # Then M_c = UW_c @ V.T  (shape nm x mp)
    
    for c in range(pn):
        k = c // n
        i = c % n
        # Expected: M_c[a,b] should be 1 if a=i*m+j and b=j*p+k for some j, else 0
        # So M_c should have 1s at positions (i*m+j, j*p+k) for j in [0,m)
        
        UW = U * W[c, :]  # element-wise multiply each column of U by W[c,:]
        M = UW @ V.T  # shape (nm, mp)
        
        # Check expected pattern
        for j in range(m):
            a = i * m + j
            b = j * p + k
            if abs(M[a, b] - 1.0) > 1e-9:
                return False
        # Check all other positions are 0
        for a in range(nm):
            a_i = a // m
            a_j = a % m
            for b in range(mp):
                b_j = b // p
                b_k = b % p
                if a_i == i and a_j == b_j and b_k == k:
                    continue  # already checked
                if abs(M[a, b]) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=100):
    """Try to reduce rank by finding linear dependencies."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]
    
    for _ in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        # Try to find a column that is a linear combination of others
        reduced = False
        for target in range(r):
            # Try to express column target as combination of others
            others = [i for i in range(r) if i != target]
            if len(others) == 0:
                continue
            
            # Check if U[:,target] is in span of U[:,others]
            U_others = U[:, others]
            U_target = U[:, target]
            
            # Check rank
            rank_others = np.linalg.matrix_rank(U_others)
            rank_with = np.linalg.matrix_rank(np.column_stack([U_others, U_target]))
            if rank_with > rank_others:
                continue  # U_target not in span
            
            # Similarly for V and W
            V_others = V[:, others]
            V_target = V[:, target]
            rank_others_v = np.linalg.matrix_rank(V_others)
            rank_with_v = np.linalg.matrix_rank(np.column_stack([V_others, V_target]))
            if rank_with_v > rank_others_v:
                continue
            
            W_others = W[:, others]
            W_target = W[:, target]
            rank_others_w = np.linalg.matrix_rank(W_others)
            rank_with_w = np.linalg.matrix_rank(np.column_stack([W_others, W_target]))
            if rank_with_w > rank_others_w:
                continue
            
            # All three are in span, so we can remove this column
            # But we need to be careful: the decomposition must remain valid
            # Actually, if each column is in the span of the others, we can remove it
            U = np.delete(U, target, axis=1)
            V = np.delete(V, target, axis=1)
            W = np.delete(W, target, axis=1)
            r -= 1
            reduced = True
            break
        
        if not reduced:
            break
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    full = n * m * p
    best = _schoolbook(n, m, p)
    best_rank = full
    
    def timed():
        return time.time() > _DEADLINE
    
    # Step 1: Start with schoolbook and try to reduce rank
    U, V, W = _try_reduce_rank(best["U"], best["V"], best["W"], n, m, p)
    if _verify_fast(U, V, W, n, m, p):
        best = {"U": U, "V": V, "W": W}
        best_rank = len(U[0])
    
    # Step 2: Try structured rank reduction
    # The matrix multiplication tensor T_{i,j,k}^{i',j',k'} = delta_{j,j'} delta_{i',i} delta_{k',k}
    # We can group terms by the "middle" index j
    # For each j, we have a 3x6 sub-matrix multiplication
    # This doesn't immediately help, but let's try random search with better initialization
    
    import random
    rng = random.Random(42)
    
    # Try various ranks
    for target_rank in range(best_rank - 1, max(20, best_rank - 20), -1):
        if timed():
            break
        
        max_attempts = 30
        for attempt in range(max_attempts):
            if timed():
                break
            
            # Initialize with schoolbook and randomly perturb
            U = np.zeros((nm, target_rank), dtype=float)
            V = np.zeros((mp, target_rank), dtype=float)
            W = np.zeros((pn, target_rank), dtype=float)
            
            # Copy first target_rank columns from schoolbook
            sb = _schoolbook(n, m, p)
            U[:, :] = np.array(sb["U"])[:, :target_rank]
            V[:, :] = np.array(sb["V"])[:, :target_rank]
            W[:, :] = np.array(sb["W"])[:, :target_rank]
            
            # Randomly perturb some entries
            for _ in range(10):
                a = rng.randint(0, nm - 1)
                b = rng.randint(0, mp - 1)
                c = rng.randint(0, pn - 1)
                r1 = rng.randint(0, target_rank - 1)
                r2 = rng.randint(0, target_rank - 1)
                U[a, r1] += rng.choice([-1, 1]) * 0.5
                V[b, r2] += rng.choice([-1, 1]) * 0.5
                W[c, r1] += rng.choice([-1, 1]) * 0.5
            
            # Try to reduce rank
            U_red, V_red, W_red = _try_reduce_rank(U.tolist(), V.tolist(), W.tolist(), n, m, p, max_iterations=20)
            
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank
                break
    
    # Step 3: Final verification
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        # Fall back to schoolbook
        best = _schoolbook(n, m, p)
    
    return best
# EVOLVE-BLOCK-END
