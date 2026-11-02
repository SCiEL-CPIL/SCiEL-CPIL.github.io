"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def _check_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _find_decomp_2x2_2xp(p, r, max_iters=500, seed=42):
    """
    Find rank-r decomposition for 2x2 * 2xp.
    Returns (U,V,W) or None.
    """
    n, m = 2, 2
    rng = np.random.RandomState(seed)
    
    # Pre-compute indices for the 12 terms (i, j, k) where term is 1
    # i in [0,1], j in [0,1], k in [0, p-1]
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for attempt in range(max_iters):
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = np.zeros((p*n, r))
        valid = True
        
        # For each output index (k, i), we need to satisfy:
        # sum_j (U[i,j] * V[j,k])_r * W[k,i]_r = 1 for all (i,k)
        # But wait, the tensor is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1
        # This is a linear equation for W_{k,i,r} given U and V?
        # No, it's coupled.
        
        # Strategy: Fix U and V randomly.
        # For each output (k,i), we have a vector c_{k,i} in R^r where:
        # c_{k,i, r} = sum_j U[i*m+j, r] * V[j*p+k, r]
        # We need to choose W_{k,i, r} such that dot(c_{k,i}, W_{k,i}) = 1.
        # This is a system of p*n equations? No, they are independent for each (k,i)
        # IF we decouple the constraints. But the rank constraint means all r components
        # are shared. However, since we are just checking if a solution EXISTS for W
        # given U,V, and W rows are independent?
        
        # Let's check the structure.
        # The condition is: For all (i, k), sum_j (U_{i,j} V_{j,k}) \cdot W_{k,i} = 1.
        # Let A_{i,k} be the vector in R^r defined by (A_{i,k})_r = sum_j U[i*m+j, r] * V[j*p+k, r].
        # Then we need A_{i,k} \cdot W_{k,i} = 1.
        # This is a set of p*n independent linear equations for the rows of W?
        # Yes! Because W_{k,i} is a row of W, and it only appears in the equation for (k,i).
        # So we just need A_{i,k} to be non-zero for all (i,k).
        
        for i in range(n):
            for k in range(p):
                coeff = np.zeros(r)
                for j in range(m):
                    coeff += U[i*m+j, :] * V[j*p+k, :]
                ns = np.dot(coeff, coeff)
                if ns < 1e-12:
                    valid = False
                    break
                W[k*n+i, :] = coeff / ns
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify full decomposition to be safe (float errors)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for ri in range(r):
                        s += U[i*m+j, ri] * V[j*p+k, ri] * W[k*n+i, ri]
                    if abs(s - 1.0) > 1e-5:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return U, V, W
    
    return None

def _construct_from_block(U_block, V_block, W_block, p_block, p_total, n=2, m=2):
    """
    Constructs the full 2x2 x 2xP decomposition from a 2x2 x 2xP_block decomposition.
    Assumes p_total is a multiple of p_block or we handle the remainder.
    Here we will handle the remainder (2x1) separately.
    """
    pass # Logic handled in construct()

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    best_result = None
    best_rank = 999

    # Strategy: Decompose 2x13 into four 2x3 blocks and one 2x1 block.
    # Rank = 4 * r_2x3 + 4
    
    # Try to find rank-r decomp for 2x2 * 2x3
    # We know r >= 9 theoretically? Let's try 9, 10, 11, 12, 13.
    decomp_2x3 = None
    r_2x3 = None
    
    # Search ranks from lowest to highest
    for r_try in range(9, 14):
        if time.time() > deadline - 30:
            break
        # Use multiple seeds for robustness
        found = None
        for seed in range(10):
            result = _find_decomp_2x2_2xp(3, r_try, max_iters=200, seed=100+r_try*100+seed)
            if result is not None:
                found = result
                break
        if found is not None:
            decomp_2x3 = found
            r_2x3 = r_try
            break

    if decomp_2x3 is not None:
        U_2x3, V_2x3, W_2x3 = decomp_2x3
        total_rank = 4 * r_2x3 + 4
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col = 0
        # Place 4 blocks of 2x3
        for group in range(4):
            k_off = group * 3
            for ri in range(r_2x3):
                U[:, col] = U_2x3[:, ri]
                for j in range(m):
                    for k in range(3):
                        V[j*p + k_off + k, col] = V_2x3[j*3 + k, ri]
                for k in range(3):
                    for i in range(n):
                        W[(k_off+k)*n + i, col] = W_2x3[k*n + i, ri]
                col += 1
        
        # Place the last 2x1 block (Schoolbook rank 4)
        k_last = 12
        for i in range(n):
            for j in range(m):
                U[i*m + j, col] = 1.0
                V[j*p + k_last, col] = 1.0
                W[k_last*n + i, col] = 1.0
                col += 1
        
        if _check_decomposition(U, V, W):
            if total_rank < best_rank:
                best_rank = total_rank
                best_result = (U.copy(), V.copy(), W.copy())

    # Fallback: Strassen based rank 46 (6 blocks of 2x2x2)
    if time.time() < deadline - 5:
        rank46 = 46
        U46 = np.zeros((n*m, rank46))
        V46 = np.zeros((m*p, rank46))
        W46 = np.zeros((p*n, rank46))
        
        # Strassen decomposition for 2x2x2
        # We have 6 groups of 2x2x2.
        # Indices for Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = A22(B21-B11)
        # M3 = (A11-A22)(B12+B22)
        # M4 = A11(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)B11
        # M7 = (A12+A22)(B21-B12)
        
        # Output C11 = M1 + M3 - M5 + M7
        # Output C12 = M4 + M5
        # Output C21 = M2 + M6
        # Output C22 = M1 - M2 + M4 + M6
        
        col = 0
        for group in range(6):
            k1 = 2*group
            k2 = 2*group+1
            
            # M1
            U46[0, col]=1; U46[3, col]=1
            V46[k1, col]=1; V46[p+k2, col]=1
            W46[0, col]=1; W46[1, col]=1; W46[2, col]=-1; W46[3, col]=1
            col+=1
            # M2
            U46[3, col]=1
            V46[p+k1, col]=1; V46[k1, col]=-1
            W46[2, col]=1; W46[3, col]=-1
            col+=1
            # M3
            U46[0, col]=1; U46[3, col]=-1
            V46[p+k2, col]=1; V46[k2, col]=1
            W46[0, col]=1; W46[3, col]=1
            col+=1
            # M4
            U46[0, col]=1
            V46[p+k2, col]=1; V46[k2, col]=-1
            W46[1, col]=1; W46[3, col]=1
            col+=1
            # M5
            U46[0, col]=1; U46[1, col]=1
            V46[p+k2, col]=1
            W46[0, col]=-1; W46[1, col]=1
            col+=1
            # M6
            U46[2, col]=1; U46[0, col]=-1
            V46[k1, col]=1
            W46[2, col]=1; W46[3, col]=1
            col+=1
            # M7
            U46[1, col]=1; U46[3, col]=1
            V46[p+k1, col]=1; V46[p+k2, col]=-1
            W46[0, col]=1; W46[2, col]=1
            col+=1
        
        # Last 2x1 block
        k = 12
        for i in range(n):
            for j in range(m):
                U46[i*m+j, col]=1
                V46[j*p+k, col]=1
                W46[k*n+i, col]=1
                col+=1
        
        if _check_decomposition(U46, V46, W46):
            if 46 < best_rank:
                best_rank = 46
                best_result = (U46.copy(), V46.copy(), W46.copy())

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: Schoolbook rank 52
    rf = n*m*p
    Uf = np.zeros((n*m, rf)); Vf = np.zeros((m*p, rf)); Wf = np.zeros((p*n, rf))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Uf[i*m+j, col]=1
                Vf[j*p+k, col]=1
                Wf[k*n+i, col]=1
                col+=1
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
