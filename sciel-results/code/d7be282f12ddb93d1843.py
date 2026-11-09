import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all ones positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val - 1) > tol:
                    return False
    
    # Check some zeros
    rng = np.random.RandomState(42)
    for _ in range(2000):
        a = rng.randint(0, n*m)
        b = rng.randint(0, m*p)
        c = rng.randint(0, p*n)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
        if abs(val) > tol:
            return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _find_3x3x3_rank7(time_limit):
    """Try to find a rank 7 decomposition for 3x3x3 matrix multiplication."""
    n = 3
    m = 3
    p = 3
    rank = 7
    
    # Tensor T for 3x3x3
    # a = 3i + j, b = 3j + k, c = 3k + i
    # T[a,b,c] = 1 if i_a=i_c, j_a=j_b, k_b=k_c else 0
    
    # We want U (9x7), V (9x7), W (9x7)
    # Minimize sum (sum_r U_ar V_br W_cr - T_abc)^2
    
    def residuals(x):
        U = x[0:63].reshape(9, 7)
        V = x[63:126].reshape(9, 7)
        W = x[126:189].reshape(9, 7)
        
        res = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = 3*i + j
                    b = 3*j + k
                    c = 3*k + i
                    # Compute sum_r U[a,r] V[b,r] W[c,r]
                    val = np.dot(U[a,:], V[b,:]*W[c,:])
                    target = 1.0
                    res.append(val - target)
                    
                    # Check zeros
                    # We only need to check a subset of zeros to keep it fast, 
                    # or all of them. 729 - 27 = 702 zeros.
                    # Checking all is better for accuracy.
        
        # Add zero constraints
        # To be efficient, we can precompute indices
        # But for now, let's just add the non-trivial ones or all.
        # 729 residuals is small.
        # Let's restructure to compute all 729 at once?
        # No, the loop is fine.
        
        # Actually, the above loop only checks the 27 ones.
        # We MUST check the zeros.
        # Let's rewrite residuals to include zeros.
        
        return res

    # Better residuals function
    def residuals_full(x):
        U = x[0:63].reshape(9, 7)
        V = x[63:126].reshape(9, 7)
        W = x[126:189].reshape(9, 7)
        
        res = np.zeros(729)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = 3*i + j
                    b = 3*j + k
                    c = 3*k + i
                    val = np.dot(U[a,:], V[b,:]*W[c,:])
                    res[idx] = val - 1.0
                    idx += 1
        
        # Fill zeros
        # a = 3i_a + j_a
        # b = 3j_b + k_b
        # c = 3k_c + i_c
        # Zero if not (i_a=i_c and j_a=j_b and k_b=k_c)
        
        # This loop is 729 iterations.
        # Let's just iterate all a,b,c
        idx = 0
        for a in range(9):
            i_a, j_a = divmod(a, 3)
            for b in range(9):
                j_b, k_b = divmod(b, 3)
                for c in range(9):
                    k_c, i_c = divmod(c, 3)
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        # Already handled in first loop?
                        # No, the first loop only iterates i,j,k which generates the 27 ones.
                        # The indices a,b,c in the first loop are specific.
                        # Let's just do one big loop for simplicity and correctness.
                        pass
                    else:
                        val = np.dot(U[a,:], V[b,:]*W[c,:])
                        res[idx] = val
                    idx += 1
                    
        # Wait, the first loop filled 27 entries.
        # The second loop fills 729 entries.
        # I should just do one loop for all 729.
        
        res = np.zeros(729)
        idx = 0
        for a in range(9):
            i_a, j_a = divmod(a, 3)
            for b in range(9):
                j_b, k_b = divmod(b, 3)
                for c in range(9):
                    k_c, i_c = divmod(c, 3)
                    val = np.dot(U[a,:], V[b,:]*W[c,:])
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        res[idx] = val - 1.0
                    else:
                        res[idx] = val
                    idx += 1
        return res

    start_time = time.time()
    rng = np.random.RandomState(42)
    
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
            
        x0 = rng.randn(189) * 0.1
        try:
            sol = least_squares(residuals_full, x0, method='lm', max_nfev=1000)
        except Exception:
            continue
            
        if sol.cost < 1e-12:
            # Found a solution
            U = sol.x[0:63].reshape(9, 7)
            V = sol.x[63:126].reshape(9, 7)
            W = sol.x[126:189].reshape(9, 7)
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify rounded solution
            # Construct full 3x3x6 solution to verify?
            # No, just verify the 3x3x3 part first.
            # If 3x3x3 is valid, 3x3x6 construction is valid.
            
            # Simple check of 3x3x3 rounded
            valid = True
            for a in range(9):
                i_a, j_a = divmod(a, 3)
                for b in range(9):
                    j_b, k_b = divmod(b, 3)
                    for c in range(9):
                        k_c, i_c = divmod(c, 3)
                        val = np.dot(U_r[a,:], V_r[b,:]*W_r[c,:])
                        if i_a == i_c and j_a == j_b and k_b == k_c:
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                        else:
                            if abs(val) > 1e-9:
                                valid = False
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U_r, V_r, W_r
                
    return None

def _build_3x3x6_from_3x3x3(U3, V3, W3):
    """
    Construct 3x3x6 decomposition from two 3x3x3 decompositions.
    U3, V3, W3 are 9x7, 9x7, 9x7.
    """
    n, m, p = 3, 3, 6
    rank = 14
    
    U = np.zeros((9, rank))
    V = np.zeros((18, rank))
    W = np.zeros((18, rank))
    
    # Block 1: k in 0,1,2
    # U[:, 0:7] = U3
    U[:, 0:7] = U3
    
    # V rows: b = 6j + k. 
    # For k in 0..2, b = 6j + k.
    # V3 rows: b3 = 3j + k.
    # So V[6j+k, 0:7] = V3[3j+k, :]
    for j in range(3):
        for k in range(3):
            b_full = 6*j + k
            b_sub = 3*j + k
            V[b_full, 0:7] = V3[b_sub, :]
            
    # W rows: c = 3k + i.
    # For k in 0..2, c = 3k + i.
    # W3 rows: c3 = 3k + i.
    # So W[3k+i, 0:7] = W3[3k+i, :]
    for k in range(3):
        for i in range(3):
            c_full = 3*k + i
            c_sub = 3*k + i
            W[c_full, 0:7] = W3[c_sub, :]
            
    # Block 2: k in 3,4,5
    # U[:, 7:14] = U3 (Same A)
    U[:, 7:14] = U3
    
    # V rows: b = 6j + k.
    # For k in 3..5, let k_sub = k - 3.
    # b_full = 6j + k.
    # b_sub = 3j + k_sub.
    # V[6j+k, 7:14] = V3[3j+k_sub, :]
    for j in range(3):
        for k in range(3, 6):
            b_full = 6*j + k
            k_sub = k - 3
            b_sub = 3*j + k_sub
            V[b_full, 7:14] = V3[b_sub, :]
            
    # W rows: c = 3k + i.
    # For k in 3..5, let k_sub = k - 3.
    # c_full = 3k + i.
    # c_sub = 3k_sub + i.
    # W[3k+i, 7:14] = W3[3k_sub+i, :]
    for k in range(3, 6):
        for i in range(3):
            c_full = 3*k + i
            k_sub = k - 3
            c_sub = 3*k_sub + i
            W[c_full, 7:14] = W3[c_sub, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_3x3x3_blocks():
    """Construct rank 18 solution using schoolbook for two 3x3x3 blocks."""
    n, m, p = 3, 3, 6
    rank = 18
    
    U = np.zeros((9, rank))
    V = np.zeros((18, rank))
    W = np.zeros((18, rank))
    
    # Block 1
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = 3*i + j
                b = 6*j + k
                c = 3*k + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    # Block 2
    for i in range(3):
        for j in range(3):
            for k in range(3, 6):
                a = 3*i + j
                b = 6*j + k
                c = 3*k + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Try rank 14
    if time.time() < deadline - 20:
        sol3 = _find_3x3x3_rank7(15)
        if sol3 is not None:
            U3, V3, W3 = sol3
            result = _build_3x3x6_from_3x3x3(U3, V3, W3)
            if _verify(result['U'], result['V'], result['W']):
                return result

    # Try rank 18
    if time.time() < deadline - 5:
        result = _schoolbook_3x3x3_blocks()
        if _verify(result['U'], result['V'], result['W']):
            return result

    # Fallback rank 54
    return _schoolbook()
