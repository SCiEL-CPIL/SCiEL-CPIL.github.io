import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build the multiplication tensor entries
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    # Precompute constraint structure
    # For each a, list of (b, c) pairs
    a_to_bc = {}
    b_to_ac = {}
    c_to_ab = {}
    for (a, b, c) in tensor_entries:
        a_to_bc.setdefault(a, []).append((b, c))
        b_to_ac.setdefault(b, []).append((a, c))
        c_to_ab.setdefault(c, []).append((a, b))
    
    # Build the rank-39 Strassen decomposition
    rank39 = 5 * 7 + 1 * 4
    U39 = np.zeros((n * m, rank39))
    V39 = np.zeros((m * p, rank39))
    W39 = np.zeros((p * n, rank39))
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U39[0, r] = 1; U39[3, r] = 1
        V39[v_e, r] = 1; V39[v_h, r] = 1
        W39[w_c00, r] = 1; W39[w_c11, r] = 1
        r += 1
        
        # m2 = (c+d)e
        U39[2, r] = 1; U39[3, r] = 1
        V39[v_e, r] = 1
        W39[w_c10, r] = 1; W39[w_c11, r] = -1
        r += 1
        
        # m3 = a(f-h)
        U39[0, r] = 1
        V39[v_f, r] = 1; V39[v_h, r] = -1
        W39[w_c01, r] = 1; W39[w_c11, r] = 1
        r += 1
        
        # m4 = d(g-e)
        U39[3, r] = 1
        V39[v_g, r] = 1; V39[v_e, r] = -1
        W39[w_c00, r] = 1; W39[w_c10, r] = 1
        r += 1
        
        # m5 = (a+b)h
        U39[0, r] = 1; U39[1, r] = 1
        V39[v_h, r] = 1
        W39[w_c00, r] = -1; W39[w_c01, r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U39[2, r] = 1; U39[0, r] = -1
        V39[v_e, r] = 1; V39[v_f, r] = 1
        W39[w_c11, r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U39[1, r] = 1; U39[3, r] = -1
        V39[v_g, r] = 1; V39[v_h, r] = 1
        W39[w_c00, r] = 1
        r += 1

    # Single column k2 = 10 using schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    # ae
    U39[0, r] = 1
    V39[v_e2, r] = 1
    W39[w_c02, r] = 1
    r += 1
    
    # bg
    U39[1, r] = 1
    V39[v_g2, r] = 1
    W39[w_c02, r] = 1
    r += 1
    
    # ce
    U39[2, r] = 1
    V39[v_e2, r] = 1
    W39[w_c12, r] = 1
    r += 1
    
    # dg
    U39[3, r] = 1
    V39[v_g2, r] = 1
    W39[w_c12, r] = 1
    r += 1
    
    # Verify the rank-39 decomposition
    def verify(U, V, W, entries):
        for (a, b, c) in entries:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1) > 1e-10:
                return False
        return True

    if not verify(U39, V39, W39, tensor_entries):
        return {"U": U39.tolist(), "V": V39.tolist(), "W": W39.tolist()}
    
    # Try to find a lower rank decomposition using optimized ALS with rounding
    def try_rank(r, max_iter=80):
        """Try to find rank-r decomposition using vectorized ALS with rounding."""
        if r >= rank39:
            return None
        
        # Multiple random initializations for robustness
        for seed in range(3):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(42 + r + seed * 100)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize columns
            for mat in [U, V, W]:
                norms = np.linalg.norm(mat, axis=0, keepdims=True)
                norms[norms < 1e-10] = 1
                mat /= norms
            
            for iteration in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each a, solve least squares
                for a in range(n * m):
                    bc_list = a_to_bc.get(a, [])
                    if not bc_list:
                        continue
                    # Build A matrix: A[i, r] = V[b_i, r] * W[c_i, r]
                    A = np.array([[V[b, col] * W[c, col] for col in range(r)] for (b, c) in bc_list])
                    try:
                        U[a, :] = np.linalg.lstsq(A, np.ones(len(bc_list)), rcond=None)[0]
                    except:
                        pass
                
                # Update V: for each b, solve least squares
                for b in range(m * p):
                    ac_list = b_to_ac.get(b, [])
                    if not ac_list:
                        continue
                    A = np.array([[U[a, col] * W[c, col] for col in range(r)] for (a, c) in ac_list])
                    try:
                        V[b, :] = np.linalg.lstsq(A, np.ones(len(ac_list)), rcond=None)[0]
                    except:
                        pass
                
                # Update W: for each c, solve least squares
                for c in range(p * n):
                    ab_list = c_to_ab.get(c, [])
                    if not ab_list:
                        continue
                    A = np.array([[U[a, col] * V[b, col] for col in range(r)] for (a, b) in ab_list])
                    try:
                        W[c, :] = np.linalg.lstsq(A, np.ones(len(ab_list)), rcond=None)[0]
                    except:
                        pass
                
                # Check convergence
                max_err = 0.0
                for (a, b, c) in tensor_entries:
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err = abs(val - 1)
                    if err > max_err:
                        max_err = err
                
                if max_err < 1e-8:
                    # Try rounding to integers/half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    max_err_rounded = 0.0
                    for (a, b, c) in tensor_entries:
                        val = np.dot(U_rounded[a, :], V_rounded[b, :] * W_rounded[c, :])
                        err = abs(val - 1)
                        if err > max_err_rounded:
                            max_err_rounded = err
                    
                    if max_err_rounded < 1e-10:
                        return (U_rounded, V_rounded, W_rounded)
        
        return None
    
    # Try lower ranks
    for r in range(38, 30, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the known rank-39 decomposition
    return {"U": U39.tolist(), "V": V39.tolist(), "W": W39.tolist()}
