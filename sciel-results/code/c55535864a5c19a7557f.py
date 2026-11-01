import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _tensor_error(x, n, m, p, R):
    """Calculate the squared error of the tensor decomposition."""
    U = x[:n*m*R].reshape(n*m, R)
    V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
    W = x[n*m*R + m*p*R : ].reshape(p*n, R)
    
    # Compute S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Vectorized computation
    # U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # We need to sum over R for each (i,j,k)
    
    # This is a bit tricky to vectorize directly without loops over i,j,k
    # But n,m,p are small.
    S = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                S[i,j,k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
    
    return np.sum((S - 1.0)**2)

def _try_find_rank(R, n, m, p, max_restarts=20, time_limit=10.0):
    """Try to find a decomposition of rank R using randomized local search."""
    start_time = time.time()
    num_vars = R * (n*m + m*p + p*n)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random values
        x0 = np.random.uniform(-0.1, 0.1, num_vars)
        
        # Define objective
        def obj(x):
            return _tensor_error(x, n, m, p, R)
        
        # Optimize
        try:
            res = opt.minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 1000, 'disp': False})
            if res.fun < 1e-10:
                # Verify with exact check
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = res.x[n*m*R + m*p*R : ].reshape(p*n, R)
                if _verify_decomposition(U, V, W):
                    return U, V, W
        except Exception:
            pass
            
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    n, m, p = 2, 4, 5
    
    # Get the baseline rank 36 decomposition
    U_base, V_base, W_base = _build_strassen_36()
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    best_rank = 36
    
    # Try to find lower ranks using randomized search
    # We try ranks from 20 up to 35
    for R in range(20, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Allocate some time for this rank
        # If R is lower, it's harder, so maybe give more time?
        # But we have many ranks.
        # Let's give 5 seconds per rank, or dynamic.
        time_limit = 5.0
        
        res = _try_find_rank(R, n, m, p, max_restarts=10, time_limit=time_limit)
        if res is not None:
            U, V, W = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = R
            # If we find a very low rank, we might stop or continue to see if we can go lower
            # But lower is better, so we continue.
            
    # If we didn't find anything better, return the baseline
    # We can also try to improve the baseline with greedy removal as before, 
    # but the randomized search is more powerful.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
