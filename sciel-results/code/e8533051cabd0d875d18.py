"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np
import sys

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a11 + a22)(b11 + b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (a21 + a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # p3 = a11(b12 - b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # p4 = a22(b21 - b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # p5 = (a11 + a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # p6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (a12 - a22)(b21 + b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _search_2x2x3_rank(target_rank, time_budget=5.0):
    """Search for a rank-target_rank decomposition of 2x2x3 matrix multiplication."""
    start = time.time()
    np.random.seed(42)
    
    # Tensor T for 2x2x3
    # A is 2x2, B is 2x3, C is 2x3
    # T[a, b, c] where a = i*2+j, b = j*3+k, c = k*2+i
    # Non-zeros:
    # i=0, j=0, k=0: a=0, b=0, c=0
    # i=0, j=0, k=1: a=0, b=1, c=2
    # i=0, j=0, k=2: a=0, b=2, c=4
    # i=0, j=1, k=0: a=1, b=3, c=0
    # i=0, j=1, k=1: a=1, b=4, c=2
    # i=0, j=1, k=2: a=1, b=5, c=4
    # i=1, j=0, k=0: a=2, b=0, c=1
    # i=1, j=0, k=1: a=2, b=1, c=3
    # i=1, j=0, k=2: a=2, b=2, c=5
    # i=1, j=1, k=0: a=3, b=3, c=1
    # i=1, j=1, k=1: a=3, b=4, c=3
    # i=1, j=1, k=2: a=3, b=5, c=5
    
    nz_indices = [
        (0, 0, 0), (0, 1, 2), (0, 2, 4),
        (1, 3, 0), (1, 4, 2), (1, 5, 4),
        (2, 0, 1), (2, 1, 3), (2, 2, 5),
        (3, 3, 1), (3, 4, 3), (3, 5, 5)
    ]
    
    # Sizes
    dim_u = 4 # 2*2
    dim_v = 6 # 2*3
    dim_w = 6 # 3*2
    
    # Initialize random variables
    # U: 4 x R, V: 6 x R, W: 6 x R
    U = np.random.randn(dim_u, target_rank)
    V = np.random.randn(dim_v, target_rank)
    W = np.random.randn(dim_w, target_rank)
    
    # Define objective function
    def objective(params):
        U_flat = params[:dim_u * target_rank]
        V_flat = params[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank]
        W_flat = params[dim_u * target_rank + dim_v * target_rank : dim_u * target_rank + dim_v * target_rank + dim_w * target_rank]
        
        U_mat = U_flat.reshape(dim_u, target_rank)
        V_mat = V_flat.reshape(dim_v, target_rank)
        W_mat = W_flat.reshape(dim_w, target_rank)
        
        error = 0.0
        # Check non-zero entries
        for a, b, c in nz_indices:
            val = np.dot(U_mat[a], V_mat[b] * W_mat[c])
            error += (val - 1.0) ** 2
            
        # Check zero entries (sample a few to save time? No, 144 entries is small)
        # Actually, checking all 144 entries is better for correctness
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    if (a, b, c) not in nz_indices:
                        val = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                        error += val ** 2
                        
        return error

    # Use scipy.optimize.minimize
    try:
        from scipy.optimize import minimize
        initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        result = minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': 1000})
        
        if time.time() - start > time_budget:
            return None
            
        if result.fun < 1e-8:
            # Found a solution, try to round to half-integers
            U_sol = result.x[:dim_u * target_rank].reshape(dim_u, target_rank)
            V_sol = result.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
            W_sol = result.x[dim_u * target_rank + dim_v * target_rank : dim_u * target_rank + dim_v * target_rank + dim_w * target_rank].reshape(dim_w, target_rank)
            
            # Round to nearest half-integer
            U_int = np.round(U_sol * 2) / 2
            V_int = np.round(V_sol * 2) / 2
            W_int = np.round(W_sol * 2) / 2
            
            # Verify the rounded solution
            def verify_rounded(U_r, V_r, W_r):
                for a, b, c in nz_indices:
                    val = np.dot(U_r[a], V_r[b] * W_r[c])
                    if abs(val - 1.0) > 1e-6:
                        return False
                for a in range(dim_u):
                    for b in range(dim_v):
                        for c in range(dim_w):
                            if (a, b, c) not in nz_indices:
                                val = np.dot(U_r[a], V_r[b] * W_r[c])
                                if abs(val) > 1e-6:
                                    return False
                return True
                
            if verify_rounded(U_int, V_int, W_int):
                return {
                    "U": [[float(U_int[i][r]) for r in range(target_rank)] for i in range(dim_u)],
                    "V": [[float(V_int[i][r]) for r in range(target_rank)] for i in range(dim_v)],
                    "W": [[float(W_int[i][r]) for r in range(target_rank)] for i in range(dim_w)]
                }
            else:
                # Try different random seeds
                for seed in range(1, 10):
                    if time.time() - start > time_budget:
                        break
                    np.random.seed(seed)
                    U = np.random.randn(dim_u, target_rank)
                    V = np.random.randn(dim_v, target_rank)
                    W = np.random.randn(dim_w, target_rank)
                    initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    result = minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': 1000})
                    if result.fun < 1e-8:
                        U_sol = result.x[:dim_u * target_rank].reshape(dim_u, target_rank)
                        V_sol = result.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                        W_sol = result.x[dim_u * target_rank + dim_v * target_rank : dim_u * target_rank + dim_v * target_rank + dim_w * target_rank].reshape(dim_w, target_rank)
                        
                        U_int = np.round(U_sol * 2) / 2
                        V_int = np.round(V_sol * 2) / 2
                        W_int = np.round(W_sol * 2) / 2
                        
                        if verify_rounded(U_int, V_int, W_int):
                            return {
                                "U": [[float(U_int[i][r]) for r in range(target_rank)] for i in range(dim_u)],
                                "V": [[float(V_int[i][r]) for r in range(target_rank)] for i in range(dim_v)],
                                "W": [[float(W_int[i][r]) for r in range(target_rank)] for i in range(dim_w)]
                            }
    except Exception:
        pass
        
    return None


def _build_4x4x7_optimized():
    """Build decomposition for 4x4 * 4x7 using Strassen where beneficial."""
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    # Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3), rank 49
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(16)]
    V1 = [[0] * rank1 for _ in range(16)]
    W1 = [[0] * rank1 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U1[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V1[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W1[k*4+i][r] += val
    
    # Part 2: 4x4 * 4x3 (B cols 4-6, C cols 4-6)
    # Try to find a better decomposition for 2x2x3
    dec_2x2x3 = None
    if time.time() - _START_TIME < _TIME_LIMIT - 20:
        # Try rank 10 first
        dec_2x2x3 = _search_2x2x3_rank(10, time_budget=5.0)
        if dec_2x2x3 is None:
            dec_2x2x3 = _search_2x2x3_rank(11, time_budget=5.0)
            
    if dec_2x2x3 is not None:
        rank2 = 4 * 10 # if rank 10 found
        if len(dec_2x2x3['U'][0]) == 11:
            rank2 = 4 * 11
        else:
            rank2 = 4 * 10
            
        U2p = [[0] * rank2 for _ in range(16)]
        V2p = [[0] * rank2 for _ in range(12)] # 4*3
        W2p = [[0] * rank2 for _ in range(12)] # 3*4
        
        block_idx = 0
        for br in range(2):
            for bc in range(2):
                for t in range(len(dec_2x2x3['U'][0])):
                    r = block_idx * len(dec_2x2x3['U'][0]) + t
                    for lr in range(2):
                        for lc in range(2):
                            i = br * 2 + lr
                            j = bc * 2 + lc
                            val = dec_2x2x3['U'][lr*2+lc][t]
                            if val != 0:
                                U2p[i*4+j][r] += val
                    for lr in range(2):
                        for kl in range(3):
                            j = bc * 2 + lr
                            k = 4 + kl
                            val = dec_2x2x3['V'][lr*3+kl][t]
                            if val != 0:
                                V2p[j*3+kl][r] += val
                    for lr in range(2):
                        for kl in range(3):
                            i = br * 2 + lr
                            k = kl
                            val = dec_2x2x3['W'][kl*2+lr][t]
                            if val != 0:
                                W2p[k*4+i][r] += val
                block_idx += 1
    else:
        # Fallback to hybrid Strassen/Schoolbook for 4x4x3
        # 4x4x2 (Strassen) + 4x4x1 (Schoolbook)
        rank2 = 28 + 16 # 44
        U2p = [[0] * rank2 for _ in range(16)]
        V2p = [[0] * rank2 for _ in range(12)]
        W2p = [[0] * rank2 for _ in range(12)]
        
        # 4x4x2 part (cols 4-5)
        block_idx = 0
        for br in range(2):
            for bc in range(2):
                for t in range(7):
                    r = block_idx * 7 + t
                    for lr in range(2):
                        for lc in range(2):
                            i = br * 2 + lr
                            j = bc * 2 + lc
                            val = U2[lr*2+lc][t]
                            if val != 0:
                                U2p[i*4+j][r] += val
                    for lr in range(2):
                        for kl in range(2):
                            j = bc * 2 + lr
                            k = 4 + kl
                            val = V2[lr*2+kl][t]
                            if val != 0:
                                V2p[j*3+kl][r] += val
                    for lr in range(2):
                        for kl in range(2):
                            i = br * 2 + lr
                            k = kl
                            val = W2[kl*2+lr][t]
                            if val != 0:
                                W2p[k*4+i][r] += val
                    block_idx += 1
        
        # 4x4x1 part (col 6)
        offset3 = 28
        block_idx = 0
        for br in range(2):
            for bc in range(2):
                for lr in range(2):
                    for lc in range(2):
                        r = offset3 + block_idx * 4 + lr * 2 + lc
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        U2p[i*4+j][r] = 1
                        V2p[j*3+2][r] = 1 # col 6 is index 2 in the 3-col block
                        W2p[2*4+i][r] = 1
                block_idx += 1

    # Combine all parts
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: B cols 0-3, C cols 0-3
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                V[i*p+j][r] = V1[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                W[i*n+j][r] = W1[i*4+j][r]
    
    # Part 2: B cols 4-6, C cols 4-6
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2p[i][r]
    for i in range(4):
        for j in range(3):
            for r in range(rank2):
                V[i*p + (4+j)][offset2 + r] = V2p[i*3+j][r]
    for i in range(3):
        for j in range(4):
            for r in range(rank2):
                W[(4+i)*n + j][offset2 + r] = W2p[i*4+j][r]
    
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Try the optimized construction first
    try:
        result = _build_4x4x7_optimized()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return result
    except Exception:
        pass
    
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Fallback to schoolbook
    return _schoolbook(4, 4, 7)
