"""Matrix multiplication 2x3 by 3x9: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 115s safety margin within 120s limit

def _check_deadline():
    return time.time() >= _DEADLINE

def construct():
    """
    Return {'U', 'V', 'W'} with integer/half-integer entries that exactly
    decompose the 2x3 * 3x9 matrix multiplication tensor.
    
    The tensor T has components T[i,j,k] = delta_{j,k'} where:
    - i in [0, n-1] = [0, 1] (output row)
    - j in [0, m-1] = [0, 1, 2] (shared dimension / input col)
    - k in [0, p-1] = [0, ..., 8] (output col)
    
    Actually, for matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor is T = sum_{i,j,k} e_i^{(n)} (x) e_j^{(m)} (y) e_k^{(p)} (z) * delta... 
    No wait. The multiplication tensor for (n,m,p) is:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} A_{ij} B_{jk} C_{ik}
    
    So T = sum_{i,j,k} x_i y_j z_k where x_i is the coefficient of A_{ij}... 
    No, more precisely:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} (e_i in R^n) (e_j in R^m) (e_k in R^p)
    where the trilinear form is sum_{i,j,k} a_{ij} b_{jk} c_{ik}
    
    This is a rank-(n*m*p) tensor in general, but we seek a decomposition:
    T = sum_{r=1}^{R} u_r (v) v_r (w) w_r (z)
    where u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{p*n}
    
    Using the trace convention W[k*n+i, r], the output index is k*n + i.
    
    For the schoolbook: R = n*m*p = 54
    Each term r corresponds to (i, j, k):
    U[(i*m+j), r] = 1
    V[(j*p+k), r] = 1  
    W[(k*n+i), r] = 1
    
    The question is whether we can do better than 54.
    For 2x3x9, this is a relatively small problem. Let's try to find a lower rank.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Build the multiplication tensor as a 3D array T[n*m, m*p, p*n]
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid (i,j,k)
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try to find a low-rank decomposition using ALS (Alternating Least Squares)
    # over the rationals (represented as Fractions for exactness)
    
    def _tensor_contract(T, U, V, W):
        """Compute the trilinear form from factor matrices."""
        # T_approx[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        result = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for r in range(U.shape[1]):
            for a in range(n*m):
                if U[a, r] != 0:
                    for b in range(m*p):
                        if V[b, r] != 0:
                            uv = U[a, r] * V[b, r]
                            if uv != 0:
                                for c in range(p*n):
                                    if W[c, r] != 0:
                                        result[a, b, c] += uv * W[c, r]
        return result
    
    def _norm_error(T, U, V, W):
        """Compute max absolute error."""
        T_approx = _tensor_contract(T, U, V, W)
        return np.max(np.abs(T - T_approx))
    
    def _is_exact(T, U, V, W):
        """Check if decomposition is exact."""
        T_approx = _tensor_contract(T, U, V, W)
        return np.all(T == T_approx)
    
    def _to_fraction_list(arr):
        """Convert numpy array to list of Fractions (for integer/half-integer check)."""
        result = []
        for val in arr.flatten():
            f = Fraction(int(val), 1)
            result.append(f)
        return result
    
    def _validate_and_convert(U_frac, V_frac, W_frac):
        """Validate that entries are integer or half-integer, return as lists."""
        def check_half_int(frac):
            # frac is a Fraction; check if 2*frac is an integer
            two_frac = 2 * frac
            return two_frac.denominator == 1
        
        for name, mat in [('U', U_frac), ('V', V_frac), ('W', W_frac)]:
            for row in mat:
                for val in row:
                    if not check_half_int(val):
                        return None
        return {
            'U': [[str(val) if val.denominator != 1 else str(val.numerator) for val in row] for row in U_frac],
            'V': [[str(val) if val.denominator != 1 else str(val.numerator) for val in row] for row in V_frac],
            'W': [[str(val) if val.denominator != 1 else str(val.numerator) for val in row] for row in W_frac]
        }
    
    # Strategy: Try ALS with small integer coefficients, starting from various initializations
    # and different ranks. Start with rank 20 and go up.
    
    best_result = None
    
    # First, establish the schoolbook solution as a baseline (rank 54)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb, 'rank': rank_sb}
    
    # Try to find a lower rank decomposition using ALS
    # We'll work with integer-valued factors and try to minimize rank
    
    for target_rank in range(20, 54):
        if _check_deadline():
            break
        
        found = False
        n_restarts = 3 if target_rank <= 30 else 2
        
        for restart in range(n_restarts):
            if _check_deadline():
                break
            
            # Random initialization with small integers
            rng = np.random.RandomState(42 + restart * 1000 + target_rank)
            U = rng.randint(-2, 3, size=(n*m, target_rank))
            V = rng.randint(-2, 3, size=(m*p, target_rank))
            W = rng.randint(-2, 3, size=(p*n, target_rank))
            
            # ALS iterations
            n_iter = 30
            for it in range(n_iter):
                if _check_deadline():
                    break
                
                # Update U
                # For each a, U[a, :] should minimize ||T[a, :, :] - U[a,:]*(V*W)^T||^2
                # (V*W)[b, r] = V[b, r] * ... no, we need V[b,r]*W[c,r] summed over r
                # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed V, W: T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r])
                # Let M_r[b,c] = V[b,r]*W[c,r], then T[a,:,:] = sum_r U[a,r] M_r
                # So U[a,:] = argmin ||T[a,:,:] - sum_r U[a,r] M_r||^2
                # This is a linear system: for each a, solve (sum_r M_r^T M_r) U[a,:]^T = sum_r M_r^T T[a,:,:].flatten()
                
                # Build the Gram matrix G = sum_r M_r.reshape(-1)^T M_r.reshape(-1)
                # and rhs = sum_r M_r.reshape(-1)^T T[a,:,:].reshape(-1)
                
                vec_len = m * p * p * n
                G = np.zeros((target_rank, target_rank), dtype=np.int64)
                for r in range(target_rank):
                    Mr = (V[:, r] * W[:, r]).reshape(vec_len)
                    for r2 in range(target_rank):
                        Mr2 = (V[:, r2] * W[:, r2]).reshape(vec_len)
                        G[r, r2] = np.dot(Mr, Mr2)
                
                # Solve for U
                try:
                    G_float = G.astype(np.float64)
                    U_sol = np.zeros((n*m, target_rank), dtype=np.float64)
                    for a in range(n*m):
                        rhs = np.zeros(target_rank, dtype=np.float64)
                        Ta_flat = T[a, :, :].reshape(vec_len)
                        for r in range(target_rank):
                            Mr = (V[:, r] * W[:, r]).reshape(vec_len)
                            rhs[r] = np.dot(Mr, Ta_flat)
                        # Regularized solve
                        U_sol[a, :] = np.linalg.solve(G_float + 1e-8 * np.eye(target_rank), rhs)
                    
                    # Round to integers
                    U_new = np.round(U_sol).astype(np.int64)
                    # Check error
                    err = 0
                    for a in range(n*m):
                        for b in range(m*p):
                            for c in range(p*n):
                                approx = 0
                                for r in range(target_rank):
                                    approx += U_new[a, r] * V[b, r] * W[c, r]
                                err += (T[a,b,c] - approx) ** 2
                        if err > 100:
                            break
                    if err == 0:
                        U = U_new
                        found = True
                        break
                    U = U_new
                except np.linalg.LinAlgError:
                    continue
                
                # Update V
                G2 = np.zeros((target_rank, target_rank), dtype=np.int64)
                for r in range(target_rank):
                    Mr = (U[:, r] * W[:, r]).reshape(n*m * p*n)
                    for r2 in range(target_rank):
                        Mr2 = (U[:, r2] * W[:, r2]).reshape(n*m * p*n)
                        G2[r, r2] = np.dot(Mr, Mr2)
                
                try:
                    G2_float = G2.astype(np.float64)
                    V_sol = np.zeros((m*p, target_rank), dtype=np.float64)
                    for b in range(m*p):
                        rhs = np.zeros(target_rank, dtype=np.float64)
                        Tb_flat = T[:, b, :].reshape(vec_len)
                        for r in range(target_rank):
                            Mr = (U[:, r] * W[:, r]).reshape(vec_len)
                            rhs[r] = np.dot(Mr, Tb_flat)
                        V_sol[b, :] = np.linalg.solve(G2_float + 1e-8 * np.eye(target_rank), rhs)
                    
                    V_new = np.round(V_sol).astype(np.int64)
                    err = 0
                    for a in range(n*m):
                        for b in range(m*p):
                            for c in range(p*n):
                                approx = 0
                                for r in range(target_rank):
                                    approx += U[a, r] * V_new[b, r] * W[c, r]
                                err += (T[a,b,c] - approx) ** 2
                        if err > 100:
                            break
                    if err == 0:
                        V = V_new
                        found = True
                        break
                    V = V_new
                except np.linalg.LinAlgError:
                    continue
                
                # Update W
                G3 = np.zeros((target_rank, target_rank), dtype=np.int64)
                for r in range(target_rank):
                    Mr = (U[:, r] * V[:, r]).reshape(n*m * m*p)
                    for r2 in range(target_rank):
                        Mr2 = (U[:, r2] * V[:, r2]).reshape(n*m * m*p)
                        G3[r, r2] = np.dot(Mr, Mr2)
                
                try:
                    G3_float = G3.astype(np.float64)
                    W_sol = np.zeros((p*n, target_rank), dtype=np.float64)
                    for c in range(p*n):
                        rhs = np.zeros(target_rank, dtype=np.float64)
                        Tc_flat = T[:, :, c].reshape(vec_len)
                        for r in range(target_rank):
                            Mr = (U[:, r] * V[:, r]).reshape(vec_len)
                            rhs[r] = np.dot(Mr, Tc_flat)
                        W_sol[c, :] = np.linalg.solve(G3_float + 1e-8 * np.eye(target_rank), rhs)
                    
                    W_new = np.round(W_sol).astype(np.int64)
                    err = 0
                    for a in range(n*m):
                        for b in range(m*p):
                            for c in range(p*n):
                                approx = 0
                                for r in range(target_rank):
                                    approx += U[a, r] * V[b, r] * W_new[c, r]
                                err += (T[a,b,c] - approx) ** 2
                        if err > 100:
                            break
                    if err == 0:
                        W = W_new
                        found = True
                        break
                    W = W_new
                except np.linalg.LinAlgError:
                    continue
            
            if found:
                # Verify exactness
                if _is_exact(T, U, V, W):
                    # Check half-integer constraint
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    # All integers, so valid
                    best_result = {'U': U_list, 'V': V_list, 'W': W_list, 'rank': target_rank}
                    break
        
        if found and _check_deadline():
            break
    
    # Return the best result found
    if best_result:
        return {'U': best_result['U'], 'V': best_result['V'], 'W': best_result['W']}
    
    # Fallback: schoolbook
    return {'U': U_sb, 'V': V_sb, 'W': W_sb}
