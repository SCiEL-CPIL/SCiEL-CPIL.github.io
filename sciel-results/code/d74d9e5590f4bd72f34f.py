import time
import numpy as np
from fractions import Fraction
import scipy.linalg

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110

    def build_rank44():
        rank = 44
        U = [[Fraction(0)] * rank for _ in range(dim_u)]
        V = [[Fraction(0)] * rank for _ in range(dim_v)]
        W = [[Fraction(0)] * rank for _ in range(dim_w)]

        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col

        # Four 2x2x2 Strassen blocks
        col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
        col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
        col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
        col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)
        
        # Schoolbook for the 3rd row of A
        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1

        return U, V, W, col

    # Precompute expected tensor
    T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0

    def verify_frac(U, V, W):
        U_arr = np.array([[float(x) for x in row] for row in U], dtype=np.float64)
        V_arr = np.array([[float(x) for x in row] for row in V], dtype=np.float64)
        W_arr = np.array([[float(x) for x in row] for row in W], dtype=np.float64)
        
        if U_arr.size == 0:
            return np.allclose(np.zeros_like(T_expected), T_expected, atol=1e-10, rtol=1e-10)
            
        T_actual = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        return np.allclose(T_actual, T_expected, atol=1e-10, rtol=1e-10)

    U, V, W, actual_rank = build_rank44()
    
    if not verify_frac(U, V, W):
        # Fallback to schoolbook
        rank = 48
        U = [[0.0]*rank for _ in range(dim_u)]
        V = [[0.0]*rank for _ in range(dim_v)]
        W = [[0.0]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Convert to float arrays for linear algebra
    U_np = np.array([[float(x) for x in row] for row in U], dtype=np.float64)
    V_np = np.array([[float(x) for x in row] for row in V], dtype=np.float64)
    W_np = np.array([[float(x) for x in row] for row in W], dtype=np.float64)
    
    # Check for linear dependencies among the rank terms
    # We want to find c such that sum_r c_r U_r V_r W_r = 0
    # This is equivalent to finding the null space of the matrix M where M_{(u,v,w), r} = U_{u,r} V_{v,r} W_{w,r}
    
    if actual_rank > 1:
        # Reshape U, V, W to (dim, rank)
        # M is (dim_u * dim_v * dim_w) x rank
        # To save memory, we can sample or use SVD on the tensor directly?
        # No, the tensor is small (2304 x 44).
        
        U_flat = U_np.flatten() # No, U is (12, 44)
        # We need to construct M
        # M[i, r] where i is index in (u,v,w) space
        
        # Efficient construction:
        # M = np.einsum('ur,vr,wr->uvwr', U_np, V_np, W_np).reshape(-1, actual_rank)
        
        if time.time() < deadline:
            M = np.einsum('ur,vr,wr->uvwr', U_np, V_np, W_np).reshape(-1, actual_rank)
            
            # Find null space
            # SVD is more robust
            try:
                _, s, vh = np.linalg.svd(M, full_matrices=False)
                # Null space corresponds to singular values close to 0
                tol = 1e-10 * max(s)
                rank_M = np.sum(s > tol)
                
                if rank_M < actual_rank:
                    # There are (actual_rank - rank_M) linear dependencies
                    # We can project the factors onto the row space of M (or null space of M.T)
                    # The new factors should be in the span of the non-zero singular vectors of M.T
                    # Actually, we want to reduce the rank of the decomposition.
                    # If M has rank R, we can replace the rank-R' factors with a rank-R basis.
                    
                    # The null space of M.T is the space of valid "coefficient" vectors for the factors?
                    # No.
                    # If M c = 0, then sum c_r T_r = 0.
                    # This means the set of tensors {T_r} is linearly dependent.
                    # We can remove any set of vectors that spans the null space.
                    
                    # A simpler approach: Keep only the first `rank_M` factors?
                    # No, that might not span the same tensor space.
                    # We need to find a basis for the column space of M.
                    
                    # Let U_svd, S, Vh = np.linalg.svd(M)
                    # The column space of M is spanned by the first rank_M columns of U_svd.
                    # But we need to express the original factors in terms of a reduced set.
                    
                    # Actually, if the tensors are dependent, we can just drop the dependent ones?
                    # Not exactly. We need to drop a set such that the remaining ones are independent.
                    # The null space of M tells us which linear combinations are zero.
                    # We can use the right singular vectors corresponding to zero singular values.
                    
                    # Let's try a greedy approach guided by the null space.
                    # Or simply, if rank_M < actual_rank, we can try to remove factors one by one,
                    # but we know it's possible.
                    
                    # Let's just do the greedy removal, but we know it will succeed at least (actual_rank - rank_M) times.
                    pass
            except:
                pass

    # Perform greedy removal, which is safe and will now find the dependencies faster if we just loop
    # But we already know there are dependencies.
    
    current_U = U_np
    current_V = V_np
    current_W = W_np
    current_rank = actual_rank
    
    max_removals = 10
    removed = 0
    while removed < max_removals and current_rank > 1:
        if time.time() > deadline:
            break
        found = False
        for r_idx in range(current_rank):
            keep = [i for i in range(current_rank) if i != r_idx]
            
            U_test = current_U[:, keep]
            V_test = current_V[:, keep]
            W_test = current_W[:, keep]
            
            T_test = np.einsum('ir,jr,kr->ijk', U_test, V_test, W_test)
            if np.allclose(T_test, T_expected, atol=1e-10, rtol=1e-10):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                removed += 1
                found = True
                break
        if not found:
            break
            
    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
