# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[alpha, beta, gamma] = 1 if alpha=(i,j), beta=(j,k), gamma=(k,i)
    # gamma = k*n + i (based on problem description W[k*n+i,r])
    # alpha = i*m + j
    # beta = j*p + k
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    T_flat = T.flatten()
    nm, mp, pn = n*m, m*p, p*n

    def check_solution(U, V, W, R):
        try:
            T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.linalg.norm(T_calc - T)
            return err
        except Exception:
            return float('inf')

    def snap_to_half_integer(x):
        return np.round(x * 2) / 2

    def alternating_projection(U, V, W, R, max_iters=100):
        """
        Alternating least squares on the integer/half-integer lattice.
        U: (nm, R), V: (mp, R), W: (pn, R)
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            for c in range(R):
                # Calculate residual excluding column c
                # T_res = T - sum_{c' != c} U_c' V_c' W_c'
                # This is expensive to compute from scratch each time.
                # For small R, it's fine.
                T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
                T_res = T - T_calc + np.einsum('a,b,d->abd', U[:,c], V[:,c], W[:,c])
                
                # We want to find u such that <u, V_c W_c> approx <T_res, V_c W_c>
                # Flatten V_c and W_c to a vector v_w of length mp*pn
                v_w = (V[:,c] * W[:,c]).flatten() # Wait, this is outer product flattened
                # The term is U_c (outer) V_c (outer) W_c
                # Flattened, the coefficient for U_c is (V_c outer W_c)
                # Let A_c be the matrix such that vec(U_c outer V_c outer W_c) = A_c vec(U_c)? No.
                # vec(U_c outer V_c outer W_c) is not linear in U_c in that way.
                # Actually, the tensor is sum_c U_c \otimes V_c \otimes W_c.
                # If we fix V_c, W_c, the contribution of U_c is U_c \otimes (V_c \otimes W_c).
                # So we project T_res onto the vector (V_c \otimes W_c).
                
                # Vector v_w_flat has shape (mp*pn,).
                # The term in the tensor for index (a, b, d) is U[a,c] * V[b,c] * W[d,c].
                # So for a fixed c, the pattern in the tensor is V[:,c] outer W[:,c] (shape mp, pn).
                # Let P_c = V[:,c] outer W[:,c].
                # Then T_calc[a,b,d] = sum_c U[a,c] * P_c[b,d].
                # We want to update U[:,c].
                # Residual R[a,b,d] = T[a,b,d] - sum_{c'!=c} U[a,c'] P_c'[b,d].
                # We want to minimize sum_{a,b,d} (R[a,b,d] - U[a,c] P_c[b,d])^2.
                # This is a linear least squares problem for U[:,c].
                # Let r_a = sum_{b,d} R[a,b,d] P_c[b,d] / sum_{b,d} P_c[b,d]^2 ?
                # No, it's:
                # For each a, we have a vector of length mp*pn: R[a, :, :] flattened.
                # We want to find scalar u_a such that u_a * P_c_flat approx R[a, :, :]_flat.
                # u_a = <R[a, :, :], P_c> / ||P_c||^2.
                
                P_c = np.outer(V[:,c], W[:,c])
                P_c_flat = P_c.flatten()
                norm_sq = np.dot(P_c_flat, P_c_flat)
                
                if norm_sq < 1e-10:
                    continue
                    
                # Compute residual excluding c
                # Efficiently: T_calc_full - U[:,c] outer P_c
                # T_calc_full is already computed above as T_calc
                # T_res = T - (T_calc - U[:,c] outer P_c)
                # T_res = T - T_calc + U[:,c] outer P_c
                
                # For each a, compute projection of T_res[a] onto P_c
                U_new_c = np.zeros(nm)
                for a in range(nm):
                    # T_res[a] is a slice of shape (mp, pn)
                    # We need <T_res[a], P_c>
                    # T_res[a] = T[a] - (T_calc[a] - U[a,c] P_c)
                    # <T_res[a], P_c> = <T[a], P_c> - <T_calc[a], P_c> + U[a,c] ||P_c||^2
                    # <T_calc[a], P_c> = sum_{c'} U[a,c'] <P_c', P_c>
                    # This requires computing inner products of all P_c' with P_c.
                    
                    # Simpler: Just compute T_res explicitly for this a?
                    # T_res = T - T_calc + U[:,c] outer P_c
                    # This is O(nm*mp*pn) per column c. R is ~30. Total O(R * nm*mp*pn).
                    # 30 * 30 * 15 * 10 = 135,000 ops. Very fast.
                    
                    pass # Placeholder, we will do it in bulk below

                # Bulk computation for U update
                # T_res = T - T_calc + np.einsum('ac,bd->abd', U[:,c], P_c) # Wait, U[:,c] is shape (nm,), P_c is (mp, pn).
                # np.einsum('a,bd->abd', U[:,c], P_c) gives (nm, mp, pn).
                
                T_res = T - T_calc + np.einsum('a,bd->abd', U[:,c], P_c)
                
                # Project each slice T_res[a] onto P_c
                # <T_res[a], P_c> = np.sum(T_res[a] * P_c)
                # This is np.einsum('abd,bd->a', T_res, P_c)
                projections = np.einsum('abd,bd->a', T_res, P_c)
                U[:,c] = projections / norm_sq

            # Snap U
            U = snap_to_half_integer(U)

            # Update V
            for c in range(R):
                if time.time() > deadline:
                    break
                T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
                P_c = np.outer(U[:,c], W[:,c]) # Shape (nm, pn)
                P_c_flat = P_c.flatten()
                norm_sq = np.dot(P_c_flat, P_c_flat)
                
                if norm_sq < 1e-10:
                    continue

                T_res = T - T_calc + np.einsum('b,ad->abd', V[:,c], P_c) # Wait, V[:,c] is (mp,), P_c is (nm, pn).
                # Term is V[b,c] * U[a,c] * W[d,c] = V[b,c] * P_c[a,d].
                # So np.einsum('b,ad->abd', V[:,c], P_c) is correct.
                
                projections = np.einsum('abd,ad->b', T_res, P_c)
                V[:,c] = projections / norm_sq

            V = snap_to_half_integer(V)

            # Update W
            for c in range(R):
                if time.time() > deadline:
                    break
                T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
                P_c = np.outer(U[:,c], V[:,c]) # Shape (nm, mp)
                P_c_flat = P_c.flatten()
                norm_sq = np.dot(P_c_flat, P_c_flat)
                
                if norm_sq < 1e-10:
                    continue

                T_res = T - T_calc + np.einsum('d,ab->abd', W[:,c], P_c)
                
                projections = np.einsum('abd,ab->d', T_res, P_c)
                W[:,c] = projections / norm_sq

            W = snap_to_half_integer(W)
            
            # Check convergence
            err = check_solution(U, V, W, R)
            if err < 1e-9:
                return U, V, W

        return None

    def try_rank(R):
        if time.time() > deadline:
            return None

        # Initialize with random integers
        # Use a few random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            U = np.random.randint(-1, 2, (nm, R)).astype(float)
            V = np.random.randint(-1, 2, (mp, R)).astype(float)
            W = np.random.randint(-1, 2, (pn, R)).astype(float)
            
            res = alternating_projection(U, V, W, R, max_iters=50)
            if res is not None:
                return res

        return None

    # Try ranks from schoolbook-1 downwards
    R_schoolbook = n * m * p
    best_result = None
    
    # We prioritize finding a solution for R=29, then 28, etc.
    # But random search might be slow.
    # Let's try R=29 first.
    for R in range(R_schoolbook - 1, R_schoolbook - 5, -1):
        if time.time() > deadline:
            break
        res = try_rank(R)
        if res is not None:
            best_result = res
            break

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return schoolbook
        rank = R_schoolbook
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
