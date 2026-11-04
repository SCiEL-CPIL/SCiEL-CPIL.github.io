import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    def compute_error(U, V, W):
        A = np.zeros((n*m, m*p, p*n))
        for u in range(U.shape[1]):
            A += np.multiply.outer(np.multiply.outer(U[:, u], V[:, u]), W[:, u])
        diff = T - A
        return np.linalg.norm(diff)

    def als_update_U(U, V, W):
        # For each row a of U, solve M_a u_a = t_a
        new_U = np.zeros_like(U)
        for a in range(n*m):
            # M_a is (m*p*p*n, R)
            # M_a[(b, c), k] = V[b, k] * W[c, k]
            # We can construct M_a efficiently
            # V is (m*p, R), W is (p*n, R)
            # Outer product of V and W? No, V[b,k] W[c,k]
            # This is element-wise product of V[:, k] and W[:, k] for each k?
            # No, it's a matrix where column k is V[:, k] * W[:, k] (element-wise)
            # V[:, k] is length m*p. W[:, k] is length p*n.
            # We need to broadcast to (m*p, p*n)? No.
            # The index is (b, c).
            # M_a[:, k] = V[:, k] * W[:, k] ? No.
            # V[b, k] is a scalar. W[c, k] is a scalar.
            # So M_a[:, k] is the element-wise product of the vector V[:, k] (length mp) and W[:, k] (length pn)?
            # No, V[:, k] has length mp. W[:, k] has length pn.
            # We need a vector of length mp*pn.
            # The element at index (b, c) is V[b, k] * W[c, k].
            # This is the outer product of V[:, k] and W[:, k] flattened?
            # Yes. np.outer(V[:, k], W[:, k]).flatten()
            
            R = U.shape[1]
            M = np.zeros((m*p*p*n, R))
            for k in range(R):
                M[:, k] = np.outer(V[:, k], W[:, k]).flatten()
            
            t = T[a, :, :].flatten()
            
            # Solve M u = t
            # Use least squares
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(M, t, rcond=None)
                new_U[a, :] = sol
            except:
                new_U[a, :] = U[a, :]
        return new_U

    def als_update_V(U, V, W):
        new_V = np.zeros_like(V)
        for b in range(m*p):
            R = V.shape[1]
            M = np.zeros((n*m*p*n, R))
            for k in range(R):
                M[:, k] = np.outer(U[:, k], W[:, k]).flatten()
            
            t = T[:, b, :].flatten()
            
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(M, t, rcond=None)
                new_V[b, :] = sol
            except:
                new_V[b, :] = V[b, :]
        return new_V

    def als_update_W(U, V, W):
        new_W = np.zeros_like(W)
        for c in range(p*n):
            R = W.shape[1]
            M = np.zeros((n*m*m*p, R))
            for k in range(R):
                M[:, k] = np.outer(U[:, k], V[:, k]).flatten()
            
            t = T[:, :, c].flatten()
            
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(M, t, rcond=None)
                new_W[c, :] = sol
            except:
                new_W[c, :] = W[c, :]
        return new_W

    def run_als(U, V, W, iterations=20):
        for _ in range(iterations):
            U = als_update_U(U, V, W)
            V = als_update_V(U, V, W)
            W = als_update_W(U, V, W)
        return U, V, W

    start_time = time.time()
    deadline = start_time + 110
    
    # Try ranks 59 down to 54
    for rank in range(rank_school - 1, 53, -1):
        if time.time() > deadline:
            break
            
        found = False
        for _ in range(10): # Fewer restarts, but ALS is faster
            if time.time() > deadline:
                break
                
            # Initialize
            # Scale factors: T has entries 1.
            # U, V, W should be O(1).
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Run ALS
            U, V, W = run_als(U, V, W, iterations=30)
            
            err = compute_error(U, V, W)
            if err < 1e-5:
                # Try rounding
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                err_int = compute_error(U_int, V_int, W_int)
                if err_int < 1e-9:
                    best_rank = rank
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    found = True
                    break
        
        if found:
            # If we found a solution for this rank, we can stop if we want the minimum.
            # Since we iterate downwards, the first found is the best.
            # But we might find 59, then 58, etc.
            # Let's continue to see if we can go lower.
            # However, if we find 54, we can stop.
            if rank == 54:
                break

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
