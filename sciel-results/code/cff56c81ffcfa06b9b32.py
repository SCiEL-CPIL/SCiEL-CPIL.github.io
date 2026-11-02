import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # else 0
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try ranks from 14 upwards
    start_time = time.time()
    max_time = 100 # Leave 20s margin
    
    for rank in range(14, 19):
        if time.time() - start_time > max_time:
            break
            
        # Try multiple restarts for each rank
        for attempt in range(20):
            if time.time() - start_time > max_time:
                break
                
            # Random initialization
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            # Simple gradient descent or alternating least squares
            # Since the problem is small, we can use a simple iterative method
            # Minimize sum (T - sum U_r V_r W_r)^2
            
            # Reshape T for easier manipulation?
            # T is (dim_u, dim_v, dim_w)
            # We want to approximate T as sum_r U[:,r] outer V[:,r] outer W[:,r]
            
            # Use ALS (Alternating Least Squares)
            # For each r, update U[:,r] given V,W
            # This is not straightforward for rank decomposition of a 3-tensor directly with ALS in this form.
            # Instead, we can use a heuristic search or just try to solve the linear system if we fix two matrices?
            
            # Alternative: Use the fact that we can solve for W given U, V?
            # T is linear in W.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For a fixed U, V, we can solve for W?
            # The equation is: sum_r (U[:,r] V[:,r])^T W[:,r] = T? No.
            # It's a tensor product.
            
            # Let's use a simple local search with random perturbations and snapping.
            # Given the small dimensions, we might be able to find a solution by solving for integer entries directly?
            # No, search space is huge.
            
            # Let's try a different strategy:
            # We know the schoolbook is rank 18.
            # We can try to merge terms.
            
            # Actually, for 2x3 by 3x3, a rank 14 decomposition is known.
            # I will hardcode a known solution if I can find it in memory, or use a robust search.
            # Since I cannot guarantee finding it in 100s with random search, I will provide a robust search that might find rank 16 or 17 quickly, 
            # but I will prioritize trying to find rank 14.
            
            # Let's use a more structured search:
            # Initialize U, V, W with the schoolbook solution (rank 18) and then try to zero out 4 columns?
            # This is hard.
            
            # Let's try to use the "Lipshitz" method or similar.
            # Or simply:
            # If I can't find a better one quickly, I should return the best found.
            
            # However, the prompt asks for a "stronger algorithm".
            # I will implement a search for rank 14 using a library like `sympy` for exact arithmetic?
            # No, `sympy` is not guaranteed to be fast enough for search.
            
            # Let's stick to numerical search with snapping.
            # To improve success rate, I will use a larger number of restarts if time permits.
            
            # Optimization step:
            # We can use the fact that the tensor is sparse.
            # Most entries of T are 0.
            
            # Let's try to solve for W given U, V using least squares.
            # T is (dim_u, dim_v, dim_w).
            # We can flatten T to (dim_u * dim_v, dim_w).
            # Let M be a matrix of size (dim_u * dim_v, rank).
            # M[a*dim_v + b, r] = U[a, r] * V[b, r]
            # Then T_flat = M @ W_flat?
            # T_flat is (dim_u * dim_v, dim_w).
            # T_flat[i, c] = T[i//dim_v, i%dim_v, c]
            # We want T_flat = M @ W^T?
            # W is (dim_w, rank). W^T is (rank, dim_w).
            # So T_flat (dim_u*dim_v, dim_w) = M (dim_u*dim_v, rank) @ W^T (rank, dim_w).
            # This is a linear least squares problem for W^T given U, V.
            
            # Algorithm:
            # 1. Random U, V.
            # 2. Compute M from U, V.
            # 3. Solve for W^T = (M^T M)^{-1} M^T T_flat.
            # 4. Compute error.
            # 5. Update U, V using gradient descent or similar.
            
            # This is a standard ALS for CP decomposition.
            
            U_curr = np.random.randn(dim_u, rank)
            V_curr = np.random.randn(dim_v, rank)
            
            T_flat = T.reshape(dim_u * dim_v, dim_w)
            
            for step in range(100):
                # Update W
                M = np.zeros((dim_u * dim_v, rank))
                for r in range(rank):
                    M[:, r] = (U_curr[:, r] * V_curr[:, r]).reshape(-1)
                
                # Solve M @ W^T = T_flat
                # W^T = (M^T M)^{-1} M^T T_flat
                try:
                    W_t = np.linalg.lstsq(M, T_flat, rcond=None)[0]
                    W_curr = W_t.T
                except:
                    break
                
                # Compute error
                # Recreate T_approx
                T_approx = np.zeros((dim_u, dim_v, dim_w))
                for r in range(rank):
                    T_approx += np.einsum('a,b,c->abc', U_curr[:, r], V_curr[:, r], W_curr[:, r])
                
                err = np.linalg.norm(T - T_approx)
                if err < 1e-6:
                    # Snap to half-integers
                    U_snap = np.round(U_curr * 2) / 2
                    V_snap = np.round(V_curr * 2) / 2
                    W_snap = np.round(W_curr * 2) / 2
                    
                    # Verify exactness
                    T_check = np.zeros((dim_u, dim_v, dim_w))
                    for r in range(rank):
                        T_check += np.einsum('a,b,c->abc', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                    
                    if np.allclose(T_check, T, atol=1e-9):
                        # Convert to lists
                        U_list = U_snap.tolist()
                        V_list = V_snap.tolist()
                        W_list = W_snap.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                
                # Update U
                # T_approx = sum U_r outer V_r outer W_r
                # For U, we can fix V, W and solve for U.
                # T_flat is not the right shape for U directly.
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed b, c: T[:, b, c] = sum_r U[:, r] (V[b, r] W[c, r])
                # Let Z_b_c be a vector of size rank: Z_b_c[r] = V[b, r] W[c, r]
                # Then T[:, b, c] = U @ Z_b_c
                # We can solve for U using all b, c.
                
                # Construct a big matrix for U update
                # We want to minimize sum_{b,c} || T[:, b, c] - U @ Z_b_c ||^2
                # This is equivalent to solving a least squares problem.
                
                # Let's just use a simple gradient step for U and V to save implementation complexity.
                # Or, we can use the same trick:
                # For U:
                # T is (dim_u, dim_v, dim_w)
                # We can reshape T to (dim_u, dim_v * dim_w)
                # Let T_u = T.reshape(dim_u, -1)
                # We want T_u = U @ M_u^T ?
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Let M_u be a matrix of size (dim_v * dim_w, rank)
                # M_u[(b, c), r] = V[b, r] W[c, r]
                # Then T_u = U @ M_u.T
                
                M_u = np.zeros((dim_v * dim_w, rank))
                for b in range(dim_v):
                    for c in range(dim_w):
                        idx = b * dim_w + c
                        M_u[idx, :] = V_curr[b, :] * W_curr[c, :]
                
                T_u = T.reshape(dim_u, -1)
                
                try:
                    U_curr = np.linalg.lstsq(M_u.T, T_u, rcond=None)[0].T
                except:
                    pass
                
                # Update V similarly
                M_v = np.zeros((dim_u * dim_w, rank))
                for a in range(dim_u):
                    for c in range(dim_w):
                        idx = a * dim_w + c
                        M_v[idx, :] = U_curr[a, :] * W_curr[c, :]
                        
                T_v = T.transpose(1, 0, 2).reshape(dim_v, -1) # T[b, a, c]
                
                try:
                    V_curr = np.linalg.lstsq(M_v.T, T_v, rcond=None)[0].T
                except:
                    pass

    # If no better rank found, return schoolbook
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

# EVOLVE-BLOCK-END
