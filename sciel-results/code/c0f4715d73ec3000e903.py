import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = time.time() + 135  # safety margin before 150s limit
    
    # Strassen 2x2x2 factors
    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),   # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),        # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),        # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),        # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),        # M7
    ]
    
    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def emit_block(a_cs, b_rs, b_cs, c_cs, off, U, V, W, rank):
            for s in range(7):
                col = off + s
                if col >= rank:
                    continue
                uc, vc, wc = strassen[s]
                for li, c in uc.items():
                    r, cc = li // 2, li % 2
                    U[r * m + a_cs + cc][col] += c
                for li, c in vc.items():
                    r, cc = li // 2, li % 2
                    V[(b_rs + r) * p + b_cs + cc][col] += c
                for li, c in wc.items():
                    r, cc = li // 2, li % 2
                    W[(c_cs + cc) * n + r][col] += c
        
        emit_block(0, 0, 0, 0, 0, U, V, W, rank)
        emit_block(0, 0, 2, 2, 7, U, V, W, rank)
        emit_block(2, 2, 0, 0, 14, U, V, W, rank)
        emit_block(2, 2, 2, 2, 21, U, V, W, rank)
        return U, V, W
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    # Start with rank-28 Strassen solution
    U28, V28, W28 = build_strassen_28()
    
    if not verify_decomposition(U28, V28, W28, n, m, p):
        # Fallback to rank-32
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try to reduce rank by finding linear dependencies among the 28 rank-1 tensors
    # The rank-1 tensors are T_r = u_r ⊗ v_r ⊗ w_r
    # If there's a linear dependence sum c_r T_r = 0, we can potentially reduce rank
    
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U28, dtype=float)
    V_np = np.array(V28, dtype=float)
    W_np = np.array(W28, dtype=float)
    
    # Flatten each rank-1 tensor
    # T_r[i,j,k] = U[i,r] * V[j,r] * W[k,r]
    # We want to check if the 28 vectors in R^(n*m*p) = R^32 are linearly independent
    
    tensor_dim = n * m * p  # 32
    T = np.zeros((28, tensor_dim))
    for r in range(28):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[r, i * m * p + j * p + k] = U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
    
    # Check rank of T using SVD
    try:
        U_svd, s_vals, Vt = np.linalg.svd(T, full_matrices=False)
        # Count significant singular values
        tol_svd = 1e-8
        rank_T = np.sum(s_vals > tol_svd)
        
        if rank_T < 28:
            # There's a linear dependence, potentially can reduce rank
            # Find null space
            null_space = Vt[rank_T:]
            # The null space gives coefficients for linear dependencies
            
            # Try to find a subset of columns that can be removed
            # This is complex; for now, just note the rank
            pass
    except:
        pass
    
    # Try a different approach: look for pairs of columns that are "close" to being linearly dependent
    # If column r2 ≈ alpha * column r1 in the tensor space, we might be able to merge them
    
    # Actually, let's try a more direct approach:
    # Randomly perturb the decomposition and see if we can find a lower-rank solution
    
    # For each target rank from 27 down to 16, try to find a valid decomposition
    # This is a hard problem, so we'll use a heuristic approach
    
    # Heuristic: try to find linear combinations of the 28 factors that give the same tensor
    # with fewer nonzero terms
    
    # Alternative: use the fact that the multiplication tensor for 2x4x4 might have
    # a known lower rank. Let's try to verify if rank 27 is possible.
    
    # For now, return the best solution we have (rank 28)
    # The search for lower rank would require more sophisticated algorithms
    
    # Try one more thing: check if any two columns are linearly dependent in a way
    # that allows removal
    
    # For each pair (r1, r2), check if there exists alpha such that
    # T[r2] ≈ alpha * T[r1]
    # If so, we can potentially eliminate one of them
    
    for r1 in range(28):
        if time.time() > deadline:
            break
        for r2 in range(r1 + 1, 28):
            if time.time() > deadline:
                break
            # Check if T[r2] is a scalar multiple of T[r1]
            # Use least squares to find alpha
            # T[r2] = alpha * T[r1]
            # alpha = dot(T[r2], T[r1]) / dot(T[r1], T[r1])
            dot_rr = np.dot(T[r1], T[r1])
            if abs(dot_rr) < 1e-10:
                continue
            alpha = np.dot(T[r2], T[r1]) / dot_rr
            residual = np.linalg.norm(T[r2] - alpha * T[r1])
            if residual < 1e-6:
                # T[r2] ≈ alpha * T[r1], so we can potentially eliminate r2
                # Create new decomposition without column r2
                new_rank = 27
                new_U = [row[:] for row in U28 if True]
                new_V = [row[:] for row in V28 if True]
                new_W = [row[:] for row in W28 if True]
                
                # Remove column r2 and adjust
                # Actually, if T[r2] = alpha * T[r1], then:
                # sum_r u_r ⊗ v_r ⊗ w_r = sum_{r≠r2} u_r ⊗ v_r ⊗ w_r + u_{r2} ⊗ v_{r2} ⊗ w_{r2}
                # = sum_{r≠r2} u_r ⊗ v_r ⊗ w_r + alpha * u_{r1} ⊗ v_{r1} ⊗ w_{r1}
                # = sum_{r≠r1,r2} u_r ⊗ v_r ⊗ w_r + (1+alpha) * u_{r1} ⊗ v_{r1} ⊗ w_{r1}
                # So we can merge r2 into r1 by scaling r1's factors by (1+alpha)
                
                # But this only works if the tensor T[r2] is exactly alpha * T[r1]
                # which means u_{r2} ⊗ v_{r2} ⊗ w_{r2} = alpha * (u_{r1} ⊗ v_{r1} ⊗ w_{r1})
                # This requires u_{r2} = c1 * u_{r1}, v_{r2} = c2 * v_{r1}, w_{r2} = c3 * w_{r1}
                # with c1*c2*c3 = alpha
                
                # Check if this is possible
                # For simplicity, just check if the vectors are proportional
                u1 = U_np[:, r1]
                u2 = U_np[:, r2]
                v1 = V_np[:, r1]
                v2 = V_np[:, r2]
                w1 = W_np[:, r1]
                w2 = W_np[:, r2]
                
                # Check if u2 is proportional to u1
                if np.linalg.norm(u1) > 1e-10:
                    c1 = np.dot(u2, u1) / np.dot(u1, u1)
                    if np.linalg.norm(u2 - c1 * u1) < 1e-6:
                        # Similarly check v and w
                        if np.linalg.norm(v1) > 1e-10:
                            c2 = np.dot(v2, v1) / np.dot(v1, v1)
                            if np.linalg.norm(v2 - c2 * v1) < 1e-6:
                                if np.linalg.norm(w1) > 1e-10:
                                    c3 = np.dot(w2, w1) / np.dot(w1, w1)
                                    if np.linalg.norm(w2 - c3 * w1) < 1e-6:
                                        # All three are proportional, so we can merge
                                        # New factor r1: scale by (1 + c1*c2*c3)
                                        scale = 1 + c1 * c2 * c3
                                        new_U = [row[:] for row in U28]
                                        new_V = [row[:] for row in V28]
                                        new_W = [row[:] for row in W28]
                                        
                                        # Remove column r2
                                        new_U = [row[:r2] + row[r2+1:] for row in new_U]
                                        new_V = [row[:r2] + row[r2+1:] for row in new_V]
                                        new_W = [row[:r2] + row[r2+1:] for row in new_W]
                                        
                                        # Scale column r1 by (1 + c1*c2*c3)
                                        # Note: r1 might be before or after r2 in the new array
                                        r1_new = r1 if r1 < r2 else r1 - 1
                                        for i in range(n * m):
                                            new_U[i][r1_new] *= scale
                                        for j in range(m * p):
                                            new_V[j][r1_new] *= scale
                                        for k in range(p * n):
                                            new_W[k][r1_new] *= scale
                                        
                                        # Verify
                                        if verify_decomposition(new_U, new_V, new_W, n, m, p):
                                            best_U, best_V, best_W = new_U, new_V, new_W
                                            best_rank = 27
                                            break
        if best_rank == 27:
            break
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
