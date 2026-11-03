import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = time.time() + 130  # safety margin before 150s limit

    # Strassen 2x2x2 factors: each factor is (U_dict, V_dict, W_dict)
    # where U_dict maps (row,col) -> coeff for the 2x2 U block
    strassen = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),   # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),            # M2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),            # M3
        ({(1,0):1}, {(1,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),            # M4
        ({(0,0):1, (0,1):1}, {(1,0):1}, {(0,0):-1, (0,1):1}),            # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1}),            # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),            # M7
    ]

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

    def build_strassen_28():
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]

        def emit_block(a_row, b_row, b_col, c_col, off, U, V, W, rank):
            for s in range(7):
                col = off + s
                if col >= rank:
                    continue
                uc, vc, wc = strassen[s]
                for (dr, dc), c in uc.items():
                    U[(a_row + dr) * m + b_row + dc][col] += c
                for (dr, dc), c in vc.items():
                    V[(b_row + dr) * p + b_col + dc][col] += c
                for (dr, dc), c in wc.items():
                    W[(c_col + dc) * n + dr][col] += c

        emit_block(0, 0, 0, 0, 0, U, V, W, rank)
        emit_block(0, 0, 2, 2, 7, U, V, W, rank)
        emit_block(2, 2, 0, 0, 14, U, V, W, rank)
        emit_block(2, 2, 2, 2, 21, U, V, W, rank)
        return U, V, W

    U28, V28, W28 = build_strassen_28()

    if not verify_decomposition(U28, V28, W28, n, m, p):
        # Fallback to rank-32
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1.0
                    V[j * p + k][r] = 1.0
                    W[k * n + i][r] = 1.0
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28

    # Convert to numpy for efficient operations
    U_np = np.array(U28, dtype=float)
    V_np = np.array(V28, dtype=float)
    W_np = np.array(W28, dtype=float)

    # Build the tensor matrix T where T[r, i*m*p + j*p + k] = U[i,r]*V[j,r]*W[k,r]
    tensor_dim = n * m * p  # 32
    T = np.zeros((28, tensor_dim))
    for r in range(28):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[r, i * m * p + j * p + k] = U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]

    # Try to find a rank-27 decomposition by checking if any factor is a linear combination of others
    # that can be expressed as a valid rank-1 tensor
    # More practically: try to find a non-trivial linear relation among the 28 rank-1 tensors
    # and see if it allows reducing the rank

    # Check if the 28 rank-1 tensors are linearly independent
    try:
        U_svd, s_vals, Vt = np.linalg.svd(T, full_matrices=False)
        tol_svd = 1e-8
        rank_T = int(np.sum(s_vals > tol_svd))
    except:
        rank_T = 28

    # If rank_T < 28, there's a linear dependence
    if rank_T < 28 and time.time() < deadline:
        # The null space of T gives linear relations
        # null_vec: T @ null_vec = 0
        # This means sum_r null_vec[r] * T_r = 0
        # We can try to use this to eliminate one factor
        null_space = Vt[rank_T:]  # shape (28-rank_T, 28)
        
        for ns_idx in range(null_space.shape[0]):
            if time.time() > deadline:
                break
            nv = null_space[ns_idx]
            # Find the factor with the largest absolute coefficient
            max_idx = int(np.argmax(np.abs(nv)))
            max_coeff = nv[max_idx]
            if abs(max_coeff) < 1e-10:
                continue
            
            # Try to eliminate factor max_idx
            # New decomposition: for r != max_idx, scale factor r by 1
            # For the eliminated factor, we need to absorb its contribution
            # T[max_idx] = -sum_{r != max_idx} (nv[r]/nv[max_idx]) * T[r]
            # This means u_mx ⊗ v_mx ⊗ w_mx = -sum_{r != mx} (nv[r]/nv[mx]) * u_r ⊗ v_r ⊗ w_r
            # This is generally NOT a rank-1 tensor, so we can't simply eliminate it
            
            # Instead, try a different approach: look for pairs that can be merged
            pass

    # Try pairwise merging: for each pair (r1, r2), check if we can find scalars
    # a, b such that a*u_r1⊗v_r1⊗w_r1 + b*u_r2⊗v_r2⊗w_r2 = u_new⊗v_new⊗w_new
    # This requires the two rank-1 tensors to be "compatible"
    
    # A simpler heuristic: try to find if any two factors have proportional U vectors
    # and proportional V vectors (or W vectors), which would allow merging
    
    for r1 in range(28):
        if time.time() > deadline:
            break
        for r2 in range(r1 + 1, 28):
            if time.time() > deadline:
                break
            
            u1 = U_np[:, r1]
            u2 = U_np[:, r2]
            v1 = V_np[:, r1]
            v2 = V_np[:, r2]
            w1 = W_np[:, r1]
            w2 = W_np[:, r2]
            
            # Check if u2 is proportional to u1
            if np.linalg.norm(u1) < 1e-10 or np.linalg.norm(u2) < 1e-10:
                continue
            c1 = np.dot(u2, u1) / np.dot(u1, u1)
            if np.linalg.norm(u2 - c1 * u1) > 1e-6:
                continue
            
            # Check if v2 is proportional to v1
            if np.linalg.norm(v1) < 1e-10 or np.linalg.norm(v2) < 1e-10:
                continue
            c2 = np.dot(v2, v1) / np.dot(v1, v1)
            if np.linalg.norm(v2 - c2 * v1) > 1e-6:
                continue
            
            # Check if w2 is proportional to w1
            if np.linalg.norm(w1) < 1e-10 or np.linalg.norm(w2) < 1e-10:
                continue
            c3 = np.dot(w2, w1) / np.dot(w1, w1)
            if np.linalg.norm(w2 - c3 * w1) > 1e-6:
                continue
            
            # All three are proportional: we can merge
            scale = 1 + c1 * c2 * c3
            new_U = [row[:] for row in U28]
            new_V = [row[:] for row in V28]
            new_W = [row[:] for row in W28]
            
            # Remove column r2
            new_U = [row[:r2] + row[r2+1:] for row in new_U]
            new_V = [row[:r2] + row[r2+1:] for row in new_V]
            new_W = [row[:r2] + row[r2+1:] for row in new_W]
            
            # Scale column r1 (adjust index if r1 > r2)
            r1_new = r1 if r1 < r2 else r1 - 1
            for i in range(n * m):
                new_U[i][r1_new] *= scale
            for j in range(m * p):
                new_V[j][r1_new] *= scale
            for k in range(p * n):
                new_W[k][r1_new] *= scale
            
            if verify_decomposition(new_U, new_V, new_W, n, m, p):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = 27
                break
        if best_rank == 27:
            break

    # Try a more aggressive approach: use the structure of the 2x4x4 multiplication
    # to find a rank-27 decomposition
    # The 2x4x4 tensor can be seen as two 2x4x2 blocks or four 2x2x2 blocks
    # with some cross-terms
    
    # Try to construct a rank-27 decomposition directly
    # Use the fact that the (2,4,4) multiplication tensor has a specific structure
    # that might allow sharing more computations than the naive 4-block Strassen
    
    # Alternative: try to find a decomposition using 27 factors by solving a system
    # This is complex, so we'll use a heuristic search
    
    # Try random perturbations of the rank-28 solution to see if we can find a valid rank-27
    if time.time() < deadline - 30:
        np.random.seed(42)
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Try to eliminate one random factor and see if the remaining 27 can be adjusted
            elim_idx = np.random.randint(0, 28)
            
            # Create a modified decomposition without factor elim_idx
            new_U = [row[:] for row in U28]
            new_V = [row[:] for row in V28]
            new_W = [row[:] for row in W28]
            
            new_U = [row[:elim_idx] + row[elim_idx+1:] for row in new_U]
            new_V = [row[:elim_idx] + row[elim_idx+1:] for row in new_V]
            new_W = [row[:elim_idx] + row[elim_idx+1:] for row in new_W]
            
            # Check if the remaining 27 factors sum to the target tensor
            # If not, we need to adjust them
            # For now, just verify
            if verify_decomposition(new_U, new_V, new_W, n, m, p):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = 27
                break
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
