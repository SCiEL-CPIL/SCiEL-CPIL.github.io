import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit

    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        # A indices: 00:0, 01:1, 10:3, 11:4
        # B indices: 00:0, 01:1, 10:4, 11:5
        # C indices (W): 00:0, 01:2, 10:1, 11:3
        
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        # B indices: 02:2, 03:3, 12:6, 13:7
        # C indices (W): 02:4, 03:6, 12:5, 13:7
        
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        # A indices: 02:2, 12:5
        # B indices: 20:8, 21:9, 22:10, 23:11
        # C indices: 00:0, 01:2, 02:4, 03:6 for i=0
        #            10:1, 11:3, 12:5, 13:7 for i=1
        
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    def verify(U, V, W, rank, tol=1e-9):
        """Check if the decomposition is exact using the trace convention W[k*n+i, r]."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T[i, j, k] == 1 for all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(arr, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                # Check if close to a half-integer
                closest = np.round(val * 2) / 2.0
                if abs(val - closest) < tol:
                    snapped[i, j] = closest
                else:
                    snapped[i, j] = val
        return snapped

    def to_lists(U, V, W):
        """Convert numpy arrays to lists of lists."""
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list

    def try_als(rank, max_iters=100, tol=1e-12):
        """
        Try to find a rank-`rank` decomposition using Alternating Least Squares (ALS).
        Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline:
            return None

        # Initialize with random matrices
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize columns to prevent scaling issues
        for r in range(rank):
            norm_U = np.linalg.norm(U[:, r])
            norm_V = np.linalg.norm(V[:, r])
            norm_W = np.linalg.norm(W[:, r])
            if norm_U > 1e-10 and norm_V > 1e-10 and norm_W > 1e-10:
                scale = (norm_U * norm_V * norm_W) ** (1.0/3.0)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale

        prev_err = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for r in range(rank):
                # Residual for column r: Target tensor - sum_{s!=r} U_s V_s W_s
                # This is expensive to compute directly. Instead, we solve a linear system.
                # The objective is to minimize || T - sum_r U_r V_r W_r ||_F
                # where T[i,j,k] = 1.
                # For a fixed r, we want to find U_r to minimize:
                # || T - U_r V_r W_r - R_r ||^2
                # where R_r is the residual from other terms.
                # This is a linear least squares problem for U_r.
                
                # Let's build the linear system for U.
                # We want to approximate the "flattened" residual.
                # Actually, it's easier to think of it as:
                # For each r, we fix V[:, r] and W[:, r] and want to find U[:, r].
                # The term in the objective involving U[:, r] is:
                # -2 * <T, U_r V_r W_r> + ||U_r V_r W_r||^2 + 2 * <R_r, U_r V_r W_r> ... 
                # This is getting complicated. Let's use a simpler ALS step.
                
                # Standard ALS for tensor decomposition:
                # Minimize || T - sum_r u_r v_r w_r ||^2
                # Update u_r: minimize || T - u_r (v_r w_r) - R_r ||^2
                # where R_r = sum_{s!=r} u_s v_s w_s.
                # This is equivalent to solving:
                # (v_r w_r)^T (v_r w_r) u_r = (v_r w_r)^T (T - R_r)
                
                # Let's compute v_r w_r as a vector of length n*m*p? No.
                # The tensor is 3-way.
                # Let's flatten T into a vector of length n*m*p.
                # T_flat[i*m*p + j*p + k] = 1.
                # The term u_r v_r w_r contributes to T_flat[i*m*p + j*p + k] the value:
                # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This is not a simple outer product in the flattened space.
                
                # Alternative: Use the fact that we are decomposing the matrix multiplication tensor.
                # T[i, j, k] = 1.
                # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
                
                # Let's try a different approach: Random restarts with local search on the rank-22 solution?
                # No, the goal is to find a LOWER rank.
                # The inspiration says ALS can find rank 18-21.
                # So we should try ranks 18, 19, 20, 21.
                
                # Let's implement a proper ALS step.
                # For a given r, we want to minimize:
                # || T - sum_s u_s v_s w_s ||^2
                # w.r.t. u_r.
                # This is a quadratic in u_r.
                # Let A_r be the matrix such that (A_r u_r)[i*m*p + j*p + k] = V[j*p+k, r] W[k*n+i, r]
                # Wait, the indices are coupled.
                # Let's just use numerical gradients or a simple iterative method.
                
                # Actually, let's use a simpler heuristic:
                # Start from the rank-22 solution, and try to zero out some columns or merge them.
                # Or, start from random and use ALS.
                
                # Let's try a different ALS formulation.
                # Fix V and W. Solve for U.
                # For each r, the contribution to T[i, j, k] is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
                # This is linear in U[i*m+j, r].
                # So for a fixed r, we can solve for the vector U[:, r] by least squares.
                # The "target" for this least squares is T - sum_{s!=r} u_s v_s w_s.
                # The "design matrix" for U[:, r] is such that:
                # (Design * U[:, r])[i*m*p + j*p + k] = V[j*p+k, r] * W[k*n+i, r]
                
                # Let's build this design matrix.
                # It has size (n*m*p) x (n*m).
                # This is 24 x 6 = 144 elements. Very small.
                
                # Build Design matrix D_r
                D_r = np.zeros((n * m * p, n * m))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_flat = i * m * p + j * p + k
                            idx_u = i * m + j
                            D_r[idx_flat, idx_u] = V[j * p + k, r] * W[k * n + i, r]
                
                # Target vector
                T_flat = np.ones(n * m * p)
                R_flat = np.zeros(n * m * p)
                for s in range(rank):
                    if s == r:
                        continue
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_flat = i * m * p + j * p + k
                                R_flat[idx_flat] += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                
                target = T_flat - R_flat
                
                # Solve D_r U[:, r] = target
                try:
                    U[:, r], residuals, rank_d, sv = np.linalg.lstsq(D_r, target, rcond=None)
                except:
                    return None
            
            # Update V
            for r in range(rank):
                D_r = np.zeros((n * m * p, m * p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_flat = i * m * p + j * p + k
                            idx_v = j * p + k
                            D_r[idx_flat, idx_v] = U[i * m + j, r] * W[k * n + i, r]
                
                R_flat = np.zeros(n * m * p)
                for s in range(rank):
                    if s == r:
                        continue
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_flat = i * m * p + j * p + k
                                R_flat[idx_flat] += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                
                target = T_flat - R_flat
                
                try:
                    V[:, r], residuals, rank_d, sv = np.linalg.lstsq(D_r, target, rcond=None)
                except:
                    return None

            # Update W
            for r in range(rank):
                D_r = np.zeros((n * m * p, p * n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_flat = i * m * p + j * p + k
                            idx_w = k * n + i
                            D_r[idx_flat, idx_w] = U[i * m + j, r] * V[j * p + k, r]
                
                R_flat = np.zeros(n * m * p)
                for s in range(rank):
                    if s == r:
                        continue
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_flat = i * m * p + j * p + k
                                R_flat[idx_flat] += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                
                target = T_flat - R_flat
                
                try:
                    W[:, r], residuals, rank_d, sv = np.linalg.lstsq(D_r, target, rcond=None)
                except:
                    return None

            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err += (s - 1.0) ** 2
            
            if abs(prev_err - err) < tol:
                if err < 1e-6:
                    # Check if it's exactly decomposable to half-integers
                    U_snapped = snap_to_half_integer(U)
                    V_snapped = snap_to_half_integer(V)
                    W_snapped = snap_to_half_integer(W)
                    if verify(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), rank):
                        return U_snapped, V_snapped, W_snapped
                return None
            
            prev_err = err

        return None

    # First, get the rank-22 solution as a fallback
    U22, V22, W22, rank22 = build_rank_22()
    
    # Try to find lower rank solutions using ALS
    # The inspiration suggests ranks 18-21 might be possible.
    # We try from lowest rank up.
    found = None
    for r in range(18, 22):
        if time.time() > deadline:
            break
        # Try multiple random restarts
        for restart in range(10):
            if time.time() > deadline:
                break
            res = try_als(r)
            if res is not None:
                found = res
                break
        if found is not None:
            break

    if found is not None:
        U, V, W = found
        U_list, V_list, W_list = to_lists(U, V, W)
        # Final verification
        if verify(U_list, V_list, W_list, len(U[0])):
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to rank 22
    U_list, V_list, W_list = to_lists(np.array(U22), np.array(V22), np.array(W22))
    return {"U": U_list, "V": V_list, "W": W_list}
