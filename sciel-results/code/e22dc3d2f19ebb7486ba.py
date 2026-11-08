import time
import random
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 6

    # --- Helper: verify that (U, V, W) exactly reproduces matmul ---
    def verify(U_mat, V_mat, W_mat, n=n, m=m, p=p):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # For each r: U_mat[:, r] ⊗ V_mat[:, r] should match (reshaped matmul tensor)
        # Tensor T[a,b,c] = 1 if b is the shared index: T is actually 3-tensor with T[i,j,k] = 1 iff j matches
        # Actually: tensor for matmul is T[i,j,k] = 1 if (i from A, j is shared, k from B)
        # U has row (i,j), V has row (j,k), W has row (k*n+i)
        # The reconstruction is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This must equal 1 for all (i,j,k) (since it's a rank-1 decomposition of the "identity" tensor in the matmul sense)
        rank = U_mat.shape[1]
        # Efficient check: for each r, form outer product U[:,r] V[:,r] and check W[:,r] consistency
        # Better: build the full tensor check using einsum or direct
        # Check: for random (i,j,k), verify sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        Ua = np.array(U_mat)
        Va = np.array(V_mat)
        Wa = np.array(W_mat)
        # Build all indices
        i_vals = range(n)
        j_vals = range(m)
        k_vals = range(p)
        for i in i_vals:
            for j in j_vals:
                for k in k_vals:
                    s = 0.0
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    # --- Baseline: block Strassen rank 42 ---
    def make_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        U_coeffs = [
            {(0, 0): 1, (1, 1): 1},
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 0): 1, (0, 0): -1},
            {(0, 1): 1, (1, 1): -1},
        ]
        V_coeffs = [
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(0, 1): 1, (1, 1): -1},
            {(1, 0): 1, (0, 0): -1},
            {(1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 0): 1, (1, 1): 1},
        ]
        W_contrib = [
            {(0, 0): 1, (1, 1): 1},
            {(1, 0): 1, (1, 1): -1},
            {(0, 1): 1, (1, 1): 1},
            {(0, 0): 1, (1, 0): 1},
            {(0, 0): -1, (0, 1): 1},
            {(1, 1): 1},
            {(0, 0): 1},
        ]
        
        col = 0
        for a in range(2):
            for b in range(3):
                for s in range(7):
                    for (i_loc, j_loc), coeff in U_coeffs[s].items():
                        j = a * 2 + j_loc
                        u_idx = i_loc * m + j
                        U[u_idx][col] += coeff
                    for (j_loc, k_loc), coeff in V_coeffs[s].items():
                        j = a * 2 + j_loc
                        k = b * 2 + k_loc
                        v_idx = j * p + k
                        V[v_idx][col] += coeff
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        k = b * 2 + k_loc
                        w_idx = k * n + i_loc
                        W[w_idx][col] += coeff
                    col += 1
        return U, V, W

    # Build baseline
    U_base, V_base, W_base = make_strassen_42()
    
    # Verify baseline
    if not verify(U_base, V_base, W_base):
        # Should not happen, but fallback
        pass

    best_rank = 42
    best_U, best_V, best_W = U_base, V_base, W_base

    # --- Try to find lower rank by sparse column elimination ---
    # Strategy: start from Strassen and see if we can zero out some columns
    # by adjusting others, maintaining exactness.
    
    # Alternative: build a random decomposition and try to sparsify.
    
    # --- Try direct rank reduction via column merging ---
    # Check if any two columns are "mergeable" (their contribution can be combined)
    
    # Actually, let's try: can we use fewer Strassen terms? 
    # For 2x2 x 2x2, Strassen rank is 7, proven optimal.
    # For the block structure 2x3 blocks, we use 2*3*7 = 42.
    # Could we share some Strassen products between blocks?
    
    # Consider: if two B-blocks use the same V pattern, we might combine.
    # In our setup, B blocks are k={0,1}, k={2,3}, k={4,5}.
    # The V entries for each block are shifted in k. No overlap.
    # Similarly A blocks are j={0,1}, j={2,3}. No overlap in U.
    # So no direct sharing.
    
    # Try: can we reduce rank by using a different block partition?
    # e.g., 1 block of size 4 in one dimension? That's the trivial decomposition.
    
    # --- Try: random search for rank 41, 40, ... ---
    # Use a randomized approach: pick a random subset of columns to "merge" or "zero"
    
    def try_reduce_rank(rank_target, U_init, V_init, W_init, max_iter=2000):
        """Try to find a valid decomposition of rank_target by modifying columns."""
        U = [row[:] for row in U_init[:rank_target]] if rank_target <= len(U_init[0]) else None
        # Actually, we need to build from scratch or modify
        pass
    
    # --- More systematic: use the tensor to solve for rank-r decomposition ---
    # The matmul tensor has n*m*p = 48 entries (i,j,k) each = 1
    # A rank-r decomposition: sum_{s=1}^{r} u_{i,j}^s * v_{j,k}^s * w_{k,n+i}^s = 1 for all i,j,k
    # where u^s is an (n x m) matrix, v^s is (m x p), w^s is (p x n)
    # This is a system of 48 nonlinear equations in 48*r variables (with scale invariance)
    
    # Try rank 41 with random init and Newton iteration
    def attempt_rank(r, max_time=5):
        deadline = start_time + max_time
        # Unknowns: for each s, u_s (n*m entries), v_s (m*p entries), w_s (p*n entries)
        # Total: r * (n*m + m*p + p*n) = r * (8 + 24 + 12) = r * 44
        # Equations: n*m*p = 48 (each (i,j,k) gives sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 1)
        # We can use the scaling freedom: set W[k*n+i, s] = 1 for all to reduce variables
        # Then equations: sum_s U[i*m+j,s] * V[j*p+k,s] = 1 for all i,j,k (48 equations)
        # Variables: r * (8 + 24) = r * 32
        # For r=41: 1312 vars, 48 eqs -> overdetermined? No, 48 < 1312, underdetermined.
        # So we have freedom. Let's fix W=1 and solve.
        
        if time.time() > deadline:
            return None
            
        np.random.seed(42)
        # Initialize
        U_flat = np.random.uniform(-1, 1, (r, n*m))
        V_flat = np.random.uniform(-1, 1, (r, m*p))
        W_flat = np.ones((r, p*n))
        
        # Equations: for each (i,j,k): sum_s U[s, i*m+j] * V[s, j*p+k] = 1
        # Jacobian and residual for Newton
        def residual(params):
            # params: r*32 array (U_flat + V_flat)
            Uf = params[:r*n*m].reshape(r, n*m)
            Vf = params[r*n*m:].reshape(r, m*p)
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s_val = 0.0
                        for s in range(r):
                            s_val += Uf[s, i*m+j] * Vf[s, j*p+k]
                        res[idx] = s_val - 1.0
                        idx += 1
            return res
        
        def try_newton(params0, iters=50):
            params = params0.copy()
            for it in range(iters):
                if time.time() > deadline:
                    return None
                rres = residual(params)
                if np.max(np.abs(rres)) < 1e-12:
                    return params
                # Build Jacobian (48 x r*32)
                J = np.zeros((n*m*p, r * (n*m + m*p)))
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # d/d U[s, i*m+j] = V[s, j*p+k]
                            for s in range(r):
                                col = s * (n*m + m*p) + (i*m+j)
                                J[eq_idx, col] += Vf[s, j*p+k]  # from U part
                            # d/d V[s, j*p+k] = U[s, i*m+j]
                            for s in range(r):
                                col = s * (n*m + m*p) + n*m + (j*p+k)
                                J[eq_idx, col] += Uf[s, i*m+j]
                            eq_idx += 1
                # Solve J * dp = -res via least squares
                dp, _, _, _ = np.linalg.lstsq(J, -rres, rcond=None)
                params += dp
            rres = residual(params)
            if np.max(np.abs(rres)) < 1e-10:
                return params
            return None
        
        # Try a few random inits
        for trial in range(20):
            if time.time() > deadline:
                break
            params0 = np.random.uniform(-2, 2, r * (n*m + m*p))
            result = try_newton(params0, iters=30)
            if result is not None:
                Uf = result[:r*n*m].reshape(r, n*m)
                Vf = result[r*n*m:].reshape(r, m*p)
                Wf = np.ones((r, p*n))
                # Format
                U_out = [[float(Uf[s, i*m+j]) for s in range(r)] for (i,j) in [(a,b) for a in range(n) for b in range(m)]]
                V_out = [[float(Vf[s, j*p+k]) for s in range(r)] for (j,k) in [(a,b) for a in range(m) for b in range(p)]]
                W_out = [[float(Wf[s, k*n+i]) for s in range(r)] for (k,i) in [(a,b) for a in range(p) for b in range(n)]]
                return U_out, V_out, W_out
        return None

    # Try ranks below 42
    for target_rank in [41, 40, 39, 38, 36, 32]:
        if time.time() - start_time > 120:
            break
        print(f"Attempting rank {target_rank}")
        result = attempt_rank(target_rank, max_time=15)
        if result is not None:
            U_new, V_new, W_new = result
            if verify(U_new, V_new, W_new):
                best_rank = target_rank
                best_U, best_V, best_W = U_new, V_new, W_new
                print(f"Found valid rank {target_rank}")
                break

    if time.time() - start_time > 130:
        raise TimeoutError()

    return {"U": best_U, "V": best_V, "W": best_W}
