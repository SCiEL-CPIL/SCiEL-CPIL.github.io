"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    """
    Solves the QAP instance using Birkhoff Polytope Continuous Relaxation
    followed by discrete 2-opt local search.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Ensure arrays are contiguous for performance
    a_c = np.ascontiguousarray(a, dtype=np.float64)
    b_c = np.ascontiguousarray(b, dtype=np.float64)
    
    # --- Phase 1: Continuous Relaxation (Birkhoff Polytope) ---
    # Objective: Minimize Tr(X^T A X B) where X is doubly stochastic.
    # Gradient: 2 * A X B
    # We use Projected Gradient Descent with Sinkhorn-Knopp projection.
    
    # Initialize with a random doubly stochastic matrix or identity-like
    # A good init is the normalized spectral projection, but random uniform is robust.
    # Let's start with a slightly perturbed identity to break symmetry if needed, 
    # or just uniform. Uniform is safe but slow to converge. 
    # Let's use the spectral init from the previous strategy as a starting point for X?
    # Actually, let's just use a random permutation matrix rounded from spectral, 
    # then relax it. Or simply start with J/n (uniform).
    # J/n is the center of the polytope. It often takes many steps to move away.
    # Better: Start with a random permutation matrix P0, then set X = P0.
    # But P0 is a vertex. Gradient descent from a vertex is tricky.
    # Let's use a "soft" initialization: X = exp(M) / sum(exp(M)) row/col normalized.
    
    # Let's try a spectral initialization for the continuous variable too.
    # Top eigenvectors of A and B.
    # However, computing eigh for n=150 is fast (<0.1s).
    
    try:
        # Spectral components
        # We want to align the "directions" of flow and distance.
        # Let's compute the top 5 eigenvectors.
        k = 5
        w_a, v_a = np.linalg.eigh(a_c)
        idx_a = np.argsort(w_a)[::-1]
        v_a_top = v_a[:, idx_a[:k]]
        
        w_b, v_b = np.linalg.eigh(b_c)
        idx_b = np.argsort(w_b)[::-1]
        v_b_top = v_b[:, idx_b[:k]]
        
        # Construct an initial similarity matrix S
        # S[i,j] = dot(v_a[i], v_b[j])
        S = v_a_top @ v_b_top.T
        
        # Initialize X from S using softmax-like scaling to make it doubly stochastic
        # This is a soft assignment.
        # X = exp(S / tau)
        # Then normalize rows and columns iteratively (Sinkhorn)
        
        tau = 1.0
        X = np.exp(S / tau)
        
        # Sinkhorn-Knopp normalization to make X doubly stochastic
        # Iterate until convergence or max iters
        for _ in range(50):
            # Normalize rows
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-9
            X /= row_sums
            
            # Normalize columns
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1e-9
            X /= col_sums
            
            # Check convergence
            # if np.max(np.abs(X.sum(axis=1) - 1/n)) < 1e-6:
            #     break
            
    except Exception:
        # Fallback to uniform
        X = np.ones((n, n), dtype=np.float64) / n
        for _ in range(10):
            X /= X.sum(axis=1, keepdims=True)
            X /= X.sum(axis=0, keepdims=True)
    
    # Gradient Descent Loop
    # Cost(X) = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l] = Tr(X^T A X B)
    # Grad(X) = 2 * A @ X @ B (symmetric part handling is automatic if A,B sym)
    
    # We will perform gradient steps with projection back to Birkhoff polytope.
    # Step size needs to be tuned.
    # Let's use a fixed step size or adaptive.
    # Given the scale of QAP costs (millions), gradients can be large.
    
    # Normalize A and B to have unit Frobenius norm to stabilize step size?
    # No, we need the actual cost. But for the *search* dynamics, scaling helps.
    # Let's just use a small step size.
    
    # Pre-compute A @ B? No, the gradient is A @ X @ B.
    
    step_size = 0.01
    max_gd_iters = 2000
    
    # Track best discrete solution found during rounding?
    # No, we round at the end or periodically.
    
    current_cost_float = np.trace(X.T @ (a_c @ X @ b_c))
    
    for it in range(max_gd_iters):
        if time.time() > deadline - 20.0: # Leave time for local search
            break
            
        # Compute Gradient
        # G = 2 * A @ X @ B
        # To save memory/time, we can do:
        # temp = A @ X
        # G = 2 * temp @ B
        
        temp = a_c @ X
        G = 2.0 * (temp @ b_c)
        
        # Gradient Descent Step
        X_new = X - step_size * G
        
        # Project to Birkhoff Polytope (Doubly Stochastic)
        # Using Sinkhorn-Knopp
        # We need to handle negative values in X_new. 
        # Standard Sinkhorn works on positive matrices.
        # If X_new has negatives, we can clip them to a small epsilon, 
        # but this breaks the projection property slightly. 
        # A better projection is the Euclidean projection onto Birkhoff polytope, 
        # which is expensive (Hungarian). 
        # For n=150, Hungarian is O(n^3) ~ 3M ops, which is fast (~1ms).
        # Let's use Hungarian projection for accuracy if we have time, 
        # or Sinkhorn with clipping for speed.
        
        # Let's try Sinkhorn with clipping for speed.
        # Clip negatives to 0
        X_new = np.maximum(X_new, 0)
        
        # If all zeros, reset to uniform
        if X_new.sum() < 1e-6:
            X_new = np.ones((n, n)) / n
        else:
            # Normalize
            X_new /= X_new.sum()
            
            # Sinkhorn iterations
            for _ in range(20):
                row_sums = X_new.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1e-9
                X_new /= row_sums
                
                col_sums = X_new.sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1e-9
                X_new /= col_sums
        
        X = X_new
        
        # Optional: Periodically round to discrete and check cost
        # This allows us to keep a "best discrete" candidate even if the continuous path is noisy.
        if it % 10 == 0:
            # Round X to a permutation
            # Use Hungarian on -X (maximize similarity)
            try:
                row_ind, col_ind = linear_sum_assignment(-X)
                p_rounded = np.empty(n, dtype=np.int64)
                p_rounded[row_ind] = col_ind
                
                # Calculate cost
                cost_rounded = np.sum(a_c * b_c[p_rounded][:, p_rounded])
                # We don't update best here yet, we'll do a dedicated local search later.
                # But we can store the best rounded solution found so far.
                # Let's keep a list or just the best one.
                if not hasattr(solve, '_best_p_cont'):
                    solve._best_p_cont = p_rounded
                    solve._best_cost_cont = cost_rounded
                elif cost_rounded < solve._best_cost_cont:
                    solve._best_p_cont = p_rounded
                    solve._best_cost_cont = cost_rounded
            except Exception:
                pass
        
        # Adaptive step size?
        # If cost increases, reduce step size?
        # cost_new = np.trace(X.T @ (a_c @ X @ b_c))
        # if cost_new > current_cost_float:
        #     step_size *= 0.5
        #     X = X + step_size * G # revert
        # else:
        #     step_size *= 1.1
        #     current_cost_float = cost_new
        
    # Clean up temporary attribute
    if hasattr(solve, '_best_p_cont'):
        p_init = solve._best_p_cont
        del solve._best_p_cont
        del solve._best_cost_cont
    else:
        # Fallback: Round the final X
        try:
            row_ind, col_ind = linear_sum_assignment(-X)
            p_init = np.empty(n, dtype=np.int64)
            p_init[row_ind] = col_ind
        except Exception:
            p_init = np.random.permutation(n)
            
    # --- Phase 2: Discrete Local Search (2-Opt) ---
    # Refine the discrete solution obtained from the continuous relaxation.
    
    # Calculate initial cost
    def calc_cost(p):
        return int(np.sum(a_c * b_c[p][:, p]))
    
    current_cost = calc_cost(p_init)
    best_cost = current_cost
    best_p = p_init.copy()
    
    # Pre-compute for delta calculation
    # Delta for swapping i, j:
    # d = sum_{v != i, j} (a[i,v] - a[j,v]) * (b[p[j], p[v]] - b[p[i], p[v]])
    
    a_c_int = a_c.astype(np.int64)
    b_c_int = b_c.astype(np.int64)
    
    def delta_swap(p, i, j):
        x = p[i]
        y = p[j]
        
        # diff_a = a[i, :] - a[j, :]
        # diff_b = b[y, p] - b[x, p]
        
        # Vectorized dot product
        # We need to exclude v=i and v=j from the sum.
        # Sum over all v: dot(a[i,:]-a[j,:], b[y,p]-b[x,p])
        # Subtract terms for v=i and v=j
        
        da = a_c_int[i] - a_c_int[j]
        db = b_c_int[y, p] - b_c_int[x, p]
        
        total = np.dot(da, db)
        
        # Term for v=i:
        # (a[i,i] - a[j,i]) * (b[y, p[i]] - b[x, p[i]])
        # p[i] = x
        term_i = (a_c_int[i, i] - a_c_int[j, i]) * (b_c_int[y, x] - b_c_int[x, x])
        
        # Term for v=j:
        # (a[i,j] - a[j,j]) * (b[y, p[j]] - b[x, p[j]])
        # p[j] = y
        term_j = (a_c_int[i, j] - a_c_int[j, j]) * (b_c_int[y, y] - b_c_int[x, y])
        
        return total - term_i - term_j

    rng = np.random.default_rng(42)
    no_improve_count = 0
    max_no_improve = 2000
    
    # Main Local Search Loop
    while time.time() < deadline - 2.0:
        improved = False
        
        # Try random swaps
        # To be efficient, we try a batch of swaps.
        for _ in range(200):
            i, j = rng.choice(n, 2, replace=False)
            
            d = delta_swap(p_init, i, j)
            
            if d < 0:
                # Accept
                p_init[i], p_init[j] = p_init[j], p_init[i]
                current_cost += d
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p_init.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            if time.time() > deadline - 2.0:
                break
                
        if not improved:
            no_improve_count += 200
            
        # Kick if stuck
        if no_improve_count > max_no_improve:
            # Random 5-swap kick
            indices = rng.choice(n, size=5, replace=False)
            for k in range(4):
                i, j = indices[k], indices[k+1]
                p_init[i], p_init[j] = p_init[j], p_init[i]
            
            current_cost = calc_cost(p_init)
            no_improve_count = 0
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
