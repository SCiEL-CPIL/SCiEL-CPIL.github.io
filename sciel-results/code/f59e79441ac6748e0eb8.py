import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 15
    N_W = p * n  # 10
    
    # Precompute indices for the tensor contraction
    # The tensor T[i, j, k] = 1 for all i, j, k
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
                
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def get_structured_init(R, seed):
        """
        Create structured initialization using block decomposition ideas.
        Partition U into row blocks (n rows of m), V into column blocks (m rows of p),
        and use bilinear structure to create better initial guesses.
        """
        np.random.seed(seed)
        
        # For rank R, we can think of R as a combination of blocks
        # Use a mix of dense random and structured patterns
        
        # Base: random initialization
        U_init = np.random.randn(N_U, R) * 0.5
        V_init = np.random.randn(N_V, R) * 0.5
        W_init = np.random.randn(N_W, R) * 0.5
        
        # Add structured components based on block decomposition
        # For each (i,j) pair in U, add contributions from different k blocks
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Assign some columns to handle this (i,j,k) interaction
                    col = (i * m + j) * p + k
                    if col < R:
                        # Add a structured component
                        U_init[i * m + j, col] += 1.0
                        V_init[j * p + k, col] += 1.0
                        W_init[k * n + i, col] += 1.0
        
        # Normalize to reasonable scale
        U_init = U_init / (np.linalg.norm(U_init) + 1e-10) * np.sqrt(R / 3)
        V_init = V_init / (np.linalg.norm(V_init) + 1e-10) * np.sqrt(R / 3)
        W_init = W_init / (np.linalg.norm(W_init) + 1e-10) * np.sqrt(R / 3)
        
        return U_init, V_init, W_init

    def loss_and_grad(x, R):
        """
        Compute loss and gradient for the tensor decomposition.
        Uses efficient computation with precomputed indices.
        """
        U = x[:N_U * R].reshape(N_U, R)
        V = x[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
        W = x[N_U * R + N_V * R :].reshape(N_W, R)
        
        # Compute the contraction: sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        terms = U[u_indices] * V[v_indices] * W[w_indices]
        sums = terms.sum(axis=1)
        
        # Loss
        errors = sums - 1.0
        loss = np.sum(errors ** 2)
        
        # Gradient computation
        # d/dU[i*m+j, r] = 2 * sum over (j',k') where u_idx points to i*m+j of 
        #   (sums[(j',k')] - 1) * V[v_idx, r] * W[w_idx, r]
        
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        # For each term in the sum
        for idx in range(len(u_indices)):
            u_idx = u_indices[idx]
            v_idx = v_indices[idx]
            w_idx = w_indices[idx]
            err = errors[idx]
            
            # Accumulate gradients for all r
            U_grad[u_idx] += 2 * err * V[v_idx] * W[w_idx]
            V_grad[v_idx] += 2 * err * U[u_idx] * W[w_idx]
            W_grad[w_idx] += 2 * err * U[u_idx] * V[v_idx]
        
        grad = np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
        return loss, grad

    def try_snap_and_verify(U_opt, V_opt, W_opt, R):
        """
        Try to snap to half-integers and verify exact decomposition.
        """
        # Try different snapping strategies
        
        # Strategy 1: Snap to nearest 0.5
        U_snap = np.round(U_opt * 2) / 2
        V_snap = np.round(V_opt * 2) / 2
        W_snap = np.round(W_opt * 2) / 2
        
        terms_snap = U_snap[u_indices] * V_snap[v_indices] * W_snap[w_indices]
        sums_snap = terms_snap.sum(axis=1)
        
        if np.allclose(sums_snap, 1.0, atol=1e-9):
            return U_snap, V_snap, W_snap
        
        # Strategy 2: Snap to nearest integer
        U_snap = np.round(U_opt)
        V_snap = np.round(V_opt)
        W_snap = np.round(W_opt)
        
        terms_snap = U_snap[u_indices] * V_snap[v_indices] * W_snap[w_indices]
        sums_snap = terms_snap.sum(axis=1)
        
        if np.allclose(sums_snap, 1.0, atol=1e-9):
            return U_snap, V_snap, W_snap
        
        # Strategy 3: Try to find exact solution near the optimized values
        # Use the optimized values as a starting point for a more careful search
        
        return None

    def polish_solution(U_opt, V_opt, W_opt, R, time_budget=5.0):
        """
        High-precision polish pass to find exact half-integer solution.
        """
        start = time.time()
        
        # Try local search around the optimized solution
        U_curr = U_opt.copy()
        V_curr = V_opt.copy()
        W_curr = W_opt.copy()
        
        best_loss = np.inf
        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
        
        # Try small perturbations in half-integer grid
        step_sizes = [0.5, 0.25, 0.125, 0.0625]
        
        for step in step_sizes:
            if time.time() - start > time_budget:
                break
            
            # Try moving each element by +-step
            for _ in range(100):  # Limited iterations
                if time.time() - start > time_budget:
                    break
                
                # Randomly select an element to perturb
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    row, col = np.random.randint(N_U), np.random.randint(R)
                    for delta in [-step, step]:
                        U_test = U_curr.copy()
                        U_test[row, col] += delta
                        terms = U_test[u_indices] * V_curr[v_indices] * W_curr[w_indices]
                        sums = terms.sum(axis=1)
                        loss = np.sum((sums - 1.0) ** 2)
                        if loss < best_loss:
                            best_loss = loss
                            best_U = U_test.copy()
                            if loss < 1e-10:
                                return best_U, V_curr, W_curr
                elif mat_idx == 1:
                    row, col = np.random.randint(N_V), np.random.randint(R)
                    for delta in [-step, step]:
                        V_test = V_curr.copy()
                        V_test[row, col] += delta
                        terms = U_curr[u_indices] * V_test[v_indices] * W_curr[w_indices]
                        sums = terms.sum(axis=1)
                        loss = np.sum((sums - 1.0) ** 2)
                        if loss < best_loss:
                            best_loss = loss
                            best_V = V_test.copy()
                            if loss < 1e-10:
                                return U_curr, best_V, W_curr
                else:
                    row, col = np.random.randint(N_W), np.random.randint(R)
                    for delta in [-step, step]:
                        W_test = W_curr.copy()
                        W_test[row, col] += delta
                        terms = U_curr[u_indices] * V_curr[v_indices] * W_test[w_indices]
                        sums = terms.sum(axis=1)
                        loss = np.sum((sums - 1.0) ** 2)
                        if loss < best_loss:
                            best_loss = loss
                            best_W = W_test.copy()
                            if loss < 1e-10:
                                return U_curr, V_curr, best_W
        
        # Check if we found an exact solution
        if best_loss < 1e-10:
            return best_U, best_V, best_W
        
        return None

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_solution = get_schoolbook()
    
    # Search for minimal rank
    # Lower bound is max(N_U, N_V, N_W) = 15
    # Upper bound is schoolbook rank 30
    for R in range(15, n * m * p):
        if time.time() > deadline:
            break
            
        # Number of restarts depends on remaining time and rank
        remaining_time = deadline - time.time()
        ranks_left = (n * m * p) - R
        time_per_rank = remaining_time / (ranks_left + 1)
        
        # Estimate restarts based on time_per_rank
        max_restarts = int(time_per_rank / 0.3)
        max_restarts = max(1, min(30, max_restarts))
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Use structured initialization
            seed = 42 + R * 1000 + restart
            U_init, V_init, W_init = get_structured_init(R, seed)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def loss_wrapper(x):
                loss, _ = loss_and_grad(x, R)
                return loss
            
            def grad_wrapper(x):
                _, grad = loss_and_grad(x, R)
                return grad
            
            try:
                # Use L-BFGS-B with gradient for better convergence
                res = minimize(loss_wrapper, x0, jac=grad_wrapper, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-3:
                # Extract optimized values
                x_opt = res.x
                U_opt = x_opt[:N_U * R].reshape(N_U, R)
                V_opt = x_opt[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                W_opt = x_opt[N_U * R + N_V * R :].reshape(N_W, R)
                
                # Try to snap and verify
                snapped = try_snap_and_verify(U_opt, V_opt, W_opt, R)
                if snapped is not None:
                    U_snap, V_snap, W_snap = snapped
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                
                # Try polish if we have time
                if time.time() < deadline - 10:
                    polished = polish_solution(U_opt, V_opt, W_opt, R, time_budget=5.0)
                    if polished is not None:
                        U_pol, V_pol, W_pol = polished
                        # Verify
                        terms = U_pol[u_indices] * V_pol[v_indices] * W_pol[w_indices]
                        sums = terms.sum(axis=1)
                        if np.allclose(sums, 1.0, atol=1e-9):
                            # Snap to half-integers
                            U_final = np.round(U_pol * 2) / 2
                            V_final = np.round(V_pol * 2) / 2
                            W_final = np.round(W_pol * 2) / 2
                            terms_final = U_final[u_indices] * V_final[v_indices] * W_final[w_indices]
                            sums_final = terms_final.sum(axis=1)
                            if np.allclose(sums_final, 1.0, atol=1e-9):
                                return {
                                    "U": U_final.tolist(),
                                    "V": V_final.tolist(),
                                    "W": W_final.tolist()
                                }
        
    return best_solution
