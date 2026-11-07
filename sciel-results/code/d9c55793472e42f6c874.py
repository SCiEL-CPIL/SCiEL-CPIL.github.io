"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """DF-BFGS with periodic exploration restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.clip(x, -5, 5).copy()
            return val

        def check_deadline():
            return (time.time() - start_time) > deadline

        # DF-BFGS implementation details
        # We use a two-point finite difference for gradients to save budget
        # and maintain a limited memory BFGS Hessian approximation.
        
        # Parameters
        n_mem = 5  # Limited memory size
        eps_fd = 1e-8  # Finite difference step
        
        # Initialize at origin (often good for BBOB instances with shift in [-4,4])
        # But Weierstrass is multi-modal. We will do a few restarts.
        
        # Strategy:
        # 1. Run DF-BFGS from origin.
        # 2. If stagnation or budget allows, restart from best point found + noise.
        
        def run_df_bfgs(x0, max_calls):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or check_deadline():
                return

            x = x0.copy()
            f = call_func(x)
            
            # Initialize BFGS state
            # We approximate Hessian H with H_k = V_k D_k V_k^T
            # Storing s_k (steps) and y_k (gradient diffs)
            S = [] # list of step vectors
            Y = [] # list of gradient diff vectors
            rho = [] # list of 1/(y^T s)
            H0 = 1.0 / (np.linalg.norm(x) + 1.0) # Initial Hessian scale
            if np.linalg.norm(x) == 0:
                H0 = 1.0

            grad = None
            last_f = f
            last_x = x.copy()
            
            # Main loop
            while calls_made < budget - max_calls and not check_deadline():
                # 1. Compute gradient at current x
                # Use forward difference
                g = np.zeros(dim)
                for i in range(dim):
                    x_pert = x.copy()
                    x_pert[i] += eps_fd
                    f_pert = call_func(x_pert)
                    g[i] = (f_pert - last_f) / eps_fd
                    if calls_made >= budget - max_calls or check_deadline():
                        break
                
                if calls_made >= budget - max_calls or check_deadline():
                    break
                
                # Update best if improved (gradient evals don't update best_x unless we check, 
                # but call_func does it)
                
                # 2. Compute search direction p = -H * g
                # Apply BFGS update formula for p
                # p = -H_k g_k
                # H_k g = H0 g + (I - H0 * g^T y / (y^T s)) ... this is the recursive form
                
                # Simpler: Use the standard L-BFGS two-loop recursion
                # We want to solve H g = q, then p = -q
                q = g.copy()
                alphas = [0.0] * len(S)
                
                # Backward loop
                for i in range(len(S) - 1, -1, -1):
                    if len(S) > i:
                        s_i = S[i]
                        y_i = Y[i]
                        rho_i = rho[i]
                        alphas[i] = rho_i * np.dot(s_i, q)
                        q -= alphas[i] * y_i
                
                # Apply H0
                q = H0 * q
                
                # Forward loop
                for i in range(len(S)):
                    s_i = S[i]
                    y_i = Y[i]
                    rho_i = rho[i]
                    beta = rho_i * np.dot(y_i, q)
                    q += (alphas[i] - beta) * s_i
                
                p = -q
                
                # 3. Line search (Backtracking)
                alpha = 1.0
                c1 = 1e-4
                max_backtracks = 10
                found_alpha = False
                f_new = f
                x_new = x.copy()
                
                for _ in range(max_backtracks):
                    x_trial = x + alpha * p
                    # Clip to box
                    x_trial = np.clip(x_trial, -5, 5)
                    
                    f_trial = call_func(x_trial)
                    if calls_made >= budget - max_calls or check_deadline():
                        break
                    
                    # Armijo condition
                    if f_trial <= last_f + c1 * alpha * np.dot(p, g):
                        found_alpha = True
                        x_new = x_trial
                        f_new = f_trial
                        break
                    alpha *= 0.5
                
                if calls_made >= budget - max_calls or check_deadline():
                    break

                if not found_alpha:
                    # Try a smaller step or break
                    # If line search fails, break to restart
                    break

                # 4. Update BFGS memory
                s_new = x_new - x
                # Need gradient at x_new
                g_new = np.zeros(dim)
                for i in range(dim):
                    x_pert = x_new.copy()
                    x_pert[i] += eps_fd
                    f_pert = call_func(x_pert)
                    g_new[i] = (f_pert - f_new) / eps_fd
                    if calls_made >= budget - max_calls or check_deadline():
                        break
                
                if calls_made >= budget - max_calls or check_deadline():
                    break

                y_new = g_new - g
                yTs = np.dot(y_new, s_new)
                
                if yTs > 1e-8: # Ensure positive curvature
                    S.append(s_new)
                    Y.append(y_new)
                    rho.append(1.0 / yTs)
                    
                    # Limit memory
                    if len(S) > n_mem:
                        S.pop(0)
                        Y.pop(0)
                        rho.pop(0)
                else:
                    # Reset memory if no improvement in gradient direction
                    S = []
                    Y = []
                    rho = []
                    H0 = 1.0 # Reset scale

                x = x_new
                last_f = f_new
                last_x = x_new.copy()
                
                # Stagnation check: if step is very small, break
                if np.linalg.norm(s_new) < 1e-6:
                    break

        # Execution Plan
        # Phase 1: Explore with a few random restarts using DF-BFGS
        # Weierstrass has many local minima.
        
        # 1. Start at 0
        run_df_bfgs(np.zeros(dim), 50)
        
        # 2. If we have budget left, try a few more restarts from best found + noise
        # Or just keep polishing best if it's good enough?
        # Given the multi-modality, restarting from best + noise is risky.
        # Restarting from random points is better for exploration.
        
        # Let's do 3 more restarts from random points if budget allows
        for i in range(3):
            if calls_made >= budget or check_deadline():
                break
            x_rand = np.random.uniform(-4, 4, dim)
            run_df_bfgs(x_rand, 50)

        return best_x, best_f

# EVOLVE-BLOCK-END
