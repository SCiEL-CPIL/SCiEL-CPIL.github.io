"""Derivative-Free Trust-Region (DFTR) optimizer for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Derivative-Free Trust-Region optimizer.
    
    Uses a 2-point Broyden-like update to approximate the Hessian and gradient.
    This allows for efficient navigation of ill-conditioned landscapes by 
    explicitly modeling local curvature rather than relying on population statistics.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        # Safety margin: 280s to ensure we finish well before the 300s/330s kill limit
        deadline = start_time + 280.0

        best_x = None
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Initialization ---
        # Start at the center of the box
        x = np.zeros(dim)
        f_x = safe_call(x)
        
        # Initialize Broyden matrix (Hessian approximation) and Gradient
        # Start with identity, will be updated
        B = np.eye(dim)
        g = np.zeros(dim)
        
        # Initial trust region radius
        # Must be large enough to find the basin, but not so large that it wastes calls
        # Box is [-5, 5]^5. Min is in [-4, 4]^5.
        # A radius of 2.0 covers a good chunk of the space.
        rho = 2.0 
        rho_min = 1e-6
        rho_max = 4.0
        
        # Pre-compute basis vectors for 2-point derivative estimation
        # We use the standard coordinate directions for simplicity and robustness
        # in this implementation, though a randomized basis could be used.
        # For dim=5, 2-point per direction = 10 calls per Broyden update.
        
        # Store previous x and f for Broyden update
        x_prev = x.copy()
        f_prev = f_x
        
        # We need to estimate the gradient and Hessian at the start.
        # Let's do a forward difference estimation for initial g and B.
        # Actually, we can just run the main loop. The first step will be a "try" step.
        # But Broyden needs a consistent state.
        
        # Let's initialize g and B using a simple finite difference scheme 
        # to get a better starting point for the first trust region step.
        # This costs 2*dim calls.
        h_init = 0.1
        for i in range(dim):
            x_plus = x.copy()
            x_plus[i] += h_init
            f_plus = safe_call(x_plus)
            
            x_minus = x.copy()
            x_minus[i] -= h_init
            f_minus = safe_call(x_minus)
            
            g[i] = (f_plus - f_minus) / (2 * h_init)
            # For Hessian, we use the standard 2-point second difference or 
            # just rely on the Broyden update from the first step.
            # To save budget, we might skip full Hessian init and let Broyden handle it.
            # However, a poor initial B can lead to bad steps.
            # Let's stick with B=I and g=0 initially and let the first few steps
            # be "exploratory" or use the first successful step to update.
            
            # Actually, a better strategy for DFTR is to use the first step 
            # as a pure search step if the model is unknown.
            # Let's reset g and B to zero/identity and let the algorithm build them.
            # But Broyden requires a "previous" state.
            # We will treat the initial state as having x=0, f=0, g=0, B=I.
            # The first step will be determined by min ||B s||^2 + 2 g^T s s.t. ||s|| <= rho.
            # With g=0, B=I, this minimizes ||s||^2, which is 0. That's bad.
            
            # So we MUST initialize g and B reasonably.
            # Let's use the finite differences calculated above for g.
            # For B, we can use a simple finite difference Hessian approximation 
            # or just set B = lambda * I where lambda is a guess.
            # Given the ill-conditioning, B=I is a weak guess.
            # Let's use the "forward difference" Hessian approximation for the diagonal 
            # and zero off-diagonals initially, or just keep B=I and rely on the 
            # fact that the first step might be small.
            
            # Alternative: Use a "pattern search" like step for the first few iterations?
            # No, let's stick to DFTR.
            
            # Let's re-evaluate: If g is estimated, and B=I, the step is -rho * g/||g||.
            # This is a steepest descent step with fixed step size rho.
            # This is actually a reasonable start!
            
        # So, we have estimated g using 2*dim calls.
        # B is still I.
        # Current x is still 0 (we didn't move x, just evaluated neighbors).
        # Wait, safe_call updates best_x/best_f.
        # x is still the center.
        # f_x is the value at center.
        # g is estimated.
        # B is I.
        
        # Main Loop
        while total_calls < budget and time.time() < deadline:
            # 1. Solve the trust region subproblem:
            #    min ||B s||^2 / 2 + g^T s  s.t. ||s|| <= rho
            
            # Since B is symmetric positive definite (we will enforce this),
            # the unconstrained minimum is s_uncon = -B^{-1} g.
            s_uncon = np.linalg.solve(B, -g)
            
            # Check if it's within the trust region
            norm_s_uncon = np.linalg.norm(s_uncon)
            
            if norm_s_uncon <= rho:
                s = s_uncon
            else:
                # Project onto the boundary of the trust region
                s = rho * s_uncon / norm_s_uncon
            
            # 2. Take the step
            x_new = x + s
            f_new = safe_call(x_new)
            
            # 3. Calculate the actual reduction
            actual_reduction = f_x - f_new
            
            # 4. Calculate the predicted reduction
            #    pred = - (g^T s + 0.5 * s^T B s)
            #    Note: The model is m(s) = f_x + g^T s + 0.5 s^T B s
            #    So m(s) - f_x = g^T s + 0.5 s^T B s
            #    Reduction = f_x - m(s) = - (g^T s + 0.5 s^T B s)
            predicted_reduction = -(g @ s + 0.5 * s @ B @ s)
            
            # 5. Calculate ratio
            if abs(predicted_reduction) < 1e-10:
                ratio = 0.0 if actual_reduction < 0 else 1.0
            else:
                ratio = actual_reduction / predicted_reduction
            
            # 6. Update Trust Region Radius
            if ratio > 0.75:
                # Good step, expand
                rho = min(rho * 2.0, rho_max)
            elif ratio < 0.25:
                # Bad step, shrink
                rho = max(rho * 0.5, rho_min)
            else:
                # Moderate step, keep rho
                pass
            
            # 7. Update B (Broyden Update)
            # We need to update B to reflect the change in gradient.
            # First, we need to estimate the new gradient g_new.
            # This requires 2*dim additional calls.
            # To save budget, we can update B only when we have enough calls left,
            # or use a cheaper update.
            # However, for DFTR to work well, we need a good B.
            
            # Let's check if we have budget for gradient estimation.
            # Cost: 2 * dim = 10 calls.
            # If total_calls + 10 > budget, we might skip the update and just move x.
            # But if we skip, B becomes stale.
            
            # Strategy: Always update if we have enough budget.
            # If we are close to budget, we can stop updating and just do steepest descent?
            # Or just stop.
            
            if total_calls + 2 * dim <= budget and time.time() < deadline - 1.0:
                # Estimate new gradient at x_new
                h_grad = 0.1
                g_new = np.zeros(dim)
                for i in range(dim):
                    x_plus = x_new.copy()
                    x_plus[i] += h_grad
                    f_plus = safe_call(x_plus)
                    
                    x_minus = x_new.copy()
                    x_minus[i] -= h_grad
                    f_minus = safe_call(x_minus)
                    
                    g_new[i] = (f_plus - f_minus) / (2 * h_grad)
                
                # Broyden Update (2-point)
                # s = x_new - x
                # y = g_new - g
                # B_new = B + (y - B s) s^T / (s^T s)
                
                s_diff = x_new - x
                y_diff = g_new - g
                
                # Check for singularity or very small s
                s_norm_sq = s_diff @ s_diff
                if s_norm_sq > 1e-10:
                    B += np.outer(y_diff - B @ s_diff, s_diff) / s_norm_sq
                    
                    # Symmetrize B to maintain positive definiteness properties
                    # Note: Standard Broyden doesn't guarantee symmetry.
                    # For trust region, we often use a symmetric version or just ensure PD.
                    # Let's symmetrize and check eigenvalues.
                    B = (B + B.T) / 2.0
                    
                    # Ensure B is PD by clipping eigenvalues
                    eigvals, eigvecs = np.linalg.eigh(B)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    B = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    
                    g = g_new
                else:
                    # s was too small, just update g
                    g = g_new
            else:
                # Not enough budget for full gradient update.
                # We can still update g using a cheaper method or just keep old g?
                # If we don't update g, the model is wrong.
                # Let's just update x and f, and keep g, B as is for next step.
                # This might lead to poor performance if we are near the end.
                # But it's better than crashing.
                pass
            
            # 8. Accept or Reject
            if ratio > 0.25:
                # Accept the step
                x = x_new
                f_x = f_new
            else:
                # Reject the step (x remains unchanged)
                # But we already spent the calls.
                # The model is still based on the old x.
                # We might want to shrink rho more aggressively if we reject.
                # Already handled above.
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
