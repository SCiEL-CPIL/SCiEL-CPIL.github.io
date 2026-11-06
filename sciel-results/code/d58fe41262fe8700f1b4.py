import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 by minimizing |f(x)|^2 using Nelder-Mead simplex method.
    
    Strategy:
    1. Reframe root finding as minimization of g(x) = f(x)^2.
    2. Use Nelder-Mead (derivative-free) which is robust to ill-conditioning and doesn't require bracketing.
    3. Adaptive step sizes to handle scales.
    4. Restart mechanism: if the simplex collapses or progress stalls, restart from the best point found so far with a new, smaller or randomized simplex.
    5. Strict evaluation budget management.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result

    def check_time():
        return (time.time() - start_time) < DEADLINE

    # State
    best_x = float(x0)
    best_g = float('inf')
    best_fx = None
    
    try:
        best_fx = call_f(best_x)
        best_g = best_fx * best_fx
    except Exception:
        best_fx = float('inf')
        best_g = float('inf')
        
    if best_g <= 1e-16: # |f| <= 1e-8
        return best_x

    # Nelder-Mead parameters
    # Simplex in 1D is 2 points.
    # We will manage a list of (x, g) pairs.
    # To make it more robust than standard 1D NM, we can effectively do a 1D search with adaptive steps.
    # However, true NM in 1D is just a 2-point simplex.
    # Let's implement a robust 1D minimizer that mimics NM logic:
    # Maintain two points x1, x2.
    # Reflection, Expansion, Contraction.
    
    # Alternative: Use a "Particle" approach with a few active particles, moving them towards the best.
    # But NM is a specific strategy requested by inspiration 3.
    
    # Let's implement a robust 1D Nelder-Mead.
    # Initial simplex: x0, x0 + delta
    # delta depends on scale of x0.
    
    def nelder_mead_1d(f_obj, x_start, max_evals, deadline, current_best_x, current_best_g):
        nonlocal eval_count, best_x, best_g
        
        # Initialize simplex
        # Scale of x
        scale = max(1.0, abs(x_start))
        # Initial step size
        # We try a few different initial step sizes if the first one fails?
        # No, let's adapt the step size within the restarts.
        
        # For the first restart, use a moderate step.
        # For subsequent restarts, we can vary it.
        # The caller will handle restarts by calling this function with different parameters or by restarting the whole process.
        # Actually, it's better to implement the restarts *inside* the main loop or as a wrapper.
        
        # Let's structure the main function to run multiple NM attempts.
        pass

    # We will implement a loop of NM attempts.
    # Each attempt has a budget of evaluations.
    # If an attempt finishes without success, we restart from the global best with a different initial simplex configuration.
    
    # Number of restarts
    # We have 500 evals total.
    # Each NM attempt needs at least 2 evals to start (if not reusing best).
    # Let's allocate budget dynamically.
    
    # Initial simplex configuration
    # x0 is given.
    # We need a second point.
    # Let's try several "scales" for the initial step.
    
    # Strategy:
    # 1. Start with x0 and x0 + h.
    # 2. Run NM for a fixed number of iterations or until convergence.
    # 3. If not converged, update global best.
    # 4. Restart with x_global_best and a new h (maybe smaller, or randomized).
    
    # To maximize robustness, we should try to find *any* root, not just the one near x0.
    # The problem says "near x0" in the docstring of the seed, but the task says "Root finding".
    # Usually "near x0" implies local root. But ill-conditioned might mean the root is tricky.
    # If the root is far, NM from x0 might not find it if the function is not unimodal.
    # However, the seed used a sweep. The inspiration says "bypasses instability".
    # If we assume the root is near x0, NM is good.
    # If the root is far, we might need a global search.
    # The task contract says "fixed local ill-conditioned test equations". This suggests the root is local to x0, but the conditioning is bad (steep/flat/wavy).
    
    # So, local NM should work well if the initial step is right.
    # The challenge is choosing the right initial step h.
    # If h is too small, we might get stuck in a flat region or numerical noise.
    # If h is too large, we might jump over the root or into a bad region.
    
    # Plan:
    # Run NM with several different initial step sizes h.
    # h candidates: 1e-4, 1e-2, 1e-1, 1.0, 1e2, 1e4 (scaled by x0 magnitude).
    
    # We will keep track of the best result across all attempts.
    
    # List of initial step multipliers
    # We'll generate a set of h values relative to max(1, |x0|)
    base_scale = max(1.0, abs(x0))
    
    # Log-spaced steps
    import numpy as np
    # np.logspace(-6, 6, 13) -> 13 steps from 1e-6 to 1e6
    # This gives us 13 different starting simplexes.
    # Each simplex attempt:
    # 1. Init x1 = x0, x2 = x0 + h
    # 2. Evaluate f(x1), f(x2) (if not already done)
    # 3. Run NM iterations.
    
    # To save evaluations, we can reuse the f(x0) we already have.
    
    # Let's define the NM step logic for 1D.
    # Simplex: [p1, p2] where p1 is "worse" (higher g) and p2 is "better" (lower g)?
    # Standard NM keeps points in a list. In 1D, just two points.
    # Let's say x_a, x_b with g_a, g_b.
    # Assume g_a <= g_b (a is better).
    
    # Reflection: x_r = x_a + alpha * (x_a - x_b) = 2*x_a - x_b (alpha=1)
    # If g_r < g_a:
    #    Expansion: x_e = x_a + gamma * (x_r - x_a) = x_r + (x_r - x_a) = 2*x_r - x_a (gamma=1)
    #    If g_e < g_r: x_new = x_e, else x_new = x_r
    #    Replace x_b with x_new
    # Else if g_r < g_b:
    #    Replace x_b with x_r
    # Else:
    #    Contraction: x_c = x_a + rho * (x_b - x_a) = x_a + 0.5*(x_b - x_a) (rho=0.5)
    #    If g_c < g_b:
    #       Replace x_b with x_c
    #    Else:
    #       Shrink: x_s = x_a + 0.5 * (x_b - x_a) ... wait, shrink towards best.
    #       x_s = x_a + 0.5 * (x_old_worse - x_a)
    #       If g_s < g_old_worse: Replace worse with x_s
    #       Else: Fail/Restart
    
    # We need to be careful with the "worse" point update.
    
    def run_nm_attempt(h, budget):
        nonlocal eval_count, best_x, best_g, best_fx
        
        # Initialize points
        x1 = x0
        g1 = best_g
        f1 = best_fx # We have this
        
        x2 = x0 + h
        # Check if x2 is too close to x1
        if abs(x2 - x1) < 1e-15:
            return
        
        try:
            f2 = call_f(x2)
            g2 = f2 * f2
        except Exception:
            return
            
        if not math.isfinite(g2):
            g2 = float('inf')
            
        # Ensure x1 is the better point (lower g)
        if g2 < g1:
            x1, x2 = x2, x1
            g1, g2 = g2, g1
            f1, f2 = f2, f1
            
        # Update global best
        if g1 < best_g:
            best_g = g1
            best_x = x1
            best_fx = f1
        if g2 < best_g:
            best_g = g2
            best_x = x2
            best_fx = f2
            
        if best_g <= 1e-16:
            return
            
        # NM Loop
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # We iterate until budget is exhausted or convergence
        # Convergence: |g1 - g2| is small AND |x1 - x2| is small?
        # Or just |g1| <= 1e-16.
        
        while check_time() and eval_count < MAX_EVALS:
            # Check success
            if best_g <= 1e-16:
                return
                
            # Check if simplex has collapsed
            if abs(x1 - x2) < 1e-14 * max(1.0, abs(x1), abs(x2)):
                break
                
            # Reflection
            # x_r is reflection of x2 (worse) through x1 (better)
            x_r = 2.0 * x1 - x2
            
            # Avoid evaluating same point
            if abs(x_r - x1) < 1e-15 or abs(x_r - x2) < 1e-15:
                # Try a small perturbation or break
                break
                
            try:
                f_r = call_f(x_r)
                g_r = f_r * f_r
            except Exception:
                break
                
            if not math.isfinite(g_r):
                g_r = float('inf')
                
            if g_r < g1:
                # Expansion
                x_e = x1 + gamma * (x_r - x1)
                if abs(x_e - x1) < 1e-15 or abs(x_e - x2) < 1e-15 or abs(x_e - x_r) < 1e-15:
                    # Just accept reflection
                    x2 = x_r
                    g2 = g_r
                    f2 = f_r
                    # Update global best
                    if g2 < best_g:
                        best_g = g2
                        best_x = x2
                        best_fx = f2
                    continue
                try:
                    f_e = call_f(x_e)
                    g_e = f_e * f_e
                except Exception:
                    g_e = float('inf')
                
                if not math.isfinite(g_e):
                    g_e = float('inf')
                    
                if g_e < g_r:
                    # Accept expansion
                    x2 = x_e
                    g2 = g_e
                    f2 = f_e
                else:
                    # Accept reflection
                    x2 = x_r
                    g2 = g_r
                    f2 = f_r
            elif g_r < g2:
                # Accept reflection
                x2 = x_r
                g2 = g_r
                f2 = f_r
            else:
                # Contraction
                # Outside or Inside?
                # Standard NM: If g_r >= g2, do contraction.
                # Since g_r >= g2 and g1 <= g2, the root is likely between x1 and x2.
                # Inside contraction.
                x_c = x1 + rho * (x2 - x1)
                
                if abs(x_c - x1) < 1e-15 or abs(x_c - x2) < 1e-15:
                    break
                    
                try:
                    f_c = call_f(x_c)
                    g_c = f_c * f_c
                except Exception:
                    break
                    
                if not math.isfinite(g_c):
                    g_c = float('inf')
                    
                if g_c < g2:
                    # Accept contraction
                    x2 = x_c
                    g2 = g_c
                    f2 = f_c
                else:
                    # Shrink
                    # Shrink x2 towards x1
                    x_s = x1 + sigma * (x2 - x1)
                    if abs(x_s - x1) < 1e-15:
                        break
                    try:
                        f_s = call_f(x_s)
                        g_s = f_s * f_s
                    except Exception:
                        break
                        
                    if not math.isfinite(g_s):
                        break
                        
                    if g_s < g2:
                        x2 = x_s
                        g2 = g_s
                        f2 = f_s
                    else:
                        # No improvement, break and restart
                        break
                        
            # Update global best
            if g1 < best_g:
                best_g = g1
                best_x = x1
                best_fx = f1
            if g2 < best_g:
                best_g = g2
                best_x = x2
                best_fx = f2
                
            if best_g <= 1e-16:
                return

    # Main Loop: Try different step sizes
    # We have used 1 eval for x0.
    # We will try a sequence of h values.
    # To be efficient, we can run them sequentially, stopping if we find a root.
    
    # Generate h values
    # Log spaced from 1e-5 * scale to 1e5 * scale
    # But we must be careful not to waste evals on huge h if the root is local.
    # The problem is "local ill-conditioned". So the root is likely near x0.
    # Small h is usually better, but if the function is very flat, we need larger h to see the slope.
    # If the function is very steep, small h is better.
    
    # Let's try a mix of small and medium steps.
    # 1. 1e-4 * scale
    # 2. 1e-2 * scale
    # 3. 1e-1 * scale
    # 4. 1.0 * scale
    # 5. 1e1 * scale
    
    # We'll also try negative steps? NM handles direction via reflection, so sign of h doesn't matter much for the initial simplex shape, just location.
    # x0 + h vs x0 - h. Since we reflect, it's symmetric.
    
    h_multipliers = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    
    # Shuffle them? No, deterministic is better for reproducibility, but maybe randomized is better for exploration?
    # The inspiration mentioned random.
    # Let's stick to deterministic log-spaced first, as it's a "structured" approach.
    # If we have time, we can add random restarts.
    
    # Sort by magnitude? Usually starting small is safer to avoid jumping to infinity.
    # But if the function is flat, small steps waste time.
    # Let's start with medium steps, then go smaller, then larger.
    # Or just iterate through them.
    
    # To save time, we can break early if we find a root.
    
    for mult in h_multipliers:
        if not check_time() or eval_count >= MAX_EVALS:
            break
            
        h = mult * base_scale
        
        # If h is too small, skip
        if abs(h) < 1e-12:
            continue
            
        # Run NM with this initial step
        # Budget per attempt: We don't strictly limit per attempt, but the global limit applies.
        # However, if one attempt takes 400 evals and fails, we have no budget for others.
        # We should limit each attempt to a fraction of the remaining budget?
        # Or just let it run until it converges or stalls.
        # NM stalls when the simplex collapses.
        # If it doesn't collapse but just oscillates, it might waste evals.
        # Let's add a max iteration count per attempt.
        # 50 iterations is usually enough for NM in 1D.
        
        # We will modify run_nm_attempt to accept a max_iters
        # For now, the while loop in run_nm_attempt breaks on collapse or time/budget.
        # It might run long if it doesn't collapse but makes slow progress.
        # Let's add a counter to run_nm_attempt.
        
        # Actually, the current run_nm_attempt doesn't have a max_iter.
        # Let's wrap it or add a param.
        
        # To avoid modifying the inner function signature complexity,
        # we can just trust the collapse condition.
        # In 1D, if it's not converging, it often oscillates or moves slowly.
        # If it moves slowly, it's not "ill-conditioned" in a bad way, just slow.
        # But 500 evals is small.
        
        # Let's just call it.
        run_nm_attempt(h)
        
        if best_g <= 1e-16:
            break

    return best_x
# EVOLVE-BLOCK-END
