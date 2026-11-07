import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Time limit safety margin: Evaluator kills at 330s. 
        # We stop at 320s to be safe.
        time_limit = 320.0
        
        # State
        self.best_x = np.zeros(n)
        self.best_f = np.inf
        self.calls_made = 0
        
        def call_func(x):
            if self.calls_made >= budget:
                return np.inf
            if time.time() - self.start_time > time_limit:
                return np.inf
            fx = func(x)
            self.calls_made += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # --- Phase 1: Global Exploration with Latin Hypercube ---
        # We want good coverage early for AOCC.
        # LHS ensures spread.
        # Budget allocation: 
        #   ~40% for initial global search (LHS + random)
        #   ~60% for local refinement (Basin Hopping)
        
        # Let's define a helper to generate LHS points
        def latin_hypercube(rng, n_points, n_dims, lower=-4.0, upper=4.0):
            points = np.zeros((n_points, n_dims))
            for i in range(n_dims):
                # Permute indices
                perm = rng.permutation(n_points)
                # Map to [0, 1)
                points[:, i] = (perm + rng.random(n_points)) / n_points
                # Scale to [lower, upper]
                points[:, i] = lower + (upper - lower) * points[:, i]
            return points

        # Initial Global Search
        # We will do a mix of LHS and pure random to be robust.
        # Let's use 100 points for initial phase.
        n_global = 100
        lhs_points = latin_hypercube(rng, n_global, n)
        
        # Shuffle to avoid systematic bias in order? 
        # Actually, for AOCC, we want to find a decent point ASAP.
        # LHS points are spread out. 
        # Let's evaluate them.
        
        # To maximize early progress, we might want to check the center (0,0,0,0,0) first?
        # The problem says minimizer is in [-4,4]. Center is a good guess.
        # Let's insert center at index 0.
        center = np.zeros(n)
        # Check if center is already in LHS (unlikely to be exactly 0)
        # We'll just prepend it.
        
        global_candidates = [center]
        global_candidates.extend(lhs_points)
        
        # Shuffle the rest? 
        # If we shuffle, we might miss the "good" region of the LHS until later.
        # But LHS is designed to be uniform. 
        # Let's keep LHS order but maybe interleave some randoms?
        # Actually, just evaluating LHS is fine. 
        # But wait, if the shift is large, center might be bad. 
        # LHS covers the space.
        
        # Let's evaluate the global candidates.
        for x in global_candidates:
            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                break
            call_func(x)
            
        # --- Phase 2: Multi-Start Basin Hopping ---
        # We have used ~100 calls. Remaining ~400.
        # Strategy:
        # Loop while budget remains:
        #   1. Pick a starting point. 
        #      - Initially, the best point found.
        #      - Later, if local search stagnates, pick a random point from the global set or a new random point.
        #   2. Run Nelder-Mead (or similar local search) from that start.
        #      - Limit function calls per local run to avoid wasting budget on a bad basin.
        #      - If improvement is significant, update best.
        #      - If no improvement after X calls, stop local run and restart.
        
        # We need a wrapper for scipy that respects our global call count and time limit.
        def wrapped_func(x):
            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                return np.inf
            return call_func(x)

        try:
            from scipy.optimize import minimize
            use_scipy = True
        except ImportError:
            use_scipy = False

        # Parameters for basin hopping
        max_local_calls = 50  # Max calls per Nelder-Mead run
        stagnation_limit = 3  # Number of failed restarts before we try a new global random point
        stagnation_count = 0
        
        # We will keep a list of "promising" points (those that led to improvements)
        # Or just always restart from the current best? 
        # Multi-start suggests trying different basins.
        
        # Let's maintain a small population of "elite" points found so far.
        # But for simplicity and robustness, let's do:
        # 1. Try to improve current best.
        # 2. If stuck (no improvement in last local run), generate a new random start point (within [-4,4]) and try to improve that.
        # 3. If that also fails, maybe try another random point.
        
        # Actually, a more robust "Multi-Start" is:
        # - Keep the best point.
        # - Every K local failures, sample a new random point from the box, evaluate it, and if it's better than current best, switch focus to it.
        # - If not better, still try to improve it locally? No, that wastes budget.
        # - Better: If a random point is better, switch. If not, ignore it (cost 1 call) and try to improve current best again?
        
        # Let's refine:
        # While budget > 0:
        #   1. Run local search from current best (or a candidate).
        #   2. If local search improved the global best, reset stagnation counter.
        #   3. If local search failed to improve, increment stagnation counter.
        #   4. If stagnation counter > threshold, pick a new random point x_rand.
        #      Evaluate x_rand.
        #      If x_rand is better than current best, set current best = x_rand, reset stagnation.
        #      Else, keep current best, but maybe try local search from x_rand if it's "close" to best? No, just discard.
        #      Actually, if x_rand is not better, we wasted 1 call. That's fine.
        #      We then try local search from current best again.
        
        # This might loop indefinitely on the same bad basin if local search always fails.
        # To avoid this, we should periodically "jump".
        
        # Let's implement a simpler loop:
        # 1. Local refine current best.
        # 2. If no improvement, generate 5 random points. Evaluate them.
        # 3. If any is better, set best to that, and go to 1.
        # 4. If none is better, go to 1 (try local refine again, maybe with different step size?).
        #    Actually, if local refine failed, it might be stuck in a local min. 
        #    Trying local refine again from the same point is useless.
        #    So, if local refine fails, we MUST move to a new point.
        #    So: If local refine fails, we pick a new random point. If it's better, we use it. 
        #    If it's not better, we discard it and pick another? 
        #    This is essentially a random restart strategy.
        
        # Let's do:
        # While budget remains:
        #   1. Attempt local optimization from current best.
        #   2. If it improved global best:
        #        - Continue local optimization from new best (loop back to 1? or just one run?).
        #        - Let's allow multiple local runs if improving.
        #   3. If it did NOT improve global best:
        #        - Generate a new random point in [-4, 4]^n.
        #        - Evaluate it.
        #        - If it improves global best, set it as current best.
        #        - Else, keep current best.
        #        - To avoid infinite loop of failed local searches, we count failures.
        #          If we have N consecutive failed local searches, we force a "larger jump" or just accept that we are stuck and use remaining budget for more random sampling?
        #          Actually, if local search fails, it's likely a sharp ridge or we are far.
        #          Random sampling is the only way out.
        
        # Let's structure it as:
        # while calls < budget and time < limit:
        #   # Try local search
        #   if use_scipy:
        #       res = minimize(wrapped_func, current_best, method='Nelder-Mead', options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4})
        #   else:
        #       # Fallback: simple random walk
        #       ...
        
        #   if self.best_f < prev_best_f:
        #       # Improved
        #       improved = True
        #   else:
        #       improved = False
        #       # Not improved, so try a random restart
        #       x_new = rng.uniform(-4, 4, size=n)
        #       f_new = call_func(x_new)
        #       if f_new < self.best_f:
        #           current_best = x_new
        #           # If we found a new best, we should try to improve it locally next iteration
        #           continue 
        #       else:
        #           # Still stuck. 
        #           # To prevent wasting all budget on failed local searches from the same point,
        #           # we can track how many times we've failed from this specific point.
        #           # If > 3, we force a random jump even if local search is "running" (i.e., we skip local search and just sample).
            
        # Let's implement a "stagnation" counter.
        # If local search fails to improve, we increment counter.
        # If counter > 5, we do 10 random samples. If any is better, switch. If not, reset counter and try local search again? 
        # No, if local search fails 5 times, it's definitely stuck. 
        # So, if counter > 5, we switch to "pure random search mode" for a while?
        # Or just keep doing random restarts.
        
        # Let's simplify:
        # 1. Local search from best.
        # 2. If no improvement, do 1 random sample. If better, switch.
        # 3. Repeat.
        
        # This is essentially: "Local search, if stuck, random jump".
        
        # Optimization: 
        # Nelder-Mead can be expensive. 
        # Let's limit maxiter to 10-15.
        
        prev_best_f = self.best_f
        current_best = self.best_x.copy()
        consecutive_failures = 0
        
        while self.calls_made < budget and time.time() - self.start_time < time_limit:
            # Try local refinement
            prev_best_f = self.best_f
            prev_best_x = self.best_x.copy()
            
            if use_scipy:
                # Limit maxiter to keep it fast and leave budget for restarts
                # 5D NM takes ~5*maxiter calls. 
                # If maxiter=10, ~50 calls.
                # We have ~300 calls left after global phase. 
                # So ~6 local runs if they all fail? 
                # If they succeed, we use more.
                
                try:
                    res = minimize(wrapped_func, current_best, method='Nelder-Mead', 
                                   options={'maxiter': 15, 'xatol': 1e-5, 'fatol': 1e-5})
                except Exception:
                    pass
            else:
                # Fallback: Simple random walk
                step = 0.5
                for _ in range(20):
                    if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                        break
                    direction = rng.standard_normal(n)
                    direction /= np.linalg.norm(direction)
                    y = current_best + step * direction
                    y = np.clip(y, -5, 5)
                    fy = call_func(y)
                    if fy < self.best_f:
                        current_best = y
                        step *= 1.1
                    else:
                        step *= 0.9
                        if step < 1e-4:
                            break
            
            if self.best_f < prev_best_f:
                # Improved
                consecutive_failures = 0
                current_best = self.best_x.copy()
            else:
                # Failed to improve
                consecutive_failures += 1
                
                # If we've failed a few times, try a random restart
                # To avoid wasting too many calls on random restarts that don't help,
                # we only do it if we haven't made progress recently.
                
                # Let's do a random restart every time we fail?
                # Cost: 1 call.
                # Benefit: Might find a new basin.
                
                x_rand = rng.uniform(-4.0, 4.0, size=n)
                f_rand = call_func(x_rand)
                
                if f_rand < self.best_f:
                    # Found a better point!
                    current_best = x_rand
                    consecutive_failures = 0
                    # Loop will try to improve this new best next iteration
                else:
                    # Didn't find a better point.
                    # If we have failed many times in a row, maybe the local search is too weak?
                    # Or we are in a flat region?
                    # If consecutive_failures > 10, we might want to try a larger jump?
                    # But random uniform is already large.
                    # Let's just continue.
                    
                    # Safety: If we are spending too much time on failed local searches + random samples,
                    # and not improving, we might want to just burn budget with random samples?
                    # No, local search is cheaper per call if it works.
                    # But if it never works, it's wasted.
                    # However, Nelder-Mead is robust. It should find *something* if there's a gradient.
                    # If it fails, it's likely a local min.
                    # Random restart is the only way out.
                    
                    # To prevent infinite loop of "Fail Local -> Random Sample -> Fail Local -> Random Sample",
                    # we can track if we are making *any* progress.
                    # If we have done 50 iterations with no global best improvement, 
                    # we might be stuck. 
                    # But the loop condition is just budget/time.
                    # This is acceptable.
                    
                    # One optimization: If we have failed > 20 times, 
                    # maybe reduce the "maxiter" of Nelder-Mead to 5 to make it faster to fail,
                    # allowing more random restarts?
                    # Or just keep it as is.
                    
                    # Let's add a small tweak: 
                    # If consecutive_failures > 5, we skip the local search and just do random samples?
                    # No, local search might work if we start from a slightly different point?
                    # But we are starting from the *best* point. 
                    # If local search from best fails, it's a local min.
                    # So skipping local search and doing more random samples is smarter.
                    
                    if consecutive_failures > 5:
                        # Skip local search, just do random samples
                        # We already did 1 random sample above.
                        # Let's do a few more?
                        # No, the loop will iterate again.
                        # In the next iteration, if we skip local search, we just do random sample.
                        # So we need a flag.
                        
                        # Let's restructure:
                        # Instead of "Local Search then Random", 
                        # do "If failed recently, Random. Else Local."
                        
                        pass # Handled by logic below
                    
                    # To implement "skip local search if stuck", we need to check before running it.
                    # Let's refactor the loop logic.
            
            # The above loop structure is a bit messy with the "inside" random restart.
            # Let's rewrite the main loop for clarity.
            
        # Actually, the code above is already running. 
        # The `consecutive_failures` logic is inside the `else` block.
        # The "skip local search" idea requires a flag checked at the top of the loop.
        
        # Let's restart the implementation of the loop to be cleaner.
        # I will delete the previous loop code in my head and write the final version.
        
        # --- Refined Loop ---
        consecutive_failures = 0
        current_best = self.best_x.copy()
        
        while self.calls_made < budget and time.time() - self.start_time < time_limit:
            # Decide whether to do local search or random restart
            # If we have failed local search many times, we are likely in a local min.
            # In that case, random restart is better.
            do_local = (consecutive_failures < 3)
            
            if do_local:
                prev_best_f = self.best_f
                prev_best_x = self.best_x.copy()
                
                if use_scipy:
                    try:
                        res = minimize(wrapped_func, current_best, method='Nelder-Mead', 
                                       options={'maxiter': 15, 'xatol': 1e-5, 'fatol': 1e-5})
                    except Exception:
                        pass
                else:
                    step = 0.5
                    for _ in range(20):
                        if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                            break
                        direction = rng.standard_normal(n)
                        direction /= np.linalg.norm(direction)
                        y = current_best + step * direction
                        y = np.clip(y, -5, 5)
                        fy = call_func(y)
                        if fy < self.best_f:
                            current_best = y
                            step *= 1.1
                        else:
                            step *= 0.9
                            if step < 1e-4:
                                break
                
                if self.best_f < prev_best_f:
                    consecutive_failures = 0
                    current_best = self.best_x.copy()
                else:
                    consecutive_failures += 1
            else:
                # Random Restart Mode
                # Generate a few random points
                for _ in range(5):
                    if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                        break
                    x_rand = rng.uniform(-4.0, 4.0, size=n)
                    f_rand = call_func(x_rand)
                    if f_rand < self.best_f:
                        current_best = x_rand
                        consecutive_failures = 0
                        # Break out of random loop to try local search on new best
                        break
                else:
                    # All 5 random points were worse
                    # Keep consecutive_failures high to stay in random mode?
                    # Or reset to 0 to try local search again?
                    # If we are in random mode, we probably should stay there until we find something better.
                    # But if we never find something better, we waste budget.
                    # Let's keep consecutive_failures as is (or increment it) to stay in random mode.
                    # Actually, if we are in random mode, we are just sampling.
                    # Let's increment it to ensure we stay in random mode.
                    consecutive_failures += 1

        return self.best_x, self.best_f
