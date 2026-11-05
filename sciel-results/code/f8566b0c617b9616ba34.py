import time
import random
import math
from fractions import Fraction

def construct():
        start_time = time.time()
        deadline = start_time + 45  # Safety margin for 60s limit
        
        n = 30
        inv = [Fraction(1, i) for i in range(1, n + 1)]
        
        best_sum = None
        best_coeffs = None
        best_abs = None
        
        def evaluate(coeffs):
            s = Fraction(0)
            for i, c in enumerate(coeffs):
                if c != 0:
                    s += c * inv[i]
            return s

        def local_search(coeffs, time_limit):
            nonlocal best_sum, best_coeffs, best_abs
            curr = list(coeffs)
            curr_sum = evaluate(curr)
            curr_abs = abs(curr_sum)
            
            if curr_sum == 0:
                # Should not happen for non-zero coeffs, but safe check
                return curr, curr_sum

            if best_abs is None or curr_abs < best_abs:
                best_sum = curr_sum
                best_coeffs = list(curr)
                best_abs = curr_abs

            improved = True
            # Limit iterations in local search to avoid getting stuck in a very slow loop if stuck
            max_iters = 50 
            iters = 0
            while improved and iters < max_iters and (time.time() - start_time) < time_limit:
                improved = False
                best_flip_idx = -1
                best_flip_val = None
                best_new_sum = None
                best_new_abs = curr_abs
                
                # Check all possible single flips
                for i in range(n):
                    curr_val = curr[i]
                    # Possible new values: -1, 0, 1 excluding curr_val
                    for new_val in (-1, 0, 1):
                        if new_val == curr_val:
                            continue
                        # Delta is (new_val - curr_val) * inv[i]
                        delta = (new_val - curr_val) * inv[i]
                        new_sum = curr_sum + delta
                        if new_sum == 0:
                            # Zero is excluded, but if we hit it, we can't keep it.
                            # However, the problem says "nonzero". 
                            # If we hit 0, we must move away. 
                            # But usually 0 is not reachable with these coeffs? 
                            # Actually, sum 1/1 - 1/2 - 1/3 - 1/6 = 0. 
                            # So 0 is reachable. We must avoid it.
                            continue
                        new_abs = abs(new_sum)
                        if new_abs < best_new_abs:
                            best_new_abs = new_abs
                            best_flip_idx = i
                            best_flip_val = new_val
                            best_new_sum = new_sum
                
                if best_flip_idx >= 0:
                    curr[best_flip_idx] = best_flip_val
                    curr_sum = best_new_sum
                    curr_abs = best_new_abs
                    improved = True
                    iters += 1
                    if best_abs is None or curr_abs < best_abs:
                        best_sum = curr_sum
                        best_coeffs = list(curr)
                        best_abs = curr_abs
            
            return curr, curr_sum

        def simulated_annealing(coeffs, steps, time_limit):
            nonlocal best_sum, best_coeffs, best_abs
            curr = list(coeffs)
            curr_sum = evaluate(curr)
            curr_abs = abs(curr_sum)
            
            if curr_sum == 0:
                return curr, curr_sum

            if best_abs is None or curr_abs < best_abs:
                best_sum = curr_sum
                best_coeffs = list(curr)
                best_abs = curr_abs

            T = 0.01  # Initial temperature
            T_min = 1e-9
            cooling_rate = 0.95
            
            for step in range(steps):
                if (time.time() - start_time) > time_limit:
                    break
                if T < T_min:
                    break
                
                # Pick random index and new value
                i = random.randint(0, n - 1)
                curr_val = curr[i]
                possible_vals = [v for v in (-1, 0, 1) if v != curr_val]
                new_val = random.choice(possible_vals)
                
                delta_coeff = new_val - curr_val
                if delta_coeff == 0:
                    continue
                
                new_sum = curr_sum + delta_coeff * inv[i]
                
                if new_sum == 0:
                    # Reject zero sum
                    continue
                
                new_abs = abs(new_sum)
                delta_abs = new_abs - curr_abs
                
                # Acceptance criterion
                if delta_abs < 0:
                    # Always accept improvement
                    curr[i] = new_val
                    curr_sum = new_sum
                    curr_abs = new_abs
                    if best_abs is None or curr_abs < best_abs:
                        best_sum = curr_sum
                        best_coeffs = list(curr)
                        best_abs = curr_abs
                else:
                    # Accept with probability exp(-delta/T)
                    # delta_abs is a Fraction. Convert to float for exp?
                    # Careful with overflow/underflow.
                    # delta_abs is small.
                    try:
                        delta_float = float(delta_abs)
                        T_float = float(T)
                        prob = math.exp(-delta_float / T_float)
                    except OverflowError:
                        prob = 0.0
                    except:
                        prob = 0.0
                    
                    if random.random() < prob:
                        curr[i] = new_val
                        curr_sum = new_sum
                        curr_abs = new_abs
                        if best_abs is None or curr_abs < best_abs:
                            best_sum = curr_sum
                            best_coeffs = list(curr)
                            best_abs = curr_abs
                
                T *= cooling_rate
            
            return curr, curr_sum

        random.seed(42)
        
        # Initial random restarts with Local Search
        # 1. Seed solution
        seed_coeffs = [0] * 28 + [1, -1]
        local_search(seed_coeffs, deadline)
        
        # 2. Random restarts
        restarts = 0
        while (time.time() - start_time) < deadline - 5: # Leave 5s for final polish
            restarts += 1
            # Random init
            coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
            
            # Local search
            coeffs, _ = local_search(coeffs, deadline)
            
            # Simulated Annealing
            # Number of steps depends on time left
            time_left = deadline - time.time()
            if time_left < 1:
                break
            
            # Estimate steps: 1 step ~ 10-20us? 
            # Let's try 1000 steps per SA run
            sa_steps = 1000
            coeffs, _ = simulated_annealing(coeffs, sa_steps, deadline)
            
            # Final local search
            coeffs, _ = local_search(coeffs, deadline)
            
            # If time is running low, break
            if (time.time() - start_time) > deadline - 5:
                break

        # Final polish on the best found
        if best_coeffs is not None:
            # Run local search one more time on the best found to ensure it's a local min
            final_coeffs, _ = local_search(best_coeffs, deadline)
            # Check if this final polish improved it
            # local_search updates global bests, so best_coeffs is already updated if improved
            
        if best_coeffs is None:
            return [0] * 28 + [1, -1]
            
        return best_coeffs
