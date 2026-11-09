import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Precompute LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # Integer weights: weight[i] = L // (i+1) for coefficient of 1/(i+1)
    weights = [L // (i + 1) for i in range(30)]
    
    def get_sum(c):
        """Compute integer sum of c[i] * weights[i]"""
        s = 0
        for i in range(30):
            if c[i] != 0:
                s += c[i] * weights[i]
        return s
    
    # Global best
    best_solution = None
    best_abs_sum = float('inf')
    
    def update_best(c, abs_sum):
        nonlocal best_solution, best_abs_sum
        if abs_sum < best_abs_sum:
            best_abs_sum = abs_sum
            best_solution = list(c)
    
    def hill_climb_step(c, current_sum, best_abs):
        """Try single and double flips to improve. Returns (new_c, new_sum, improved)"""
        # Try single-coordinate changes
        for i in range(30):
            for new_val in (-1, 0, 1):
                if new_val == c[i]:
                    continue
                delta = (new_val - c[i]) * weights[i]
                new_sum = current_sum + delta
                if new_sum == 0:
                    continue
                new_abs = abs(new_sum)
                if new_abs < best_abs:
                    c[i] = new_val
                    return c, new_sum, True
        
        # Try pairwise changes
        for i in range(30):
            for j in range(i + 1, 30):
                for ni in (-1, 0, 1):
                    if ni == c[i]:
                        continue
                    for nj in (-1, 0, 1):
                        if nj == c[j]:
                            continue
                        delta = (ni - c[i]) * weights[i] + (nj - c[j]) * weights[j]
                        new_sum = current_sum + delta
                        if new_sum == 0:
                            continue
                        new_abs = abs(new_sum)
                        if new_abs < best_abs:
                            c[i] = ni
                            c[j] = nj
                            return c, new_sum, True
        
        return c, current_sum, False
    
    def simulated_annealing_escape(c, current_sum, best_abs, T):
        """Try a random move with simulated annealing acceptance. Returns (new_c, new_sum, accepted)"""
        # Pick 1 or 2 random coordinates to change
        num_changes = random.choice([1, 2])
        indices = random.sample(range(30), num_changes)
        
        old_vals = [c[i] for i in indices]
        new_vals = [random.choice([-1, 0, 1]) for _ in indices]
        
        if all(old_vals[k] == new_vals[k] for k in range(num_changes)):
            return c, current_sum, False
        
        # Compute new sum
        delta = sum((new_vals[k] - old_vals[k]) * weights[indices[k]] for k in range(num_changes))
        new_sum = current_sum + delta
        
        if new_sum == 0:
            return c, current_sum, False
        
        new_abs = abs(new_sum)
        
        # Acceptance criterion
        if new_abs < best_abs:
            # Always accept improvement
            for k in range(num_changes):
                c[indices[k]] = new_vals[k]
            return c, new_sum, True
        else:
            # Accept worse with probability exp(-(new_abs - best_abs) / T)
            delta_E = new_abs - best_abs
            if delta_E < T * 100:  # Avoid overflow
                prob = min(1.0, max(0.0, 1.0 - delta_E / (T * 1000)))
                if random.random() < prob:
                    for k in range(num_changes):
                        c[indices[k]] = new_vals[k]
                    return c, new_sum, True
        
        return c, current_sum, False
    
    def local_search_with_sa(c, max_iter=200000):
        """Hill-climbing with simulated annealing escape when stuck."""
        current_sum = get_sum(c)
        if current_sum == 0:
            # Force nonzero
            for i in range(30):
                if c[i] == 0:
                    c[i] = 1
                    break
                else:
                    c[i] = -c[i]
                    break
            current_sum = get_sum(c)
            if current_sum == 0:
                c[0] = 1
                current_sum = get_sum(c)
        
        best_abs_local = abs(current_sum)
        update_best(c, best_abs_local)
        
        # Temperature schedule
        T = 1e6
        T_min = 1.0
        cooling_rate = 0.999
        stuck_count = 0
        max_stuck = 5000
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try hill climbing
            c, current_sum, improved = hill_climb_step(c, current_sum, best_abs_local)
            
            if improved:
                best_abs_local = abs(current_sum)
                update_best(c, best_abs_local)
                stuck_count = 0
                T = max(T_min, T * cooling_rate)
            else:
                stuck_count += 1
                # If stuck for a while, try simulated annealing escape
                if stuck_count > 100 and T > T_min:
                    c, current_sum, accepted = simulated_annealing_escape(c, current_sum, best_abs_local, T)
                    if accepted:
                        best_abs_local = min(best_abs_local, abs(current_sum))
                        update_best(c, best_abs_local)
                        stuck_count = 0
                        T = max(T_min, T * cooling_rate)
                    else:
                        T = max(T_min, T * cooling_rate)
                
                # If very stuck, break and restart
                if stuck_count > max_stuck:
                    break
        
        return c, abs(current_sum)
    
    # Multiple random restarts with different strategies
    num_restarts = 80
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Different initialization strategies
        strategy = restart % 4
        
        if strategy == 0:
            # Sparse: few nonzeros
            c = [0] * 30
            k = random.randint(2, 8)
            positions = random.sample(range(30), k)
            for pos in positions:
                c[pos] = random.choice([-1, 1])
        elif strategy == 1:
            # Medium sparse
            c = [0] * 30
            k = random.randint(8, 15)
            positions = random.sample(range(30), k)
            for pos in positions:
                c[pos] = random.choice([-1, 1])
        elif strategy == 2:
            # Dense random
            c = [random.choice([-1, 0, 1]) for _ in range(30)]
        else:
            # Seed-like: start with adjacent pair
            c = [0] * 30
            i = random.randint(0, 28)
            c[i] = 1
            c[i + 1] = -1
            # Add some random perturbations
            for _ in range(random.randint(0, 5)):
                j = random.randint(0, 29)
                c[j] = random.choice([-1, 0, 1])
        
        # Ensure nonzero sum
        s = get_sum(c)
        if s == 0:
            for i in range(30):
                if c[i] == 0:
                    c[i] = 1
                    break
                else:
                    c[i] = -c[i]
                    break
            s = get_sum(c)
            if s == 0:
                c[0] = 1
                s = get_sum(c)
        
        # Run local search with simulated annealing
        c, abs_sum = local_search_with_sa(c)
        
        if abs_sum < best_abs_sum:
            best_abs_sum = abs_sum
            best_solution = list(c)
        
        # Early termination if we found a very good solution
        if best_abs_sum <= 1:
            break
    
    # Return best found, or fallback to seed
    if best_solution is not None:
        # Verify it's valid
        if get_sum(best_solution) != 0:
            return best_solution
    
    # Fallback: seed solution
    return [0] * 28 + [1, -1]
