"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # 5s safety margin from 60s wall-clock limit
    
    N = 30
    
    # Seed solution: [0]*28 + [1, -1] gives sum = 1/29 - 1/30 = 1/870
    best_solution = [0] * 28 + [1, -1]
    best_val = Fraction(1, 870)
    
    def eval_solution(coeffs):
        """Evaluate the absolute value of the signed harmonic sum."""
        s = Fraction(0)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                s += Fraction(c, i)
        if s == 0:
            return None
        return abs(s)
    
    # Strategy: Use a multi-phase approach
    # Phase 1: Try known structured patterns
    # Phase 2: Local search from seed
    # Phase 3: Random restarts with local search
    
    random.seed(42)
    
    # Phase 1: Try some structured patterns that might give small sums
    # These are hand-crafted or based on number theory insights
    
    structured_patterns = [
        # Seed
        [0]*28 + [1, -1],
        # Try consecutive pairs
        [1, -1] + [0]*28,
        [0, 1, -1] + [0]*27,
        [0]*2 + [1, -1] + [0]*26,
        # Try using more terms to cancel
        [1, 0, -1, 0, 0, -1] + [0]*24,  # 1 - 1/3 - 1/6 = 1/2
        [0, 1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 1] + [0]*18,  # 1/2 - 1/3 - 1/6 + 1/12 = 0 (bad)
        [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
        # Try symmetric patterns
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
        # Try using reciprocals that sum to small values
        [0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/2 - 1/6 = 1/3
        [0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/3 - 1/6 = 1/6
        [0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/4 - 1/6 = 1/12
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/5 - 1/6 = 1/30
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/6 - 1/7 = 1/42
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/7 - 1/8 = 1/56
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/8 - 1/9 = 1/72
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/9 - 1/10 = 1/90
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/10 - 1/11 = 1/110
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/11 - 1/12 = 1/132
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/12 - 1/13 = 1/156
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/13 - 1/14 = 1/182
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/14 - 1/15 = 1/210
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/15 - 1/16 = 1/240
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/16 - 1/17 = 1/272
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/17 - 1/18 = 1/306
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/18 - 1/19 = 1/342
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/19 - 1/20 = 1/380
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/20 - 1/21 = 1/420
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1/21 - 1/22 = 1/462
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],  # 1/22 - 1/23 = 1/506
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # 1/23 - 1/24 = 1/552
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],  # 1/24 - 1/25 = 1/600
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # 1/25 - 1/26 = 1/650
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],  # 1/26 - 1/27 = 1/702
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],  # 1/27 - 1/28 = 1/756
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],  # 1/28 - 1/29 = 1/812
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],  # 1/29 - 1/30 = 1/870 (seed)
    ]
    
    for pattern in structured_patterns:
        if len(pattern) != N:
            continue
        val = eval_solution(pattern)
        if val is not None and val < best_val:
            best_val = val
            best_solution = pattern[:]
    
    # Phase 2: Local search from current best
    current = best_solution[:]
    current_val = best_val
    
    def local_search(start_coeffs, max_time=5.0):
        """Perform local search, return (best_coeffs, best_val)."""
        nonlocal best_solution, best_val
        local_best = start_coeffs[:]
        local_val = eval_solution(local_best)
        if local_val is None:
            return local_best, Fraction(1, 1)  # Large value
        
        local_start = time.time()
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() - local_start < max_time and no_improve_count < max_no_improve:
            improved = False
            # Try all single flips
            for i in range(N):
                old = local_best[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    local_best[i] = new_val
                    new_abs = eval_solution(local_best)
                    if new_abs is not None and new_abs < local_val:
                        local_val = new_abs
                        improved = True
                        no_improve_count = 0
                        # Update global best
                        if local_val < best_val:
                            best_val = local_val
                            best_solution = local_best[:]
                        break
                    else:
                        local_best[i] = old
                if improved:
                    break
                local_best[i] = old
            
            if not improved:
                no_improve_count += 1
                # Try a random perturbation to escape local minimum
                if no_improve_count % 10 == 0:
                    for _ in range(3):
                        i = random.randint(0, N - 1)
                        old = local_best[i]
                        options = [v for v in [-1, 0, 1] if v != old]
                        local_best[i] = random.choice(options)
                        pert_val = eval_solution(local_best)
                        if pert_val is not None and pert_val < local_val:
                            local_val = pert_val
                            no_improve_count = 0
                            if local_val < best_val:
                                best_val = local_val
                                best_solution = local_best[:]
                        else:
                            local_best[i] = old
        
        return local_best, local_val
    
    # Run local search from current best
    if time.time() - start_time < deadline - 10:
        current, current_val = local_search(current, max_time=10.0)
    
    # Phase 3: Random restarts with local search
    restart_count = 0
    while time.time() - start_time < deadline - 5:
        restart_count += 1
        # Generate a random starting point
        # Bias towards sparse solutions (more zeros)
        fresh = []
        for _ in range(N):
            r = random.random()
            if r < 0.4:
                fresh.append(0)
            elif r < 0.7:
                fresh.append(1)
            else:
                fresh.append(-1)
        
        fresh_val = eval_solution(fresh)
        if fresh_val is None:
            continue
        
        # If this is already good, do local search
        if fresh_val < best_val or random.random() < 0.3:
            if time.time() - start_time < deadline - 3:
                fresh, fresh_val = local_search(fresh, max_time=3.0)
    
    # Final check
    if time.time() - start_time > deadline:
        return best_solution
    
    # Verify the solution is valid
    final_val = eval_solution(best_solution)
    if final_val is None:
        # Fallback to seed
        return [0] * 28 + [1, -1]
    
    return best_solution

# EVOLVE-BLOCK-END
