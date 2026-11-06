import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # weights[i] = L / (i+1), so sum c[i]*weights[i] = L * sum c[i]/(i+1)
    weights = [L // (i + 1) for i in range(n)]
    
    def total_num(coeffs):
        return sum(coeffs[i] * weights[i] for i in range(n))
    
    def abs_val(coeffs):
        return abs(total_num(coeffs))
    
    best_coeffs = None
    best_abs = None
    best_num = None
    
    def update_best(coeffs):
        nonlocal best_coeffs, best_abs, best_num
        num = total_num(coeffs)
        if num == 0:
            return
        a = abs(num)
        if best_abs is None or a < best_abs:
            best_abs = a
            best_num = num
            best_coeffs = list(coeffs)
    
    def check_deadline():
        return time.time() >= deadline
    
    # Phase 1: Greedy + local search with restarts
    def greedy_init():
        coeffs = [0] * n
        # Try to build a near-zero sum greedily
        remaining = list(range(n))
        random.shuffle(remaining)
        current = 0
        for i in remaining:
            # Try -1, 0, 1 and pick the one that keeps |sum| smallest
            best_c = 0
            best_abs_next = abs(current)
            for c in (-1, 0, 1):
                next_abs = abs(current + c * weights[i])
                if next_abs < best_abs_next:
                    best_abs_next = next_abs
                    best_c = c
            coeffs[i] = best_c
            current += best_c * weights[i]
        return coeffs
    
    def local_search(coeffs, max_iter=20000):
        """Best-improvement local search with systematic perturbation."""
        coeffs = list(coeffs)
        cur_abs = abs_val(coeffs)
        if cur_abs == 0:
            return coeffs, cur_abs
        
        improved = True
        while improved and not check_deadline():
            improved = False
            best_new = cur_abs
            best_idx = -1
            best_val = 0
            
            for i in range(n):
                for dv in (-2, -1, 1, 2):
                    new_val = coeffs[i] + dv
                    if new_val < -1 or new_val > 1:
                        continue
                    new_abs = abs(cur_abs - coeffs[i] * weights[i] + new_val * weights[i])
                    if new_abs < best_new:
                        best_new = new_abs
                        best_idx = i
                        best_val = new_val
                        if best_new == 0:
                            break
                if best_new == 0:
                    break
            
            if best_idx >= 0:
                coeffs[best_idx] = best_val
                cur_abs = best_new
                improved = True
                if cur_abs == 0:
                    break
        return coeffs, cur_abs
    
    # Multiple greedy restarts
    for restart in range(50):
        if check_deadline():
            break
        coeffs = greedy_init()
        coeffs, cur_abs = local_search(coeffs, max_iter=5000)
        update_best(coeffs)
    
    # Phase 2: Systematic perturbation-based restarts
    if best_coeffs is not None:
        for restart in range(100):
            if check_deadline():
                break
            coeffs = list(best_coeffs)
            # Perturb 1-3 random positions
            k = random.randint(1, 3)
            for _ in range(k):
                i = random.randint(0, n - 1)
                coeffs[i] = random.choice([-1, 0, 1])
            coeffs, cur_abs = local_search(coeffs, max_iter=5000)
            update_best(coeffs)
    
    # Phase 3: Random restarts with local search
    for restart in range(200):
        if check_deadline():
            break
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        coeffs, cur_abs = local_search(coeffs, max_iter=3000)
        update_best(coeffs)
    
    # Phase 4: If very small |N| found, do a tighter local search
    if best_coeffs is not None and best_abs < 1000000:
        for restart in range(200):
            if check_deadline():
                break
            coeffs = list(best_coeffs)
            k = random.randint(1, 2)
            for _ in range(k):
                i = random.randint(0, n - 1)
                coeffs[i] = random.choice([-1, 0, 1])
            coeffs, cur_abs = local_search(coeffs, max_iter=10000)
            update_best(coeffs)
    
    # Phase 5: Try pair-swap moves (exchange coefficients between two positions)
    if best_coeffs is not None and not check_deadline():
        coeffs = list(best_coeffs)
        cur_abs = best_abs
        for iteration in range(5000):
            if check_deadline():
                break
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    # Try swapping or exchanging values
                    for vi in (-1, 0, 1):
                        for vj in (-1, 0, 1):
                            if vi == coeffs[i] and vj == coeffs[j]:
                                continue
                            new_abs = abs(cur_abs - coeffs[i]*weights[i] - coeffs[j]*weights[j] + vi*weights[i] + vj*weights[j])
                            if new_abs < cur_abs:
                                coeffs[i] = vi
                                coeffs[j] = vj
                                cur_abs = new_abs
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                # Perturb
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                vi = random.choice([-1, 0, 1])
                vj = random.choice([-1, 0, 1])
                coeffs[i] = vi
                coeffs[j] = vj
                cur_abs = abs_val(coeffs)
        update_best(coeffs)
    
    if best_coeffs is None:
        best_coeffs = [1] + [0] * (n - 1)
    
    return best_coeffs
