import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    n = 30
    best_abs_val = None
    best_coeffs = None
    
    # Precompute inverses
    inv = [Fraction(1, i + 1) for i in range(n)]
    
    # Initial valid nonzero solution
    init_coeffs = [0] * n
    init_coeffs[0] = 1
    best_abs_val = Fraction(1)
    best_coeffs = init_coeffs[:]

    def compute_sum(coeffs):
        s = Fraction(0)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * inv[i]
        return s

    def local_search(start_coeffs):
        coeffs = start_coeffs[:]
        current_sum = compute_sum(coeffs)
        
        if current_sum == 0:
            # Perturb to make it nonzero
            idx = random.randint(0, n - 1)
            old = coeffs[idx]
            new = 1 if old != 1 else -1
            coeffs[idx] = new
            current_sum += (new - old) * inv[idx]
            if current_sum == 0:
                # Force a change
                coeffs[0] = 1
                coeffs[1] = -1
                current_sum = compute_sum(coeffs)
                if current_sum == 0:
                    return None, None

        improved = True
        while improved:
            improved = False
            for i in range(n):
                old_val = coeffs[i]
                candidates = [v for v in (-1, 0, 1) if v != old_val]
                random.shuffle(candidates)
                
                for new_val in candidates:
                    delta = (new_val - old_val) * inv[i]
                    new_sum = current_sum + delta
                    
                    if new_sum == 0:
                        continue
                        
                    new_abs = abs(new_sum)
                    current_abs = abs(current_sum)
                    
                    if new_abs < current_abs:
                        coeffs[i] = new_val
                        current_sum = new_sum
                        improved = True
                        break # Restart for-loop to check other indices with new state
                
                if improved:
                    break
            
            if not improved:
                # Escape local minimum with a random jump
                # Only do this if we have time? The loop condition handles global time.
                # But we should avoid infinite loops if random jumps don't help.
                # Since we only jump when no single flip improves, and the space is finite,
                # we might cycle. But with random choices, it's unlikely to cycle indefinitely.
                # To be safe, we can limit the number of consecutive non-improving steps?
                # Or just rely on the global deadline.
                
                idx = random.randint(0, n - 1)
                old_val = coeffs[idx]
                candidates = [v for v in (-1, 0, 1) if v != old_val]
                new_val = random.choice(candidates)
                delta = (new_val - old_val) * inv[idx]
                new_sum = current_sum + delta
                if new_sum != 0:
                    coeffs[idx] = new_val
                    current_sum = new_sum
                    # We don't set improved=True, so the while loop will terminate
                    # unless the next iteration finds an improvement.
                    # This means we jump, then try to improve from the new spot.
                    # If no improvement, we jump again.
                    # This is fine.
        
        return coeffs, abs(current_sum)

    # Multi-start search
    while time.time() < deadline:
        start_coeffs = [random.choice((-1, 0, 1)) for _ in range(n)]
        
        s = compute_sum(start_coeffs)
        if s == 0:
            idx = random.randint(0, n - 1)
            old = start_coeffs[idx]
            new = 1 if old != 1 else -1
            start_coeffs[idx] = new
            s += (new - old) * inv[idx]
            if s == 0:
                continue

        res_coeffs, res_abs = local_search(start_coeffs)
        
        if res_coeffs is not None and res_abs is not None:
            if best_abs_val is None or res_abs < best_abs_val:
                best_abs_val = res_abs
                best_coeffs = res_coeffs

    return best_coeffs
