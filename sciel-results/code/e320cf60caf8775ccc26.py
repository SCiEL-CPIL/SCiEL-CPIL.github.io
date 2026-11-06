import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_coeffs = None
    best_abs_val = float('inf')
    
    n = 30
    reciprocals = [1.0 / i for i in range(1, n + 1)]
    
    def calc_sum(coeffs):
        s = 0.0
        for i in range(n):
            s += coeffs[i] * reciprocals[i]
        return s

    while time.time() < deadline:
        # Random initialization
        current_coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        current_sum = calc_sum(current_coeffs)
        
        # Ensure non-zero
        if current_sum == 0:
            idx = random.randint(0, n - 1)
            current_coeffs[idx] = 1
            current_sum += reciprocals[idx]
            if current_sum == 0: # extremely unlikely but possible if 1/i cancels? No, 1/i is unique.
                # If it is still 0, try another
                for k in range(n):
                    if k != idx:
                        current_coeffs[k] = 1
                        current_sum += reciprocals[k]
                        break

        # Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                old_val = current_coeffs[i]
                # Try other values
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * reciprocals[i]
                    new_sum = current_sum + delta
                    
                    # Must be non-zero
                    if new_sum == 0:
                        continue
                    
                    if abs(new_sum) < abs(current_sum):
                        current_coeffs[i] = new_val
                        current_sum = new_sum
                        improved = True
                        break # Restart scan from i=0? Or continue?
                        # Breaking the inner loop and continuing the outer loop (next i) is standard.
                        # But since the sum changed, previous indices might now be improvable.
                        # However, N=30 is small. We can just set improved=True and break the inner.
                        # The outer loop continues to i+1.
                        # This is a "steepest descent" variant.
            
            if not improved:
                # Local minimum reached. 
                # We can try a random kick to escape, or just break and restart.
                # Given the high dimensionality and many local minima, restarting is often better.
                # But let's try one kick.
                kick_idx = random.randint(0, n - 1)
                old_kick = current_coeffs[kick_idx]
                new_kick = random.choice([-1, 0, 1])
                if new_kick != old_kick:
                    delta = (new_kick - old_kick) * reciprocals[kick_idx]
                    new_sum = current_sum + delta
                    if new_sum != 0:
                        current_coeffs[kick_idx] = new_kick
                        current_sum = new_sum
                        improved = True # Try to climb again
                    else:
                        # If kick leads to 0, revert or try another
                        # Just revert for simplicity, next iter will be new random
                        pass

        # Update best
        if abs(current_sum) < best_abs_val:
            best_abs_val = abs(current_sum)
            best_coeffs = current_coeffs[:]

    return best_coeffs
