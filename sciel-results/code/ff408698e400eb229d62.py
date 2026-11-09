import time
import random
from math import gcd

# Precompute LCM of 1..30
def compute_lcm(n):
    lcm = 1
    for i in range(1, n + 1):
        lcm = lcm * i // gcd(lcm, i)
    return lcm

L = compute_lcm(30)
# Precompute weights w_i = L / i
weights = [L // i for i in range(1, 31)]

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s to be safe
    
    best_n = None
    best_coeffs = None
    
    # Local search function
    def local_search(c):
        # c is a list of 30 ints from {-1, 0, 1}
        n = sum(c[i] * weights[i] for i in range(30))
        
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            # Try flipping each index
            for i in range(30):
                old_val = c[i]
                # Try all other values
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * weights[i]
                    new_n = n + delta
                    # We want to minimize |new_n|, but new_n != 0
                    # If best_n is None, any non-zero is better
                    # If best_n is not None, compare |new_n| < |best_n|
                    # Also, we should prefer moves that reduce |n| locally
                    if new_n == 0:
                        # We can accept 0 to jump over, but final result must be non-zero.
                        # For now, let's say we accept if it reduces |n| or if n is large.
                        # Actually, if we hit 0, the next step will move away.
                        # Let's accept if |new_n| < |n| (which is 0 < |n|, always true if n != 0)
                        # But if n is already small, moving to 0 might be a dead end if we can't move away.
                        # However, from 0, any move is an increase in |n|.
                        # So we should probably avoid accepting 0 unless we are far away.
                        # Let's just minimize |n| and handle 0 at the end.
                        pass
                    
                    # Accept if it reduces the absolute value
                    if abs(new_n) < abs(n):
                        c[i] = new_val
                        n = new_n
                        improved = True
                        break # Restart the loop to pick another index
                if improved:
                    break
        
        return c, n

    # Run multiple restarts
    while time.time() < deadline:
        # Random initialization
        c = [random.choice([-1, 0, 1]) for _ in range(30)]
        
        # Run local search
        c, n = local_search(c)
        
        if n != 0:
            if best_n is None or abs(n) < abs(best_n):
                best_n = n
                best_coeffs = c[:]
        
        # If we found a very small value, we might stop early or continue
        # The smallest possible non-zero value is 1/L.
        if abs(best_n) == 1:
            break

    # If we ended up with n=0, we need to find a non-zero neighbor
    if best_n == 0 or best_n is None:
        # Fallback: find the smallest non-zero sum
        # This shouldn't happen if we track best_n correctly, but just in case
        # We can do a quick search from the last c
        if best_coeffs is None:
            best_coeffs = [0]*28 + [1, -1]
            best_n = weights[28] - weights[29]
        else:
            # Find the smallest non-zero n reachable by one flip
            min_n = None
            for i in range(30):
                for new_val in (-1, 0, 1):
                    if new_val == best_coeffs[i]:
                        continue
                    delta = (new_val - best_coeffs[i]) * weights[i]
                    new_n = best_n + delta
                    if new_n != 0:
                        if min_n is None or abs(new_n) < abs(min_n):
                            min_n = new_n
                            best_coeffs[i] = new_val
            best_n = min_n

    return best_coeffs
