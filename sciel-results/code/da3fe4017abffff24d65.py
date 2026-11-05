import random
import time
from math import lcm

def construct():
    deadline = time.time() + 85.0 # 5s margin
    N = 30
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
    
    # weights[i] corresponds to coefficient c_i for term 1/(i+1)
    # So weights[0] = L/1, weights[1] = L/2, ..., weights[29] = L/30
    weights = [L // (i + 1) for i in range(N)]
    
    # Initialize with a random valid state (non-zero sum)
    # A good start might be alternating signs or all 1s, but random is safer to avoid bias
    # Let's try a few random starts and keep the best
    best_coeffs = None
    best_sum = None
    best_abs = float('inf')
    
    # Precompute indices for speed
    indices = list(range(N))
    
    for _ in range(20): # 20 random restarts
        if time.time() > deadline - 5.0: break
        
        coeffs = [random.choice([-1, 0, 1]) for _ in range(N)]
        s = sum(c * w for c, w in zip(coeffs, weights))
        
        if s == 0:
            # If sum is 0, we must change one bit to make it non-zero
            # Find a bit with max weight to change? Or just pick one.
            # Changing any bit to a different value will make it non-zero unless the new value is same (impossible) or results in 0 (possible if s=0 and we change c to c' such that (c'-c)w = 0, which implies c'=c, but we change it).
            # Wait, if s=0, and we change c_k to c_k', new sum = (c_k'-c_k)w_k. This is non-zero if c_k' != c_k.
            # So just flip one bit.
            k = random.randint(0, N-1)
            old = coeffs[k]
            new = random.choice([v for v in [-1, 0, 1] if v != old])
            coeffs[k] = new
            s += (new - old) * weights[k]
            
        abs_s = abs(s)
        if abs_s < best_abs:
            best_abs = abs_s
            best_sum = s
            best_coeffs = coeffs[:]
            
        # Local Search from this start
        current_coeffs = coeffs[:]
        current_sum = s
        current_abs = abs_s
        
        # Hill Climbing with 1-flip and 2-flip moves
        # We want to minimize current_abs
        # If current_abs becomes 0, we must kick out.
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline - 2.0:
            improved = False
            
            # Try 1-flip moves
            # To speed up, we can iterate through a random subset of indices or all indices
            # Let's try a random permutation of indices to avoid bias
            random.shuffle(indices)
            
            for k in indices:
                if time.time() > deadline - 2.0: break
                
                old_c = current_coeffs[k]
                w = weights[k]
                
                # Try changing to other 2 values
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    
                    delta = (new_c - old_c) * w
                    new_sum = current_sum + delta
                    
                    if new_sum == 0:
                        # Invalid if we care about non-zero, but we can use it as a stepping stone?
                        # The task says "signed sum ... must be nonzero".
                        # If we land on 0, we must immediately move away or reject.
                        # Rejecting is safer for local search unless we are stuck.
                        continue
                    
                    new_abs = abs(new_sum)
                    
                    if new_abs < current_abs:
                        # Accept
                        current_coeffs[k] = new_c
                        current_sum = new_sum
                        current_abs = new_abs
                        improved = True
                        
                        # Update global best
                        if current_abs < best_abs:
                            best_abs = current_abs
                            best_sum = current_sum
                            best_coeffs = current_coeffs[:]
                        
                        # Restart inner loop or continue?
                        # Continue to see if we can improve further
                        break # Break inner loop for this k, try next k or restart?
                        # Breaking inner loop allows trying other k's.
                    
            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    # Kick: change 2-3 random bits to escape local minimum
                    for _ in range(3):
                        k = random.randint(0, N-1)
                        old_c = current_coeffs[k]
                        new_c = random.choice([v for v in [-1, 0, 1] if v != old_c])
                        current_coeffs[k] = new_c
                        current_sum += (new_c - old_c) * weights[k]
                    
                    if current_sum == 0:
                        # Ensure non-zero
                        k = random.randint(0, N-1)
                        old_c = current_coeffs[k]
                        new_c = random.choice([v for v in [-1, 0, 1] if v != old_c])
                        current_coeffs[k] = new_c
                        current_sum += (new_c - old_c) * weights[k]
                    
                    current_abs = abs(current_sum) if current_sum != 0 else float('inf')
                    no_improve = 0
            else:
                no_improve = 0
                
    return best_coeffs
