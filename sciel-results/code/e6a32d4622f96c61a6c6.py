import numpy as np
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
        
    # Precompute weights w_i = L // i
    weights = [0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = L // i

    # Initialize coefficients with random values from {-1, 0, 1}
    def random_coeffs():
        return [random.choice([-1, 0, 1]) for _ in range(N)]
    
    def calc_sum(coeffs):
        s = 0
        for i in range(N):
            if coeffs[i] != 0:
                s += coeffs[i] * weights[i + 1]
        return s

    best_abs_num = float('inf')
    best_coeffs = [1] + [0] * (N - 1) # Fallback
    
    # Local search strategy
    # We want to minimize |sum| subject to sum != 0.
    
    for restart in range(200): # Number of restarts
        if time.time() > deadline - 5:
            break
            
        coeffs = random_coeffs()
        current_sum = calc_sum(coeffs)
        
        # If current sum is 0, we need to move away. 
        # If current sum is nonzero, we try to reduce |current_sum|.
        
        # Simple hill climbing / local search
        for step in range(1000): # Max steps per restart
            if time.time() > deadline - 2:
                break
                
            # Find a move that improves the objective
            # Objective: minimize |sum| if sum != 0.
            # If sum == 0, we want to make it nonzero with smallest possible magnitude?
            # Actually, the task says "signed sum ... must be nonzero. Minimize its absolute value".
            # So if sum is 0, it's invalid. We must move to a state where sum != 0.
            # Among valid states, minimize |sum|.
            
            best_local_sum = current_sum
            best_local_move = None
            best_local_abs = abs(current_sum) if current_sum != 0 else float('inf')
            
            # Try flipping each coefficient
            # Optimization: only try flips that might help.
            # But N=30 is small. 30 flips per step is fine.
            
            for i in range(N):
                old_val = coeffs[i]
                w = weights[i + 1]
                
                # Try changing to the other two values
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    
                    # Calculate new sum
                    # sum_new = sum_old - old_val * w + new_val * w
                    new_sum = current_sum - old_val * w + new_val * w
                    
                    if new_sum == 0:
                        # Invalid, skip unless we are forced? 
                        # We prefer nonzero. 
                        # If current is 0, this is still 0, so bad.
                        # If current is nonzero, this is invalid.
                        continue
                    
                    new_abs = abs(new_sum)
                    
                    # We want to minimize new_abs.
                    # But if current_sum is 0 (invalid), we must accept any nonzero new_sum.
                    # If current_sum is nonzero, we accept if new_abs < best_local_abs.
                    
                    if current_sum == 0:
                        # We are in an invalid state. Any valid move is an improvement.
                        if new_abs < best_local_abs:
                            best_local_abs = new_abs
                            best_local_sum = new_sum
                            best_local_move = (i, new_val)
                    else:
                        if new_abs < best_local_abs:
                            best_local_abs = new_abs
                            best_local_sum = new_sum
                            best_local_move = (i, new_val)
            
            # Apply best move if found
            if best_local_move:
                i, new_val = best_local_move
                coeffs[i] = new_val
                current_sum = best_local_sum
                
                # Check if this is the global best
                if current_sum != 0 and abs(current_sum) < best_abs_num:
                    best_abs_num = abs(current_sum)
                    best_coeffs = coeffs[:]
                    
                if best_abs_num == 1:
                    # Can't do better than 1 in numerator (since L is integer, sum is integer/L)
                    # Wait, sum is integer. |sum| >= 1.
                    return best_coeffs
            else:
                # No improving move found. Local minimum.
                # If current state is invalid (sum=0), we need to escape.
                # If current state is valid, we are stuck.
                if current_sum == 0:
                    # Force a random move to escape zero
                    i = random.randint(0, N - 1)
                    old_val = coeffs[i]
                    w = weights[i + 1]
                    # Pick a random new value different from old
                    options = [v for v in [-1, 0, 1] if v != old_val]
                    new_val = random.choice(options)
                    coeffs[i] = new_val
                    current_sum = current_sum - old_val * w + new_val * w
                    # This might still be 0, but unlikely. 
                    # If it is 0, next iteration will try to move again.
                else:
                    # Local minimum reached. Break inner loop.
                    break

    return best_coeffs
