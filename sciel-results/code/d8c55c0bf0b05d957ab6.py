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

    # Initialize random coefficients
    coeffs = [random.choice([-1, 0, 1]) for _ in range(N)]
    
    # Calculate initial sum
    current_sum = 0
    for i in range(N):
        if coeffs[i] != 0:
            current_sum += coeffs[i] * weights[i + 1]
            
    best_abs_num = float('inf')
    best_coeffs = coeffs[:]
    
    if current_sum != 0:
        best_abs_num = abs(current_sum)
        if best_abs_num == 1:
            return best_coeffs

    # Simulated Annealing
    # Temperature schedule
    T_start = L  # Large
    T_end = 1
    # We want to run for a fixed number of steps or time
    # Let's estimate steps. Python loop ~ 5-10M steps in 55s.
    # Let's use time-based loop.
    
    # Cooling rate
    # We don't know exact steps in advance, so we can use a time-based temperature decay
    # Or just a fixed number of steps with a good cooling schedule.
    # Let's do 5,000,000 steps.
    num_steps = 5000000
    # To ensure we cover the range, we can just run until time limit.
    
    # Better: run loops, check time.
    # Temperature update: T = T_start * (T_end/T_start)^(step/num_steps)
    # But num_steps is arbitrary. Let's just use a simple cooling factor per step.
    # Or, simpler: just use a fixed T for a while, then reduce?
    # Standard SA: T = T0 * alpha^t
    
    # Let's try a fixed number of iterations per "epoch" of temperature?
    # No, let's just use a simple time-based check.
    
    # We'll use a cooling factor that depends on the step count, but since we break on time,
    # we can just use a small alpha.
    # Let's assume ~5M steps.
    # alpha = (T_end/T_start)^(1/5000000)
    # This is very close to 1.
    # Let's just use a simple schedule: T decreases linearly or exponentially.
    
    # Alternative: Random Restarting Local Search.
    # But SA is better for escaping local minima.
    
    # Let's stick to SA.
    # T = L * exp(-k * t) ?
    # Let's just use a fixed T for the first half, then lower it?
    # No, let's use the standard geometric cooling with an estimated total steps.
    # If we run out of time, we just return best.
    
    # Estimate 5M steps.
    total_steps_estimate = 5000000
    alpha = (T_end / T_start) ** (1.0 / total_steps_estimate)
    
    T = T_start
    step = 0
    
    while time.time() < deadline:
        # Pick random index
        i = random.randint(0, N - 1)
        old_val = coeffs[i]
        
        # Pick new value from {-1, 0, 1} \ {old_val}
        # Options are the other two
        if old_val == -1:
            new_val = random.choice([0, 1])
        elif old_val == 0:
            new_val = random.choice([-1, 1])
        else: # old_val == 1
            new_val = random.choice([-1, 0])
            
        delta = (new_val - old_val) * weights[i + 1]
        new_sum = current_sum + delta
        
        old_cost = abs(current_sum) if current_sum != 0 else L
        new_cost = abs(new_sum) if new_sum != 0 else L
        
        # Acceptance criterion
        if new_cost < old_cost:
            # Always accept improvement
            coeffs[i] = new_val
            current_sum = new_sum
            step += 1
            T *= alpha
            
            # Update best
            if new_sum != 0 and new_cost < best_abs_num:
                best_abs_num = new_cost
                best_coeffs = coeffs[:]
                if best_abs_num == 1:
                    return best_coeffs
        else:
            # Accept with probability exp(-(new_cost - old_cost) / T)
            if new_cost == old_cost:
                # No change in cost, accept with prob 0.5 or something? 
                # Actually if cost is same, it's neutral.
                if random.random() < 0.5:
                    coeffs[i] = new_val
                    current_sum = new_sum
                    step += 1
                    T *= alpha
            else:
                # new_cost > old_cost
                # Calculate delta E
                delta_E = new_cost - old_cost
                # If T is small, exp(-delta_E/T) is very small.
                # Optimization: if delta_E > T * 20, exp is ~0
                if delta_E > T * 20:
                    pass # Reject
                else:
                    prob = 2.718281828 ** (-delta_E / T) # Approx e
                    if random.random() < prob:
                        coeffs[i] = new_val
                        current_sum = new_sum
                        step += 1
                        T *= alpha
        
        # Check time more frequently to be safe
        if time.time() > deadline - 0.5:
            break

    # Ensure we return a valid non-zero sum
    # If best_coeffs sum is 0 (shouldn't happen if we tracked correctly, but just in case)
    # We tracked best_nonzero, so best_coeffs should be non-zero sum.
    # However, if we never found a non-zero sum (impossible, initial is likely non-zero), 
    # we need a fallback.
    
    # Verify best_coeffs
    check_sum = 0
    for i in range(N):
        if best_coeffs[i] != 0:
            check_sum += best_coeffs[i] * weights[i + 1]
            
    if check_sum == 0:
        # Fallback: find any non-zero neighbor
        # Just flip one bit
        for i in range(N):
            if best_coeffs[i] == 0:
                best_coeffs[i] = 1
                break
            elif best_coeffs[i] == 1:
                best_coeffs[i] = -1
                break
            else:
                best_coeffs[i] = 0
                break
        # Recalculate to be sure
        check_sum = 0
        for i in range(N):
            if best_coeffs[i] != 0:
                check_sum += best_coeffs[i] * weights[i + 1]
        if check_sum == 0:
            # Extremely unlikely, but force a non-zero
            best_coeffs[0] = 1
            best_coeffs[1] = 0
            best_coeffs[2] = 0
            
    return best_coeffs
