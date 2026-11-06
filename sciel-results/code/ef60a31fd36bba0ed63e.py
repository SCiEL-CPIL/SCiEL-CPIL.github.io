import itertools
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s kill
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
        
    # Precompute weights w_i = L // i
    weights = [0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = L // i

    best_abs_num = float('inf')
    best_coeffs = [1] + [0] * (N - 1)  # Fallback: sum = 1

    # Phase 1: Systematic search over small support sets (k=2..6)
    # We search for the minimum non-zero absolute numerator among sparse solutions.
    
    max_k = 6
    
    for k in range(2, max_k + 1):
        if time.time() > deadline - 30:  # Leave 30s for Phase 2
            break
            
        indices = range(1, N + 1)
        
        for combo in itertools.combinations(indices, k):
            # Get weights for this combo
            combo_weights = [weights[idx] for idx in combo]
            total_pos = sum(combo_weights)
            
            # Iterate over all masks
            for mask in range(1 << k):
                # Calculate subset sum
                subset_sum = 0
                for j in range(k):
                    if (mask >> j) & 1:
                        subset_sum += combo_weights[j]
                
                num = total_pos - 2 * subset_sum
                if num != 0:
                    abs_num = abs(num)
                    if abs_num < best_abs_num:
                        best_abs_num = abs_num
                        coeffs = [0] * N
                        for j in range(k):
                            idx = combo[j]
                            if (mask >> j) & 1:
                                coeffs[idx - 1] = -1
                            else:
                                coeffs[idx - 1] = 1
                        best_coeffs = coeffs
                        if best_abs_num == 1:
                            return best_coeffs

    # Phase 2: Simulated Annealing / Local Search
    # We start from the best sparse solution found, and try to improve it.
    
    if best_abs_num == float('inf'):
        # Fallback to random if Phase 1 found nothing (unlikely)
        coeffs = [random.choice([-1, 0, 1]) for _ in range(N)]
        current_sum = sum(coeffs[i] * weights[i+1] for i in range(N))
        if current_sum != 0:
            best_abs_num = abs(current_sum)
            best_coeffs = coeffs[:]
    else:
        coeffs = best_coeffs[:]
        current_sum = sum(coeffs[i] * weights[i+1] for i in range(N))
        if current_sum == 0:
            # Should not happen if Phase 1 worked, but just in case
            coeffs[0] = 1
            current_sum = weights[1]
            best_abs_num = abs(current_sum)
            best_coeffs = coeffs[:]

    # SA Parameters
    T_start = max(10.0, best_abs_num * 0.1)
    T_end = 0.1
    T = T_start
    
    step = 0
    total_steps_estimate = 5000000
    alpha = (T_end / T_start) ** (1.0 / total_steps_estimate)
    
    while time.time() < deadline - 2:
        i = random.randint(0, N - 1)
        old_val = coeffs[i]
        
        # Pick new value
        if old_val == -1:
            new_val = random.choice([0, 1])
        elif old_val == 0:
            new_val = random.choice([-1, 1])
        else:
            new_val = random.choice([-1, 0])
            
        delta = (new_val - old_val) * weights[i + 1]
        new_sum = current_sum + delta
        
        old_cost = abs(current_sum) if current_sum != 0 else L
        new_cost = abs(new_sum) if new_sum != 0 else L
        
        if new_cost < old_cost:
            coeffs[i] = new_val
            current_sum = new_sum
            step += 1
            T *= alpha
            
            if new_sum != 0 and new_cost < best_abs_num:
                best_abs_num = new_cost
                best_coeffs = coeffs[:]
                if best_abs_num == 1:
                    return best_coeffs
        else:
            if new_cost == old_cost:
                if random.random() < 0.5:
                    coeffs[i] = new_val
                    current_sum = new_sum
                    step += 1
                    T *= alpha
            else:
                delta_E = new_cost - old_cost
                if delta_E > T * 20:
                    pass
                else:
                    prob = 2.718281828 ** (-delta_E / T)
                    if random.random() < prob:
                        coeffs[i] = new_val
                        current_sum = new_sum
                        step += 1
                        T *= alpha

        if step % 10000 == 0 and time.time() > deadline - 2:
            break

    # Final verification
    check_sum = sum(best_coeffs[i] * weights[i+1] for i in range(N))
    if check_sum == 0:
        # Force non-zero
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
        check_sum = sum(best_coeffs[i] * weights[i+1] for i in range(N))
        if check_sum == 0:
            best_coeffs[0] = 1
            best_coeffs[1] = 0

    return best_coeffs
