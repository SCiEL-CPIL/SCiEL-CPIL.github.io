import time
import random
import math
from fractions import Fraction

TIME_LIMIT = 42.0
N = 256

def can_add_fast(bitset, x):
    """
    Check if adding x to the set represented by bitset creates a 3-AP.
    Returns True if x can be added without creating a 3-AP.
    bitset: integer where bit i is set if (i+1) is in the set.
    """
    # Case 1: x is the middle term: a, x, b with a = x-d, b = x+d
    # Need a >= 1 and b <= N
    for d in range(1, min(x, N - x + 1)):
        a = x - d
        b = x + d
        if (bitset & (1 << (a - 1))) and (bitset & (1 << (b - 1))):
            return False
    
    # Case 2: x is the first term: x, x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        m = x + d
        n = x + 2 * d
        if (bitset & (1 << (m - 1))) and (bitset & (1 << (n - 1))):
            return False
    
    # Case 3: x is the last term: x-2d, x-d, x
    for d in range(1, x // 2 + 1):
        m = x - 2 * d
        n = x - d
        if (bitset & (1 << (m - 1))) and (bitset & (1 << (n - 1))):
            return False
    
    return True

def greedy_fill(bitset):
    """
    Greedily add any number that can be added to maximize reciprocal sum.
    Returns updated bitset.
    """
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if not (bitset & (1 << (x - 1))):
                if can_add_fast(bitset, x):
                    bitset |= (1 << (x - 1))
                    improved = True
    return bitset

def construct():
    start_time = time.time()
    
    # Reciprocals for numbers 1 to 256
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
    
    # Initial solution: Greedy adding in order of increasing number (decreasing reciprocal)
    current_bitset = 0
    for x in range(1, N + 1):
        if can_add_fast(current_bitset, x):
            current_bitset |= (1 << (x - 1))
    
    # Ensure maximal by greedy fill
    current_bitset = greedy_fill(current_bitset)
    
    # Calculate initial score
    current_score = 0.0
    for x in range(1, N + 1):
        if current_bitset & (1 << (x - 1)):
            current_score += recip[x]
    
    # Track best solution found
    best_bitset = current_bitset
    best_score = current_score
    
    # Simulated Annealing with better parameters
    current_temp = 0.05
    cooling_rate = 0.9995
    min_temp = 1e-7
    step_count = 0
    
    # Precompute lists for faster access
    while time.time() - start_time < TIME_LIMIT:
        if current_temp < min_temp:
            # Restart with higher temperature if we're stuck
            current_temp = 0.01
            # Random restart from best
            current_bitset = best_bitset
            current_score = best_score
        
        step_count += 1
        move_type = random.random()
        
        if move_type < 0.5:
            # Try adding a random number not in set
            # Find available numbers
            available = []
            for x in range(1, N + 1):
                if not (current_bitset & (1 << (x - 1))):
                    available.append(x)
            
            if available:
                x_add = random.choice(available)
                if can_add_fast(current_bitset, x_add):
                    delta_score = recip[x_add]
                    # Always accept improvement
                    if delta_score > 0:
                        current_bitset |= (1 << (x_add - 1))
                        current_score += delta_score
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset
                    else:
                        # Accept worse with prob exp(delta/T)
                        if random.random() < math.exp(delta_score / current_temp):
                            current_bitset |= (1 << (x_add - 1))
                            current_score += delta_score
        
        elif move_type < 0.75:
            # Try removing a random number from set
            in_set = []
            for x in range(1, N + 1):
                if current_bitset & (1 << (x - 1)):
                    in_set.append(x)
            
            if in_set:
                x_rem = random.choice(in_set)
                delta_score = -recip[x_rem]
                # Always accept if it helps escape local optima, or with prob exp(delta/T)
                if random.random() < math.exp(delta_score / current_temp):
                    current_bitset &= ~(1 << (x_rem - 1))
                    current_score += delta_score
        
        else:
            # Swap: remove one, add another
            in_set = []
            for x in range(1, N + 1):
                if current_bitset & (1 << (x - 1)):
                    in_set.append(x)
            
            not_in_set = []
            for x in range(1, N + 1):
                if not (current_bitset & (1 << (x - 1))):
                    not_in_set.append(x)
            
            if in_set and not_in_set:
                x_rem = random.choice(in_set)
                x_add = random.choice(not_in_set)
                
                # Try the swap
                trial_bitset = current_bitset & ~(1 << (x_rem - 1))
                if can_add_fast(trial_bitset, x_add):
                    trial_bitset |= (1 << (x_add - 1))
                    delta_score = recip[x_add] - recip[x_rem]
                    
                    if delta_score > 0:
                        current_bitset = trial_bitset
                        current_score += delta_score
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset
                    else:
                        if random.random() < math.exp(delta_score / current_temp):
                            current_bitset = trial_bitset
                            current_score += delta_score
        
        current_temp *= cooling_rate
        
        # Periodic local optimization: try to add any number that can be added
        if step_count % 500 == 0:
            current_bitset = greedy_fill(current_bitset)
            # Recalculate score
            current_score = 0.0
            for x in range(1, N + 1):
                if current_bitset & (1 << (x - 1)):
                    current_score += recip[x]
            if current_score > best_score:
                best_score = current_score
                best_bitset = current_bitset
    
    # Final polish: try to add any remaining numbers that can be added
    best_bitset = greedy_fill(best_bitset)
    
    # Convert bitset to list
    result = []
    for x in range(1, N + 1):
        if best_bitset & (1 << (x - 1)):
            result.append(x)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
