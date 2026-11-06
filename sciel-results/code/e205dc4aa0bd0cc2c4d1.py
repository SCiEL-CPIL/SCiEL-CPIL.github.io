import numpy as np
import time
import random

# Precompute list of d values
_D_LIST = list(range(1, 256))

# Precompute shift amounts for faster scoring
_D_SHIFTS = [(d, 2*d) for d in _D_LIST]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    best_score = 0
    best_set = None
    all_nums = list(range(1, 513))
    
    for restart in range(200):
        if time.time() > deadline - 15:
            break
        
        # Greedy coverage initialization
        S, S_bits = _greedy_init()
        cur_score = _score_set_bits(S_bits)
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = set(S)
        
        # Targeted swap local search with simulated annealing
        T = 5.0
        T_min = 0.1
        cooling = 0.997
        no_improve = 0
        max_no_improve = 30000
        S_list = list(S)
        
        while time.time() < deadline - 5:
            if no_improve > max_no_improve:
                break
            
            # Get missing differences to target
            missing_ds = _missing_diffs(S_bits)
            
            # Try targeted swaps: pick x from S, y not in S, prioritize if y helps cover missing d
            x = random.choice(S_list)
            
            # With 70% probability, try a targeted swap for a missing d
            if missing_ds and random.random() < 0.7:
                d_target = random.choice(missing_ds)
                d2 = 2 * d_target
                candidates = []
                max_a = 512 - d2
                for a in range(1, max_a + 1):
                    a1 = a
                    a2 = a + d_target
                    a3 = a + d2
                    b1 = bool(S_bits & (1 << a1))
                    b2 = bool(S_bits & (1 << a2))
                    b3 = bool(S_bits & (1 << a3))
                    c = b1 + b2 + b3
                    if c == 2:
                        if not b1:
                            candidates.append(a1)
                        elif not b2:
                            candidates.append(a2)
                        else:
                            candidates.append(a3)
                
                if candidates:
                    y = random.choice(candidates)
                else:
                    # Fall back to random y
                    while True:
                        y = random.choice(all_nums)
                        if y not in S:
                            break
            else:
                # Random swap
                while True:
                    y = random.choice(all_nums)
                    if y not in S:
                        break
            
            # Try swap
            S_bits ^= (1 << x)  # Remove x
            S_bits ^= (1 << y)  # Add y
            
            new_score = _score_set_bits(S_bits)
            
            delta = new_score - cur_score
            if delta > 0 or random.random() < np.exp(delta / T):
                # Accept
                S.discard(x)
                S.add(y)
                S_list.remove(x)
                S_list.append(y)
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = set(S)
            else:
                # Reject, revert bits
                S_bits ^= (1 << x)
                S_bits ^= (1 << y)
                no_improve += 1
            
            T = max(T_min, T * cooling)
        
        # 2-swap polish: try pairs of swaps to escape local optima
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            if not S_list:
                break
            i = random.randrange(len(S_list))
            j = random.randrange(len(S_list))
            if i == j:
                continue
            x1 = S_list[i]
            x2 = S_list[j]
            # Find y1, y2 not in S
            while True:
                y1 = random.choice(all_nums)
                if y1 not in S:
                    break
            while True:
                y2 = random.choice(all_nums)
                if y2 not in S and y2 != y1:
                    break
            
            S_bits ^= (1 << x1)
            S_bits ^= (1 << x2)
            S_bits ^= (1 << y1)
            S_bits ^= (1 << y2)
            
            new_score = _score_set_bits(S_bits)
            delta = new_score - cur_score
            if delta > 0 or random.random() < np.exp(delta / T):
                S.discard(x1)
                S.discard(x2)
                S.add(y1)
                S.add(y2)
                S_list[i] = y1
                S_list[j] = y2
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = set(S)
            else:
                S_bits ^= (1 << x1)
                S_bits ^= (1 << x2)
                S_bits ^= (1 << y1)
                S_bits ^= (1 << y2)
        
        # Reset temperature for next restart
        T = 5.0
    
    if best_set is None:
        return list(range(1, 33))
    
    return list(best_set)

def _greedy_init():
    """Greedy set cover: select 32 elements to maximize distinct 3-AP differences."""
    S = set()
    S_bits = 0
    all_nums = list(range(1, 513))
    
    # Greedy: at each step, pick element that maximizes new differences covered
    # Use a smaller candidate pool for speed: prefer elements that create many pairs
    for _ in range(32):
        best_elem = None
        best_gain = -1
        best_new_bits = 0
        
        # Limit candidates to speed up: use all but skip if already in S
        for y in all_nums:
            if y in S:
                continue
            new_bits = S_bits | (1 << y)
            new_score = _score_set_bits(new_bits)
            
            if new_score > best_gain:
                best_gain = new_score
                best_elem = y
                best_new_bits = new_bits
        
        if best_elem is not None:
            S.add(best_elem)
            S_bits = best_new_bits
    
    return S, S_bits

def _missing_diffs(bits):
    """Return list of differences d that are NOT covered by the set."""
    missing = []
    for d, d2 in _D_SHIFTS:
        if (bits & (bits >> d) & (bits >> d2)) == 0:
            missing.append(d)
    return missing

def _score_set_bits(bits):
    """Count distinct d > 0 such that a, a+d, a+2d all in set for some a."""
    count = 0
    for d, d2 in _D_SHIFTS:
        if (bits & (bits >> d) & (bits >> d2)) != 0:
            count += 1
    return count
