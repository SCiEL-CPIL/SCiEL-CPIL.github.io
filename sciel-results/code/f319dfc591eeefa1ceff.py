import numpy as np
import time
import random

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
        cooling = 0.995
        no_improve = 0
        max_no_improve = 15000
        
        while time.time() < deadline - 5:
            if no_improve > max_no_improve:
                break
            
            # Get missing differences to target
            missing_ds = _missing_diffs(S_bits)
            
            # Try targeted swaps: pick x from S, y not in S, prioritize if y helps cover missing d
            x = random.choice(list(S))
            
            # With 70% probability, try a targeted swap for a missing d
            if missing_ds and random.random() < 0.7:
                d_target = random.choice(missing_ds)
                # Find y such that adding y could help cover d_target
                # Need a, a+d, a+2d all in S after swap
                # If we add y, we need two elements in S that are y-d and y-2d, or y+d and y+2d, or one of each
                candidates = []
                for a in range(1, 513 - 2*d_target + 1):
                    a1, a2, a3 = a, a + d_target, a + 2*d_target
                    # Check how many are already in S
                    in_s = 0
                    missing_pos = []
                    for pos, val in enumerate([a1, a2, a3]):
                        if S_bits & (1 << val):
                            in_s += 1
                        else:
                            missing_pos.append((pos, val))
                    
                    if in_s == 2 and len(missing_pos) == 1:
                        pos, val = missing_pos[0]
                        if val not in S:
                            candidates.append((val, 1))  # Adding val covers this d
                    elif in_s == 1 and len(missing_pos) == 2:
                        # Would need to add two elements, but we're only swapping one
                        # Skip for now
                        pass
                
                if candidates:
                    # Pick a random candidate
                    y, _ = random.choice(candidates)
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
    
    # Start with a random element
    S.add(random.choice(all_nums))
    S_bits |= (1 << S.pop())
    S_bits |= (1 << random.choice(all_nums))
    S = set()
    S_bits = 0
    
    # Greedy: at each step, pick element that maximizes new differences covered
    for _ in range(32):
        best_elem = None
        best_gain = -1
        best_new_bits = 0
        
        # Sample a subset of candidates for efficiency
        candidates = random.sample(all_nums, min(100, len(all_nums)))
        
        for y in candidates:
            if y in S:
                continue
            new_bits = S_bits | (1 << y)
            new_score = _score_set_bits(new_bits)
            gain = new_score  # Since we're building from scratch, gain = score
            
            if gain > best_gain:
                best_gain = gain
                best_elem = y
                best_new_bits = new_bits
        
        if best_elem is not None:
            S.add(best_elem)
            S_bits = best_new_bits
    
    return S, S_bits

def _missing_diffs(bits):
    """Return list of differences d that are NOT covered by the set."""
    missing = []
    for d in range(1, 256):
        if (bits & (bits >> d) & (bits >> (2*d))) == 0:
            missing.append(d)
    return missing

def _score_set_bits(bits):
    count = 0
    # d ranges from 1 to 255
    for d in range(1, 256):
        if (bits & (bits >> d) & (bits >> (2*d))) != 0:
            count += 1
    return count
