"""Rewritten program using bitset-based scoring for fast 3-AP difference counting."""
import time
import random

def _score_bitset(bits):
    """Count distinct d>0 such that a, a+d, a+2d all in the set, using bit operations.
    
    bits is a Python int where bit i (0-indexed) represents element i+1 in the set.
    """
    diffs = 0
    # For each possible difference d, check if (bits & (bits >> d) & (bits >> (2*d))) != 0
    # But we need to count distinct d values, so iterate over d
    # Max d: if a=1, a+2d <= 512, so d <= 255
    # Actually a can be as large as 512-2d, so d can be up to 255
    # But also a+d must be in set, so d can be up to 511 theoretically but constrained by a+2d <= 512
    # So d ranges from 1 to 255
    # For each d, check if there exists a such that bits has bits at positions a-1, a+d-1, a+2d-1
    # i.e., bits & (bits >> d) & (bits >> (2*d)) != 0
    # But wait: bit positions are 0-indexed for values 1..512
    # If value v is in set, bit (v-1) is set
    # a is in set: bit (a-1) set
    # a+d is in set: bit (a+d-1) set  
    # a+2d is in set: bit (a+2d-1) set
    # So we need bits & (bits >> d) & (bits >> (2*d)) to have some common bit set
    # That common bit would be at position (a-1), meaning a-1, a-1+d, a-1+2d are all set
    # which means values a, a+d, a+2d are all in set. Correct.
    
    for d in range(1, 256):
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            diffs += 1
    return diffs

def _make_bits(arr):
    """Convert a sorted list of integers (1..512) to a bitmask integer."""
    bits = 0
    for x in arr:
        bits |= (1 << (x - 1))
    return bits

def _bits_to_arr(bits):
    """Convert bitmask back to sorted list of integers."""
    arr = []
    for i in range(512):
        if (bits >> i) & 1:
            arr.append(i + 1)
    return arr

def _random_initial(rng):
    """Generate a random initial set of 32 distinct integers in 1..512."""
    return sorted(rng.sample(range(1, 513), 32))

def _dense_initial(rng):
    """Generate a consecutive block shifted randomly."""
    start = rng.randint(1, 481)
    return list(range(start, start + 32))

def _structured_initial(rng):
    """Generate a structured initial set with some arithmetic progression."""
    d = rng.randint(1, 15)
    max_start = 512 - 31 * d
    if max_start < 1:
        return _random_initial(rng)
    start = rng.randint(1, max_start)
    base = list(range(start, start + 32 * d, d))[:32]
    if len(base) < 32:
        remaining = [x for x in range(1, 513) if x not in base]
        if len(remaining) >= 32 - len(base):
            extra = rng.sample(remaining, 32 - len(base))
            base.extend(extra)
    return sorted(base)

def _sa(arr, start_time, deadline, rng):
    """Simulated annealing using bitset scoring."""
    n = len(arr)
    arr = list(arr)
    bits = _make_bits(arr)
    best_score = _score_bitset(bits)
    best_bits = bits
    cur_score = best_score
    cur_bits = bits
    
    T = 50.0
    T_min = 0.5
    alpha = 0.997
    steps_since_best = 0
    max_steps_since_best = 3000
    
    while True:
        if time.time() > deadline:
            break
        if T < T_min:
            break
        
        move_type = rng.random()
        if move_type < 0.6:
            # Single swap: remove one element, add a new random one
            # Pick a random element to remove
            # We need to pick from the 32 set bits
            # Find all set bit positions
            set_positions = [i for i in range(512) if (cur_bits >> i) & 1]
            if not set_positions:
                break
            idx = rng.randrange(len(set_positions))
            old_pos = set_positions[idx]
            old_val = old_pos + 1
            
            new_val = rng.randint(1, 512)
            if new_val == old_val:
                continue
            new_pos = new_val - 1
            if (cur_bits >> new_pos) & 1:
                continue
            
            trial_bits = cur_bits & ~(1 << old_pos) | (1 << new_pos)
            trial_score = _score_bitset(trial_bits)
            delta = trial_score - cur_score
        elif move_type < 0.85:
            # Double swap: remove two, add two
            set_positions = [i for i in range(512) if (cur_bits >> i) & 1]
            if len(set_positions) < 2:
                break
            i1, i2 = rng.sample(range(len(set_positions)), 2)
            old1_pos = set_positions[i1]
            old2_pos = set_positions[i2]
            old1_val = old1_pos + 1
            old2_val = old2_pos + 1
            
            new1_val, new2_val = rng.sample(range(1, 513), 2)
            new1_pos = new1_val - 1
            new2_pos = new2_val - 1
            if ((cur_bits >> new1_pos) & 1) or ((cur_bits >> new2_pos) & 1):
                continue
            if new1_pos == new2_pos:
                continue
            
            trial_bits = cur_bits & ~(1 << old1_pos) & ~(1 << old2_pos) | (1 << new1_pos) | (1 << new2_pos)
            trial_score = _score_bitset(trial_bits)
            delta = trial_score - cur_score
        else:
            # Local perturbation: shift one element by small amount
            set_positions = [i for i in range(512) if (cur_bits >> i) & 1]
            if not set_positions:
                break
            idx = rng.randrange(len(set_positions))
            old_pos = set_positions[idx]
            old_val = old_pos + 1
            shift = rng.choice([-5, -3, -2, -1, 1, 2, 3, 5, 7])
            new_val = old_val + shift
            if new_val < 1 or new_val > 512:
                continue
            new_pos = new_val - 1
            if (cur_bits >> new_pos) & 1:
                continue
            
            trial_bits = cur_bits & ~(1 << old_pos) | (1 << new_pos)
            trial_score = _score_bitset(trial_bits)
            delta = trial_score - cur_score
        
        # Accept or reject
        if delta >= 0:
            accept = True
        else:
            # Use exponential with base e
            # P = exp(delta / T)
            import math
            accept = rng.random() < math.exp(delta / T)
        
        if accept:
            cur_bits = trial_bits
            cur_score = trial_score
            if cur_score > best_score:
                best_score = cur_score
                best_bits = cur_bits
                steps_since_best = 0
            else:
                steps_since_best += 1
        else:
            steps_since_best += 1
        
        T *= alpha
        
        if steps_since_best > max_steps_since_best:
            break
    
    return best_bits, best_score

def _hill_climb(bits, start_time, deadline, rng):
    """Hill climbing polish using bitset scoring."""
    cur_bits = bits
    cur_score = _score_bitset(cur_bits)
    best_bits = cur_bits
    best_score = cur_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try swapping each of the 32 elements with every other possible value
        set_positions = [i for i in range(512) if (cur_bits >> i) & 1]
        for old_pos in set_positions:
            if time.time() > deadline:
                break
            old_val = old_pos + 1
            for new_val in range(1, 513):
                if new_val == old_val:
                    continue
                new_pos = new_val - 1
                if (cur_bits >> new_pos) & 1:
                    continue
                
                trial_bits = cur_bits & ~(1 << old_pos) | (1 << new_pos)
                trial_score = _score_bitset(trial_bits)
                if trial_score > cur_score:
                    cur_bits = trial_bits
                    cur_score = trial_score
                    if cur_score > best_score:
                        best_score = cur_score
                        best_bits = cur_bits
                    improved = True
                    break
    
    return best_bits, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Generate diverse initial solutions
    inits = []
    inits.append(_dense_initial(rng))
    inits.append(_structured_initial(rng))
    inits.append(_random_initial(rng))
    inits.append(_random_initial(rng))
    inits.append(_structured_initial(rng))
    inits.append(_random_initial(rng))
    inits.append(_dense_initial(rng))
    inits.append(_structured_initial(rng))
    
    best_overall_bits = None
    best_overall_score = 0
    
    for init in inits:
        if time.time() > deadline - 5.0:
            break
        bits, score = _sa(init, start_time, deadline, rng)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_bits = bits
    
    if best_overall_bits is None:
        # Fallback
        best_overall_bits = _make_bits(list(range(1, 33)))
        best_overall_score = _score_bitset(best_overall_bits)
    
    # Hill climbing polish
    if time.time() < deadline - 2.0:
        best_overall_bits, best_overall_score = _hill_climb(
            best_overall_bits, start_time, deadline - 1.0, rng
        )
    
    # Convert back to list
    result = _bits_to_arr(best_overall_bits)
    
    # Ensure validity: exactly 32 distinct integers in 1..512
    if len(result) != 32:
        # Fallback
        result = list(range(1, 33))
    
    return result
