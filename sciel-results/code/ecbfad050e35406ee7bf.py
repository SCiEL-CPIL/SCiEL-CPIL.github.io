"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _arr_to_mask(arr):
    """Convert a list of 1..512 values to a Python int bitmask (bit i = value i+1)."""
    mask = 0
    for x in arr:
        mask |= (1 << x)
    return mask

def _mask_to_arr(mask):
    """Convert a Python int bitmask back to sorted list of values."""
    arr = []
    x = 1
    while x <= 512:
        if mask & (1 << x):
            arr.append(x)
        x += 1
    return arr

def _score_bitset(mask):
    """Count distinct d>0 such that exists a with a, a+d, a+2d all in set.
    
    Uses bitset: for each d, check if (mask & (mask >> d) & (mask >> (2*d))) != 0.
    This is O(512/d) per d, total O(512 * H_512) ~ O(512 * ln(512)) bit operations.
    """
    count = 0
    for d in range(1, 257):  # max d such that a+2d <= 512 for some a>=1 => d <= 255
        # Check if there exists a where bits a, a+d, a+2d are all set
        # (mask >> d) has bit a set iff mask has bit a+d set
        # (mask >> (2*d)) has bit a set iff mask has bit a+2d set
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _random_initial(rng):
    return sorted(rng.sample(range(1, 513), 32))

def _dense_initial(rng):
    start = rng.randint(1, 481)
    return list(range(start, start + 32))

def _structured_initial(rng):
    d = rng.randint(1, 15)
    max_start = 512 - 31 * d
    if max_start < 1:
        d = 1
        max_start = 481
    start = rng.randint(1, max_start)
    base = list(range(start, start + 32 * d, d))[:32]
    base_set = set(base)
    remaining = [x for x in range(1, 513) if x not in base_set]
    if len(base) < 32:
        extra = rng.sample(remaining, 32 - len(base))
        base.extend(extra)
    return sorted(base)

def _sa_bitset(init_arr, deadline, rng):
    """Simulated annealing using bitset representation for fast scoring."""
    n = 32
    cur_arr = list(init_arr)
    cur_mask = _arr_to_mask(cur_arr)
    cur_score = _score_bitset(cur_mask)
    best_arr = cur_arr[:]
    best_score = cur_score
    
    T = 30.0
    T_min = 0.1
    alpha = 0.9995
    steps_since_best = 0
    max_steps_since_best = 5000
    
    e = 2.718281828459045
    
    while True:
        if time.time() > deadline:
            break
        if T < T_min:
            break
        
        move_type = rng.random()
        if move_type < 0.55:
            # Single swap
            idx = rng.randrange(n)
            old = cur_arr[idx]
            new = rng.randint(1, 512)
            if new == old or (cur_mask >> new) & 1:
                continue
            # Update mask
            new_mask = cur_mask & ~(1 << old) | (1 << new)
            trial_score = _score_bitset(new_mask)
            delta = trial_score - cur_score
            if delta >= 0 or rng.random() < pow(e, delta / T):
                cur_mask = new_mask
                cur_arr[idx] = new
                cur_arr.sort()
                cur_score = trial_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_arr = cur_arr[:]
                    steps_since_best = 0
                else:
                    steps_since_best += 1
            else:
                steps_since_best += 1
        elif move_type < 0.80:
            # Double swap
            i1, i2 = rng.sample(range(n), 2)
            old1, old2 = cur_arr[i1], cur_arr[i2]
            new1, new2 = rng.sample(range(1, 513), 2)
            if new1 == new2:
                continue
            if (cur_mask >> new1) & 1 or (cur_mask >> new2) & 1:
                continue
            new_mask = cur_mask & ~(1 << old1) & ~(1 << old2) | (1 << new1) | (1 << new2)
            trial_score = _score_bitset(new_mask)
            delta = trial_score - cur_score
            if delta >= 0 or rng.random() < pow(e, delta / T):
                cur_mask = new_mask
                cur_arr[i1] = new1
                cur_arr[i2] = new2
                cur_arr.sort()
                cur_score = trial_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_arr = cur_arr[:]
                    steps_since_best = 0
                else:
                    steps_since_best += 1
            else:
                steps_since_best += 1
        else:
            # Local perturbation
            idx = rng.randrange(n)
            old = cur_arr[idx]
            shift = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 8, 10, 12, 16, 20])
            new = old + shift
            if new < 1 or new > 512 or (cur_mask >> new) & 1:
                continue
            new_mask = cur_mask & ~(1 << old) | (1 << new)
            trial_score = _score_bitset(new_mask)
            delta = trial_score - cur_score
            if delta >= 0 or rng.random() < pow(e, delta / T):
                cur_mask = new_mask
                cur_arr[idx] = new
                cur_arr.sort()
                cur_score = trial_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_arr = cur_arr[:]
                    steps_since_best = 0
                else:
                    steps_since_best += 1
            else:
                steps_since_best += 1
        
        T *= alpha
        
        if steps_since_best > max_steps_since_best:
            break
    
    return best_arr, best_score

def _hill_climb_bitset(arr, deadline, rng):
    """Exhaustive single-swap hill climbing with bitset scoring."""
    cur_arr = list(arr)
    cur_mask = _arr_to_mask(cur_arr)
    cur_score = _score_bitset(cur_mask)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(32):
            old = cur_arr[idx]
            # Try all possible replacements
            for new in range(1, 513):
                if new == old or (cur_mask >> new) & 1:
                    continue
                new_mask = cur_mask & ~(1 << old) | (1 << new)
                trial_score = _score_bitset(new_mask)
                if trial_score > cur_score:
                    cur_mask = new_mask
                    cur_arr[idx] = new
                    cur_arr.sort()
                    cur_score = trial_score
                    improved = True
                    break
            if improved:
                break
    return cur_arr, cur_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
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
    
    best_overall = None
    best_overall_score = 0
    
    for init in inits:
        if time.time() > deadline - 15.0:
            break
        arr, score = _sa_bitset(init, deadline, rng)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = arr
    
    if best_overall is None:
        best_overall = _dense_initial(rng)
        best_overall_score = _score_bitset(_arr_to_mask(best_overall))
    
    # Hill-climb polish
    if time.time() < deadline - 5.0:
        arr, score = _hill_climb_bitset(best_overall, deadline, rng)
        if score > best_overall_score:
            best_overall = arr
            best_overall_score = score
    
    # Ensure validity
    result = sorted(set(best_overall))
    if len(result) != 32:
        result = list(range(1, 33))
    result = [x for x in result if 1 <= x <= 512]
    if len(result) < 32:
        extra = [x for x in range(1, 513) if x not in result]
        result.extend(extra[:32 - len(result)])
        result = sorted(result[:32])
    
    return result
# EVOLVE-BLOCK-END
