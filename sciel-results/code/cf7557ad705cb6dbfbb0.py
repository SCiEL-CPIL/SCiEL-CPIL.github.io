"""Rewritten program using bitset-based scoring for fast 3-AP difference counting."""
import time
import random
import math

def _score_bitset(bits):
    """Count distinct d>0 such that a, a+d, a+2d all in the set, using bit operations."""
    diffs = 0
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
    """Simulated annealing using bitset scoring with fast set-position tracking."""
    bits = _make_bits(arr)
    best_score = _score_bitset(bits)
    best_bits = bits
    cur_score = best_score
    cur_bits = bits
    
    # Maintain list of set positions for fast random access
    set_positions = [i for i in range(512) if (cur_bits >> i) & 1]
    
    T = 50.0
    T_min = 0.5
    alpha = 0.997
    steps_since_best = 0
    max_steps_since_best = 2000
    
    while True:
        if time.time() > deadline:
            break
        if T < T_min:
            break
        
        move_type = rng.random()
        if move_type < 0.55:
            # Single swap
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
        elif move_type < 0.80:
            # Double swap
            if len(set_positions) < 2:
                break
            i1, i2 = rng.sample(range(len(set_positions)), 2)
            old1_pos = set_positions[i1]
            old2_pos = set_positions[i2]
            
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
            if not set_positions:
                break
            idx = rng.randrange(len(set_positions))
            old_pos = set_positions[idx]
            old_val = old_pos + 1
            shift = rng.choice([-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 9, 11])
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
            accept = rng.random() < math.exp(delta / T)
        
        if accept:
            cur_bits = trial_bits
            cur_score = trial_score
            # Update set_positions
            set_positions = [i for i in range(512) if (cur_bits >> i) & 1]
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
    """Hill climbing polish with targeted neighborhood search."""
    cur_bits = bits
    cur_score = _score_bitset(cur_bits)
    best_bits = cur_bits
    best_score = cur_score
    
    # Multiple passes with expanding neighborhood
    for pass_num in range(5):
        if time.time() > deadline:
            break
        set_positions = [i for i in range(512) if (cur_bits >> i) & 1]
        improved = False
        for old_pos in set_positions:
            if time.time() > deadline:
                break
            old_val = old_pos + 1
            # Try nearby values first (within radius that grows with pass)
            radius = 20 + pass_num * 40
            for offset in range(-radius, radius + 1):
                if offset == 0:
                    continue
                new_val = old_val + offset
                if new_val < 1 or new_val > 512:
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
                    # Rebuild set_positions after change
                    set_positions = [i for i in range(512) if (cur_bits >> i) & 1]
                    break
        if not improved:
            break
    
    # Final full sweep if time permits
    if time.time() < deadline - 5.0:
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
    inits.append(_random_initial(rng))
    inits.append(_structured_initial(rng))
    
    # Pre-score all initial solutions to prioritize best starts
    scored_inits = []
    for init in inits:
        b = _make_bits(init)
        s = _score_bitset(b)
        scored_inits.append((s, b, init))
    scored_inits.sort(key=lambda x: -x[0])
    
    best_overall_bits = None
    best_overall_score = 0
    
    # Allocate time: ~80% for SA, ~20% for hill climb
    sa_deadline = start_time + (deadline - start_time) * 0.75
    
    for s, b, init in scored_inits:
        if time.time() > sa_deadline:
            break
        bits, score = _sa(init, start_time, sa_deadline, rng)
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
