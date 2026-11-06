import time
import random
import numpy as np

def _score_bitmask(mask):
    """
    Fast score using a 512-bit integer bitmask.
    mask: int with bit i set iff (i+1) is in the set (i ranges 0..511).
    Returns number of distinct d>0 such that a, a+d, a+2d all in set.
    """
    count = 0
    # d can range from 1 to 255 (since 2*255 = 510 < 512)
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _set_to_mask(s):
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    return m

def _mask_to_set(mask):
    result = []
    temp = mask
    idx = 0
    while temp:
        if temp & 1:
            result.append(idx + 1)
        temp >>= 1
        idx += 1
    return result

def _get_set_bits(mask):
    bits = []
    temp = mask
    idx = 0
    while temp:
        if temp & 1:
            bits.append(idx)
        temp >>= 1
        idx += 1
    return bits

def _get_missing_d(mask):
    """Return a list of d values that are NOT covered."""
    missing = []
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) == 0:
            missing.append(d)
    return missing

def _find_gaps_for_d(mask, d):
    """
    Find all y such that adding y would create a 3-AP with difference d.
    y is a gap if a, a+2d are in mask, but y=a+d is not.
    Returns list of indices (0-based) for y.
    """
    gaps = []
    # a can range such that a+2d < 512
    # a in 0..512-2d-1
    for a in range(0, 512 - 2 * d):
        if (mask & (1 << a)) and (mask & (1 << (a + 2 * d))):
            y_idx = a + d
            if not (mask & (1 << y_idx)):
                gaps.append(y_idx)
    return gaps

def _is_removal_safe(mask, idx):
    """
    Check if removing 'idx' from the set results in the same score.
    This is expensive to check exactly by re-scoring, so we use a heuristic:
    If the element is part of a triple for a d, and that d has no other triples, it's unsafe.
    However, a faster proxy is: just try the removal and score? No, that's the cost of the move.
    For move generation, we just want to bias towards 'safe' removals.
    We can approximate safety by checking if the element is in few APs.
    """
    # This function is actually not used in the main loop to save time, 
    # but we can use it in the polish phase or to filter candidates.
    # For now, we rely on the SA acceptance criterion.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    rng = random.Random(42)
    
    best_mask = 0
    best_score = -1

    # Initial diverse starts
    # 1. Random
    s = list(rng.sample(range(1, 513), 32))
    m = _set_to_mask(s)
    sc = _score_bitmask(m)
    if sc > best_score:
        best_score = sc
        best_mask = m

    # 2. Structured: Union of APs
    # Try to construct sets that naturally have many APs
    # Example: {1, 2, 3, 4, 5, 6, 7, 8} has d=1,2,3,4
    # {1, 3, 5, 7, 9, 11, 13, 15} has d=2,4,6,8
    # Combining these might yield many d.
    
    # Strategy: Start with a dense block and spread out.
    # Let's try a few structured inits
    structured_starts = []
    
    # Init A: Two dense blocks
    block1 = list(range(1, 17)) # 1..16
    block2 = list(range(300, 316)) # 300..315
    s = block1 + block2
    if len(set(s)) == 32:
        m = _set_to_mask(s)
        sc = _score_bitmask(m)
        if sc > best_score:
            best_score = sc
            best_mask = m

    # Init B: Interleaved APs
    s = set()
    for i in range(16):
        s.add(1 + i * 30)
    for i in range(16):
        s.add(15 + i * 30)
    if len(s) == 32:
        m = _set_to_mask(list(s))
        sc = _score_bitmask(m)
        if sc > best_score:
            best_score = sc
            best_mask = m

    # Multi-restart Simulated Annealing
    num_restarts = 12
    restart_budget = 10.0  # seconds per restart
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Start from best or perturbation
        if restart == 0:
            current_mask = best_mask
            current_score = best_score
        else:
            current_mask = best_mask
            # Perturb: remove 3-5 random elements, add new ones
            num_perturb = rng.randint(3, 5)
            for _ in range(num_perturb):
                bits_set = _get_set_bits(current_mask)
                if not bits_set: break
                remove_idx = rng.choice(bits_set)
                current_mask &= ~(1 << remove_idx)
                while True:
                    add_idx = rng.randint(0, 511)
                    if not (current_mask & (1 << add_idx)):
                        current_mask |= (1 << add_idx)
                        break
            current_score = _score_bitmask(current_mask)
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask

        # SA Parameters
        T = 1.5
        T_min = 0.05
        cooling = 0.9995
        restart_deadline = start_time + (time.time() - start_time) + restart_budget
        if restart_deadline > deadline:
            restart_deadline = deadline

        while time.time() < restart_deadline and T > T_min:
            # Generate Move
            # 80% Directed, 20% Random
            if rng.random() < 0.8:
                missing_d = _get_missing_d(current_mask)
                if not missing_d:
                    # All d covered? Max score 255. Unlikely for 32 elements.
                    # But if so, we are done.
                    break 
                
                # Pick a random missing d, weighted towards small d? 
                # Small d are easier to fill?
                target_d = rng.choice(missing_d)
                gaps = _find_gaps_for_d(current_mask, target_d)
                
                if gaps:
                    add_idx = rng.choice(gaps)
                    # Choose remove_idx: prefer elements that are "redundant"
                    # Heuristic: remove an element that is far from the new add_idx? 
                    # Or just random.
                    bits_set = _get_set_bits(current_mask)
                    remove_idx = rng.choice(bits_set)
                else:
                    # No gaps for this d, fall back to random or try another d
                    # Try another d if possible
                    if len(missing_d) > 1:
                        target_d = rng.choice([d for d in missing_d if d != target_d])
                        gaps = _find_gaps_for_d(current_mask, target_d)
                        if gaps:
                            add_idx = rng.choice(gaps)
                            bits_set = _get_set_bits(current_mask)
                            remove_idx = rng.choice(bits_set)
                        else:
                            add_idx = rng.randint(0, 511)
                            bits_set = _get_set_bits(current_mask)
                            remove_idx = rng.choice(bits_set)
                    else:
                        add_idx = rng.randint(0, 511)
                        bits_set = _get_set_bits(current_mask)
                        remove_idx = rng.choice(bits_set)
            else:
                add_idx = rng.randint(0, 511)
                bits_set = _get_set_bits(current_mask)
                remove_idx = rng.choice(bits_set)

            if add_idx == remove_idx:
                continue
            if current_mask & (1 << add_idx):
                continue

            new_mask = current_mask
            new_mask &= ~(1 << remove_idx)
            new_mask |= (1 << add_idx)

            new_score = _score_bitmask(new_mask)
            delta_score = new_score - current_score

            # Acceptance: Always accept improvements. Accept worsening with prob exp(-delta/T)
            if delta_score > 0 or (T > T_min and rng.random() < np.exp(delta_score / T)):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask

            T *= cooling

    # Greedy Polish
    # Try single swaps to improve. 
    # This is a local search. We repeat until no improvement or time runs out.
    improved = True
    polish_deadline = deadline
    while improved and time.time() < polish_deadline:
        improved = False
        bits_set = _get_set_bits(best_mask)
        # Shuffle to avoid bias in the order of checking
        rng.shuffle(bits_set)
        
        for remove_idx in bits_set:
            if time.time() > polish_deadline:
                break
            
            # Try all possible add_idx
            # Optimization: We can break early if we find an improvement, 
            # but we must update best_mask and restart the outer loop to be safe.
            for add_idx in range(512):
                if add_idx == remove_idx:
                    continue
                if best_mask & (1 << add_idx):
                    continue
                
                new_mask = best_mask
                new_mask &= ~(1 << remove_idx)
                new_mask |= (1 << add_idx)
                
                new_score = _score_bitmask(new_mask)
                if new_score > best_score:
                    best_mask = new_mask
                    best_score = new_score
                    improved = True
                    break # Break inner loop, restart outer loop with new best
            
            if improved:
                break # Restart the outer loop with new best_mask

    result = _mask_to_set(best_mask)
    
    # Ensure exactly 32 elements
    if len(result) != 32:
        remaining = [v for v in range(1, 513) if v not in set(result)]
        result.extend(rng.sample(remaining, 32 - len(result)))
        result = result[:32]

    return sorted(result)
