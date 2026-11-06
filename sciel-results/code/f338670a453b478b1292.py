"""Optimized construction for 32 integers in 1..512 maximizing distinct 3-AP differences."""
import time
import random
import sys

# EVOLVE-BLOCK-START

# Precompute difference masks: for each d, a 512-bit integer where bit i is set
# iff a=i+1, a+d, a+2d are all in [1,512]
# We use 0-indexed positions: position p corresponds to value p+1
# For difference d, starting position p (0-indexed) works if p+2d < 512
# diff_mask[d] has bits set at positions p where p, p+d, p+2d are all valid

N = 512

def _build_diff_masks():
    """Precompute bitmask for each difference d: positions where AP of diff d fits."""
    masks = {}
    for d in range(1, N // 2 + 1):  # d up to 255 (since a+2d <= 512 means d <= 255)
        mask = 0
        for p in range(0, N - 2 * d + 1):
            # p, p+d, p+2d all < N (0-indexed)
            mask |= (1 << p)
        masks[d] = mask
    return masks

DIFF_MASKS = _build_diff_masks()
MAX_D = max(DIFF_MASKS.keys())

def _score_mask(mask):
    """Count distinct d > 0 such that some a, a+d, a+2d all in set (mask representation)."""
    count = 0
    for d, dm in DIFF_MASKS.items():
        # Check if there exists p where bits p, p+d, p+2d are all set in mask
        # mask & dm gives positions p where p is in set and p is a valid start
        # (mask >> d) & dm gives positions p where p+d is in set and p is valid start
        # (mask >> (2*d)) & dm gives positions p where p+2d is in set and p is valid start
        # Intersection non-zero means all three are in set
        if (mask & dm & (mask >> d) & (mask >> (2 * d))):
            count += 1
    return count

def _score_mask_fast(mask):
    """Same as _score_mask but with early exit if we already have all diffs."""
    count = 0
    for d, dm in DIFF_MASKS.items():
        if (mask & dm & (mask >> d) & (mask >> (2 * d))):
            count += 1
    return count

def _mask_to_list(mask):
    """Convert bitmask to list of values (1-indexed)."""
    result = []
    for p in range(N):
        if mask & (1 << p):
            result.append(p + 1)
    return result

def _list_to_mask(lst):
    """Convert list of values (1-indexed) to bitmask."""
    mask = 0
    for v in lst:
        mask |= (1 << (v - 1))
    return mask

def _greedy_init():
    """Greedy initialization: add elements one by one maximizing marginal score gain."""
    mask = 0
    used = set()
    for _ in range(32):
        best_val = None
        best_score = -1
        for val in range(1, N + 1):
            if val in used:
                continue
            new_mask = mask | (1 << (val - 1))
            sc = _score_mask(new_mask)
            if sc > best_score:
                best_score = sc
                best_val = val
        if best_val is not None:
            mask |= (1 << (best_val - 1))
            used.add(best_val)
        else:
            # Fallback
            for val in range(1, N + 1):
                if val not in used:
                    mask |= (1 << (val - 1))
                    used.add(val)
                    break
    return mask

def _single_swap_mask(mask, rng):
    """Remove one element, add one new element."""
    # Find set bits
    bits = []
    m = mask
    while m:
        lb = m & (-m)
        bits.append(lb)
        m ^= lb
    
    if not bits:
        return None
    
    # Remove one random element
    removed = rng.choice(bits)
    new_mask = mask ^ removed  # XOR to remove
    
    # Find available positions
    available = []
    for p in range(N):
        if not (new_mask & (1 << p)):
            available.append(1 << p)
    
    if not available:
        return None
    
    added = rng.choice(available)
    return new_mask | added

def _double_swap_mask(mask, rng):
    """Remove two elements, add two new elements."""
    bits = []
    m = mask
    while m:
        lb = m & (-m)
        bits.append(lb)
        m ^= lb
    
    if len(bits) < 2:
        return None
    
    removed = rng.sample(bits, 2)
    new_mask = mask
    for r in removed:
        new_mask ^= r
    
    available = []
    for p in range(N):
        if not (new_mask & (1 << p)):
            available.append(1 << p)
    
    if len(available) < 2:
        return None
    
    added = rng.sample(available, 2)
    for a in added:
        new_mask |= a
    
    return new_mask

def _triple_swap_mask(mask, rng):
    """Remove three elements, add three new elements."""
    bits = []
    m = mask
    while m:
        lb = m & (-m)
        bits.append(lb)
        m ^= lb
    
    if len(bits) < 3:
        return None
    
    removed = rng.sample(bits, 3)
    new_mask = mask
    for r in removed:
        new_mask ^= r
    
    available = []
    for p in range(N):
        if not (new_mask & (1 << p)):
            available.append(1 << p)
    
    if len(available) < 3:
        return None
    
    added = rng.sample(available, 3)
    for a in added:
        new_mask |= a
    
    return new_mask

def _local_refine(mask, deadline):
    """Try all single swaps to improve the mask. Returns improved mask."""
    best_mask = mask
    best_score = _score_mask(mask)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        bits = []
        m = mask
        while m:
            lb = m & (-m)
            bits.append(lb)
            m ^= lb
        
        for rem_bit in bits:
            if time.time() > deadline:
                break
            trial_mask = mask ^ rem_bit
            for p in range(N):
                if trial_mask & (1 << p):
                    continue
                candidate = trial_mask | (1 << p)
                sc = _score_mask(candidate)
                if sc > best_score:
                    best_score = sc
                    best_mask = candidate
                    mask = candidate
                    improved = True
                    break
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    best_mask = None
    best_score = -1
    
    # Multiple initialization strategies
    inits = []
    
    # Strategy 1: Greedy
    inits.append(_greedy_init())
    
    # Strategy 2: Dense small range
    mask2 = 0
    for v in range(1, 33):
        mask2 |= (1 << (v - 1))
    inits.append(mask2)
    
    # Strategy 3: Spread out
    mask3 = 0
    for i in range(32):
        val = 1 + i * 16
        mask3 |= (1 << (val - 1))
    inits.append(mask3)
    
    # Strategy 4: Random biased small
    mask4 = 0
    used = set()
    for _ in range(32):
        val = 1 + int(rng.random() ** 1.5 * 511)
        while val in used:
            val = 1 + int(rng.random() ** 1.5 * 511)
        used.add(val)
        mask4 |= (1 << (val - 1))
    inits.append(mask4)
    
    # Strategy 5: Fully random
    mask5 = 0
    for v in rng.sample(range(1, N + 1), 32):
        mask5 |= (1 << (v - 1))
    inits.append(mask5)
    
    # Strategy 6: Structured - multiples pattern
    mask6 = 0
    used = set()
    for mult in range(1, 9):
        base = mult * 10
        for k in range(4):
            val = base + k * mult + rng.randint(-2, 2)
            val = max(1, min(512, val))
            if val not in used:
                used.add(val)
                mask6 |= (1 << (val - 1))
    while len(used) < 32:
        v = rng.randint(1, 512)
        if v not in used:
            used.add(v)
            mask6 |= (1 << (v - 1))
    inits.append(mask6)
    
    # Strategy 7: Two-cluster
    mask7 = 0
    used = set()
    for v in rng.sample(range(1, 129), 16):
        used.add(v)
        mask7 |= (1 << (v - 1))
    for v in rng.sample(range(385, 513), 16):
        if v not in used:
            used.add(v)
            mask7 |= (1 << (v - 1))
    while len(used) < 32:
        v = rng.randint(1, 512)
        if v not in used:
            used.add(v)
            mask7 |= (1 << (v - 1))
    inits.append(mask7)
    
    # Evaluate all initializations
    for init_mask in inits:
        sc = _score_mask(init_mask)
        if sc > best_score:
            best_score = sc
            best_mask = init_mask
    
    # Simulated annealing from each good initialization
    for init_idx, init_mask in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        current_mask = init_mask
        current_score = _score_mask(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
        
        temp = 8.0
        cooling = 0.9995
        min_temp = 0.01
        steps = 0
        
        while temp > min_temp and time.time() < deadline - 5:
            move_type = rng.random()
            if move_type < 0.35:
                new_mask = _triple_swap_mask(current_mask, rng)
            elif move_type < 0.65:
                new_mask = _double_swap_mask(current_mask, rng)
            else:
                new_mask = _single_swap_mask(current_mask, rng)
            
            if new_mask is None:
                temp *= cooling
                continue
            
            new_score = _score_mask(new_mask)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < pow(2.718281828, delta / temp):
                current_mask = new_mask
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            
            temp *= cooling
            steps += 1
    
    # Final local refinement
    if best_mask is not None and time.time() < deadline - 3:
        best_mask = _local_refine(best_mask, deadline - 1)
        best_score = _score_mask(best_mask)
    
    if best_mask is None:
        return list(range(1, 33))
    
    return _mask_to_list(best_mask)

# EVOLVE-BLOCK-END
