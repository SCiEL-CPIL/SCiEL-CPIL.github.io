import random
import time

def _compute_diff_set_bitset(S_bits):
    """
    Given a bitset (Python int) where bit i is set if (i+1) is in the set,
    compute the set of differences d>0 such that there exists a with
    a, a+d, a+2d all in the set. Returns a frozenset of such d values.
    """
    diff_bits = 0
    # For each possible difference d, check if there exists a such that
    # bits at positions (a-1), (a+d-1), (a+2d-1) are all set.
    # We iterate over d from 1 to 255 (max d where a+2d <= 512, a>=1 => 2d <= 511)
    for d in range(1, 256):
        # We need: exists a in [1, 512-2d] such that
        # bit (a-1) AND bit (a+d-1) AND bit (a+2d-1) are all 1
        # Let mask_d be a bitmask where bit i is set if bits i, i+d, i+2d are all set
        # We can compute this by shifting:
        # S & (S >> d) & (S >> 2*d) - if any bit is set, we have a valid a
        # But we need to be careful about the range
        shifted = S_bits & (S_bits >> d) & (S_bits >> 2*d)
        if shifted != 0:
            diff_bits |= (1 << d)
    return diff_bits

def _score_from_diff_bits(diff_bits):
    """Count number of set bits in diff_bits (i.e., distinct differences)."""
    return bin(diff_bits).count('1')

def _dense_seeds(n=32):
    """Return n integers from [1,512] that participate in many 3-APs."""
    counts = []
    for x in range(1, 513):
        c_mid = min(x - 1, 512 - x)
        c_first = (512 - x) // 2
        c_last = (x - 1) // 2
        total = c_mid + c_first + c_last
        counts.append((total, x))
    counts.sort(reverse=True)
    return [x for _, x in counts[:n]]

def _build_initial_set():
    """Build an initial 32-element set using greedy selection based on dense seeds."""
    seeds = _dense_seeds(32)
    # Start with top seeds and greedily add to maximize distinct differences
    S = set()
    for x in seeds:
        S.add(x)
    return S

def construct():
    random.seed(42)
    start_time = time.time()
    time_limit = 115  # seconds, leave buffer for 120s limit
    
    # Build initial set
    current = _build_initial_set()
    
    # Convert to bitset: bit (x-1) is set if x is in the set
    current_bits = 0
    for x in current:
        current_bits |= (1 << (x - 1))
    
    current_diff_bits = _compute_diff_set_bitset(current_bits)
    current_score = _score_from_diff_bits(current_diff_bits)
    
    best = set(current)
    best_bits = current_bits
    best_diff_bits = current_diff_bits
    best_score = current_score
    
    all_nums = list(range(1, 513))
    
    temp = 3.0
    cooling = 0.9999
    
    # Precompute for each x, the list of (d, a) pairs that x can form
    # Actually, let's just use the bitset approach for scoring
    
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Pick a random element to remove from current
        to_remove = random.choice(list(current))
        
        # Pick a random element to add (not already in current)
        # Use a simple approach: pick random from 1..512, if in current, try again
        to_add = random.randint(1, 512)
        attempts = 0
        while to_add in current and attempts < 10:
            to_add = random.randint(1, 512)
            attempts += 1
        if to_add in current:
            continue
        
        # Compute new bitset
        new_bits = current_bits
        new_bits &= ~(1 << (to_remove - 1))  # remove to_remove
        new_bits |= (1 << (to_add - 1))       # add to_add
        
        # Compute new diff set
        new_diff_bits = _compute_diff_set_bitset(new_bits)
        new_score = _score_from_diff_bits(new_diff_bits)
        
        delta = new_score - current_score
        
        # Accept if better or with probability exp(delta/temp)
        if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
            current = set(current)
            current.discard(to_remove)
            current.add(to_add)
            current_bits = new_bits
            current_diff_bits = new_diff_bits
            current_score = new_score
            
            if current_score > best_score:
                best = set(current)
                best_bits = current_bits
                best_diff_bits = current_diff_bits
                best_score = current_score
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return sorted(best)
