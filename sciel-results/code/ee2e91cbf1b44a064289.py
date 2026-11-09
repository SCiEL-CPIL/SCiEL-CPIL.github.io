"""Reciprocal weight maximization for 3-AP-free subsets of [1,256]."""
import time
from fractions import Fraction

N = 256
DEADLINE_MARGIN = 10  # seconds to leave before the 90s kill

def _has_3ap_bitset(bits):
    """Check if bitset (int, bit i set means i+1 in set) contains a 3-AP.
    A 3-AP a,b,c with a<b<c and b-a=c-b=d means a=c-2d, b=c-d.
    For each c and d: check if bits has c-2d, c-d, c all set."""
    # Use a different approach: for each pair (a,c) with c-a even, check midpoint
    # Faster: iterate over possible d and check (x, x+d, x+2d)
    # Bit trick: for a given d, the set of x where x and x+d and x+2d are all in bits
    # (bits >> d) & (bits >> 2*d) & bits gives positions x where x, x+d, x+2d all set
    # But we need x >= 0 and x+2d <= N-1 (0-indexed)
    # Actually bits represents elements 1..N, so bit i (0-indexed) = element i+1
    # 3-AP: elements a, a+d, a+2d where a>=1, a+2d<=N
    # In 0-indexed bits: positions (a-1), (a+d-1), (a+2d-1)
    # So for d>=1: (bits >> d) & (bits >> 2*d) & bits != 0 means there's a 3-AP
    # But we must ensure a+2d <= N, i.e., a-1+2d <= N-1, i.e., a <= N-2d+1
    # The bit trick naturally handles this since bits only has N bits
    for d in range(1, N // 2 + 1):
        if (bits & (bits >> d) & (bits >> 2 * d)) != 0:
            return True
    return False

def _can_add(bits, x):
    """Check if adding element x (1-indexed) to set represented by bits keeps it 3-AP-free.
    Only need to check 3-APs involving x.
    x can be: first (x, x+d, x+2d), middle (x-d, x, x+d), last (x-2d, x-d, x).
    """
    pos = x - 1  # 0-indexed position in bits
    # x as first element: x, x+d, x+2d for d>=1, x+2d <= N
    for d in range(1, (N - x + 1) // 2 + 1):
        if (bits >> (x + d - 1)) & 1 and (bits >> (x + 2*d - 1)) & 1:
            return False
    # x as middle: x-d, x, x+d for d>=1, x-d >= 1
    for d in range(1, x):
        if (bits >> (x - d - 1)) & 1 and (bits >> (x + d - 1)) & 1:
            return False
    # x as last: x-2d, x-d, x for d>=1, x-2d >= 1
    for d in range(1, (x + 1) // 2):
        if (bits >> (x - 2*d - 1)) & 1 and (bits >> (x - d - 1)) & 1:
            return False
    return True

def _remove_and_check(bits, x, y):
    """Check if removing x and adding y (1-indexed) keeps 3-AP-free.
    Returns True if valid."""
    new_bits = (bits & ~(1 << (x - 1))) | (1 << (y - 1))
    return not _has_3ap_bitset(new_bits)

def _reciprocal_sum_fraction(bits):
    """Compute exact sum of reciprocals as Fraction."""
    total = Fraction(0, 1)
    for i in range(N):
        if (bits >> i) & 1:
            total += Fraction(1, i + 1)
    return total

def _reciprocal_sum_float(bits):
    """Compute floating sum of reciprocals."""
    total = 0.0
    for i in range(N):
        if (bits >> i) & 1:
            total += 1.0 / (i + 1)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Phase 1: Greedy construction prioritizing high reciprocal weight
    # Try multiple greedy strategies and keep the best
    best_bits = 0
    best_score = 0.0
    
    # Strategy 1: Greedy by 1/x descending (smallest x first)
    bits = 0
    for x in range(1, N + 1):
        if _can_add(bits, x):
            bits |= (1 << (x - 1))
    score = _reciprocal_sum_float(bits)
    if score > best_score:
        best_score = score
        best_bits = bits
    
    # Strategy 2: Greedy by 1/x but with lookahead - try to include small numbers
    # Actually, let's try a different greedy: process in order but skip if it blocks many small numbers
    # For now, let's do simulated annealing from the best greedy solution
    
    # Phase 2: Simulated annealing with swap moves
    if time.time() > deadline:
        return _bits_to_list(best_bits)
    
    import random
    rng = random.Random(42)
    
    current_bits = best_bits
    current_score = best_score
    best_bits_sa = best_bits
    best_score_sa = best_score
    
    # Get list of elements in current set
    def bits_to_list(b):
        return [i + 1 for i in range(N) if (b >> i) & 1]
    
    # SA parameters
    T_start = 0.01
    T_end = 0.0001
    alpha = 0.995  # cooling rate
    max_iter = 50000
    
    current_T = T_start
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate a swap move: remove one element, add another
        cur_list = bits_to_list(current_bits)
        if len(cur_list) < 2:
            break
        
        # Remove a random element
        idx_remove = rng.randint(0, len(cur_list) - 1)
        x_remove = cur_list[idx_remove]
        
        # Try to add a random element not in set
        # Prefer elements with high 1/x (small x) to increase reciprocal sum
        # But also try random
        if rng.random() < 0.7:
            # Pick from small numbers not in set
            candidates = [x for x in range(1, N + 1) if not ((current_bits >> (x - 1)) & 1)]
            if not candidates:
                continue
            # Weight by 1/x
            weights = [1.0 / x for x in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            x_add = candidates[-1]
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    x_add = c
                    break
        else:
            candidates = [x for x in range(1, N + 1) if not ((current_bits >> (x - 1)) & 1)]
            if not candidates:
                continue
            x_add = candidates[rng.randint(0, len(candidates) - 1)]
        
        if x_remove == x_add:
            continue
        
        # Check if swap is valid
        new_bits = (current_bits & ~(1 << (x_remove - 1))) | (1 << (x_add - 1))
        if _has_3ap_bitset(new_bits):
            continue
        
        new_score = _reciprocal_sum_float(new_bits)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < pow(2.718281828, delta / current_T):
            current_bits = new_bits
            current_score = new_score
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_bits_sa = current_bits
        
        current_T *= alpha
        if current_T < T_end:
            break
    
    if best_score_sa > best_score:
        best_bits = best_bits_sa
        best_score = best_score_sa
    
    # Phase 3: Local search - try all single swaps to find improvements
    if time.time() > deadline:
        return _bits_to_list(best_bits)
    
    improved = True
    while improved:
        improved = False
        cur_list = bits_to_list(best_bits)
        for x_remove in cur_list:
            if time.time() > deadline:
                break
            for x_add in range(1, N + 1):
                if x_add == x_remove or ((best_bits >> (x_add - 1)) & 1):
                    continue
                new_bits = (best_bits & ~(1 << (x_remove - 1))) | (1 << (x_add - 1))
                if _has_3ap_bitset(new_bits):
                    continue
                new_score = _reciprocal_sum_float(new_bits)
                if new_score > best_score:
                    best_bits = new_bits
                    best_score = new_score
                    improved = True
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return _bits_to_list(best_bits)

def _bits_to_list(bits):
    return [i + 1 for i in range(N) if (bits >> i) & 1]
