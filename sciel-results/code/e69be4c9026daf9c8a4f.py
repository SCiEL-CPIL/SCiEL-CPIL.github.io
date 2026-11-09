import time
import random
import sys

_N = 128
_LO = 128
_HI = 256
_RANGE = _HI - _LO + 1

def _get_counts(A_set):
    """
    Returns a list of representation counts for sums in [_LO, _HI].
    Returns None if A is empty.
    """
    if not A_set:
        return None
    A_sorted = sorted(A_set)
    n = len(A_sorted)
    counts = [0] * _RANGE
    
    for i in range(n):
        a = A_sorted[i]
        for j in range(i, n):
            b = A_sorted[j]
            s = a + b
            if _LO <= s <= _HI:
                counts[s - _LO] += 1
    return counts

def _is_valid(counts):
    """Check if all sums in the range are covered."""
    if counts is None:
        return False
    return all(c >= 1 for c in counts)

def _get_peak(counts):
    """Get the maximum representation count."""
    if counts is None:
        return float('inf')
    return max(counts)

def _local_search(A_init, time_limit, rng):
    """
    Perform local search starting from A_init.
    Returns the best set found and its peak.
    """
    A = set(A_init)
    counts = _get_counts(A)
    
    if not _is_valid(counts):
        # If initial is invalid, try to repair it quickly or discard
        # For simplicity in this draft, we assume initial sets are valid or we repair in a pre-step.
        # However, random sets are often invalid.
        # Let's add a simple repair phase: add random elements until valid.
        # But this might explode the size.
        # Better: Only start from valid sets.
        return None, float('inf')

    current_peak = _get_peak(counts)
    best_A = A.copy()
    best_peak = current_peak
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Generate moves
        in_A = list(A)
        not_in_A = [x for x in range(_N + 1) if x not in A]
        
        moves = []
        
        # 1. Swaps: Remove x, Add y
        # Limit number of swaps to keep it fast
        num_swaps = min(50, len(in_A) * len(not_in_A))
        if num_swaps > 0:
            for _ in range(num_swaps):
                x = in_A[rng.randint(0, len(in_A) - 1)]
                y = not_in_A[rng.randint(0, len(not_in_A) - 1)]
                moves.append(('swap', x, y))
        
        # 2. Adds: Add y
        # Only if size is not too large? Or just try.
        # Smaller sets are generally better for minimizing peak, but we need coverage.
        # If we add, we must be able to remove later.
        num_adds = min(10, len(not_in_A))
        if num_adds > 0:
            for _ in range(num_adds):
                y = not_in_A[rng.randint(0, len(not_in_A) - 1)]
                moves.append(('add', None, y))
        
        # 3. Removes: Remove x
        # Only if it keeps validity.
        num_rems = min(10, len(in_A))
        if num_rems > 0:
            for _ in range(num_rems):
                x = in_A[rng.randint(0, len(in_A) - 1)]
                moves.append(('remove', x, None))
        
        rng.shuffle(moves)
        
        for move in moves:
            if time.time() - start_time > time_limit:
                break
            
            new_A = A.copy()
            
            if move[0] == 'swap':
                _, x, y = move
                new_A.discard(x)
                new_A.add(y)
            elif move[0] == 'add':
                _, _, y = move
                new_A.add(y)
            elif move[0] == 'remove':
                _, x, _ = move
                new_A.discard(x)
            
            if not new_A:
                continue
                
            new_counts = _get_counts(new_A)
            if not _is_valid(new_counts):
                continue
                
            new_peak = _get_peak(new_counts)
            
            if new_peak < current_peak:
                A = new_A
                counts = new_counts
                current_peak = new_peak
                improved = True
                no_improve_count = 0
                
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = A.copy()
                
                # Restart move generation with new state to exploit local structure
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a random restart within the local search? 
                # Or just break to let outer loop handle diversity.
                # For now, break to save time for other starts.
                break
        
    return best_A, best_peak

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85  # 90s limit, 5s margin
    
    rng = random.Random(42)
    
    best_A_global = None
    best_peak_global = float('inf')
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Contiguous intervals
    for s in range(0, 50):
        # Need 2s <= 128 and 2(s+w-1) >= 256
        # w >= 129 - s
        w_min = max(1, 129 - s)
        w_max = _N - s + 1
        if w_min > w_max:
            continue
        # Try a few widths
        for w in range(w_min, min(w_min + 3, w_max + 1)):
            A = set(range(s, s + w))
            counts = _get_counts(A)
            if counts and _is_valid(counts):
                peak = _get_peak(counts)
                if peak < best_peak_global:
                    best_peak_global = peak
                    best_A_global = A.copy()
                candidates.append(A)
                
    # 2. Two-block sets: [0, a] U [b, 128]
    # Sums: [0, 2a], [2b, 256], [b, 128+a]
    # To cover [128, 256]:
    # High block must cover [256 - (something), 256].
    # If b is small, high block is large.
    # Let's just generate random valid two-block sets.
    for _ in range(50):
        a = rng.randint(0, 40)
        b = rng.randint(80, 128)
        A = set(range(0, a + 1)) | set(range(b, 129))
        counts = _get_counts(A)
        if counts and _is_valid(counts):
            peak = _get_peak(counts)
            if peak < best_peak_global:
                best_peak_global = peak
                best_A_global = A.copy()
            candidates.append(A)
            
    # 3. Random sets of varying sizes
    for size in range(20, 60, 5):
        for _ in range(10):
            A = set(rng.sample(range(0, 129), size))
            # Ensure 0 and 128 are likely included to help with extremes?
            # 0 helps with low sums, 128 with high sums.
            # Actually, to cover 128, we need pairs summing to 128.
            # To cover 256, we need (128, 128). So 128 MUST be in A.
            A.add(128)
            # To cover 128, if 128 is in A, we need 0 in A? Or other pairs.
            # If 128 is in A, 128+128=256 is covered.
            # 128 can be covered by (0,128), (1,127), etc.
            # Including 0 is helpful.
            A.add(0)
            
            counts = _get_counts(A)
            if counts and _is_valid(counts):
                peak = _get_peak(counts)
                if peak < best_peak_global:
                    best_peak_global = peak
                    best_A_global = A.copy()
                candidates.append(A)
    
    # If no valid candidates found, fallback (should not happen)
    if best_A_global is None:
        return list(range(64, 129))
        
    # Sort candidates by initial peak to prioritize good starts
    candidates.sort(key=lambda x: _get_peak(_get_counts(x)))
    
    # Run local search on top candidates
    # Time budget per candidate
    time_elapsed = time.time() - start_time
    remaining_time = DEADLINE - time_elapsed
    num_searches = min(len(candidates), 5) # Do at most 5 local searches
    if num_searches == 0:
        return sorted(best_A_global)
        
    time_per_search = remaining_time / num_searches
    
    for i in range(num_searches):
        if time.time() > DEADLINE:
            break
            
        A_init = candidates[i]
        # Ensure we don't spend too long on one
        t_limit = min(time_per_search, 10) # Cap individual search at 10s
        
        A_res, peak_res = _local_search(A_init, t_limit, rng)
        
        if A_res is not None and peak_res < best_peak_global:
            best_peak_global = peak_res
            best_A_global = A_res
            
    return sorted(best_A_global)
