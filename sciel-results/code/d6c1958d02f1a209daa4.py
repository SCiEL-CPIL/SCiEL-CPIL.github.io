"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
_MAX_SUM = 2 * N  # max possible sum is 512
_DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START

def _build_sum_mask(S):
    """Build an integer bitmask of all pairwise sums a+b for a<=b in S.
    Bit i is set if sum value i is used. Returns (mask, is_valid_sidon)."""
    mask = 0
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            bit = 1 << s
            if mask & bit:
                return mask, False  # not Sidon
            mask |= bit
    return mask, True

def _can_add_element(x, S, sum_mask):
    """Check if x can be added to S while preserving Sidon property.
    Returns True if x+x and x+a for all a in S are all unused and distinct."""
    # Check x+x
    bit_xx = 1 << (2 * x)
    if sum_mask & bit_xx:
        return False
    # Check x + a for each a in S, and ensure all new sums are distinct from each other
    new_sums = set()
    new_sums.add(2 * x)
    for a in S:
        s = x + a
        if s == 2 * x:
            continue  # already checked
        if s in new_sums:
            return False
        new_sums.add(s)
        bit = 1 << s
        if sum_mask & bit:
            return False
    return True

def _is_maximal_fast(S, sum_mask):
    """Check if S is inclusion-maximal Sidon in [1,N] using bitmask."""
    S_set = set(S)
    for x in range(1, N + 1):
        if x in S_set:
            continue
        if _can_add_element(x, S, sum_mask):
            return False  # can extend
    return True

def _sidon_and_maximal_fast(S):
    """Check if S is a valid Sidon set that is inclusion-maximal."""
    if not S:
        return False
    sum_mask, valid = _build_sum_mask(S)
    if not valid:
        return False
    return _is_maximal_fast(S, sum_mask)

def _greedy_sidon_fast(randomize=False, seed=None):
    """Build a Sidon set greedily using bitmask. Returns (S, sum_mask)."""
    if seed is not None:
        random.seed(seed)
    selected = []
    sum_mask = 0
    candidates = list(range(1, N + 1))
    if randomize:
        random.shuffle(candidates)
    for x in candidates:
        if _can_add_element(x, selected, sum_mask):
            selected.append(x)
            # Update mask
            sum_mask |= (1 << (2 * x))
            for a in selected[:-1]:
                sum_mask |= (1 << (x + a))
    selected.sort()
    return selected, sum_mask

def _try_minimize_by_removal_fast(S, sum_mask):
    """Try to remove elements while keeping Sidon + maximal. Returns (S, sum_mask) of minimal size found."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            if not candidate:
                continue
            new_mask, valid = _build_sum_mask(candidate)
            if valid and _is_maximal_fast(candidate, new_mask):
                S = candidate
                sum_mask = new_mask
                improved = True
                break
    return S, sum_mask

def _local_search(S, sum_mask, max_iters=500):
    """Local search: try single removals and swaps to minimize size while maintaining maximality."""
    S = list(S)
    n = len(S)
    best_size = n
    best_S = list(S)
    best_mask = sum_mask
    
    for it in range(max_iters):
        improved = False
        
        # Try removing each element
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            if not candidate:
                continue
            new_mask, valid = _build_sum_mask(candidate)
            if valid and _is_maximal_fast(candidate, new_mask):
                S = candidate
                sum_mask = new_mask
                improved = True
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = list(S)
                    best_mask = sum_mask
                break
        
        if not improved:
            # Try swaps: remove one, add another
            for i in range(len(S)):
                without_i = S[:i] + S[i+1:]
                if not without_i:
                    continue
                base_mask, valid_base = _build_sum_mask(without_i)
                if not valid_base:
                    continue
                # Try adding each missing element
                S_set = set(S)
                for x in range(1, N + 1):
                    if x in S_set:
                        continue
                    if _can_add_element(x, without_i, base_mask):
                        candidate = without_i + [x]
                        candidate.sort()
                        new_mask, valid = _build_sum_mask(candidate)
                        if valid and _is_maximal_fast(candidate, new_mask):
                            S = candidate
                            sum_mask = new_mask
                            improved = True
                            if len(S) < best_size:
                                best_size = len(S)
                                best_S = list(S)
                                best_mask = sum_mask
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    if best_size < len(S):
        return best_S, best_mask
    return S, sum_mask

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    # Phase 1: Try several random greedy constructions, then minimize
    for seed in range(5000):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 3:
            break
        S, sum_mask = _greedy_sidon_fast(randomize=True, seed=seed)
        if len(S) < best_size:
            # Try to minimize further
            S_min, mask_min = _try_minimize_by_removal_fast(S, sum_mask)
            if _sidon_and_maximal_fast(S_min) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min
            # Also try local search if we have time
            if elapsed < _DEADLINE - 10:
                S_ls, mask_ls = _local_search(S_min, mask_min, max_iters=200)
                if _sidon_and_maximal_fast(S_ls) and len(S_ls) < best_size:
                    best_size = len(S_ls)
                    best_set = S_ls
            if elapsed > _DEADLINE - 5:
                break
    
    # If nothing found, use deterministic greedy
    if best_set is None:
        S, sum_mask = _greedy_sidon_fast(randomize=False)
        S_min, mask_min = _try_minimize_by_removal_fast(S, sum_mask)
        if _sidon_and_maximal_fast(S_min):
            best_set = S_min
        else:
            best_set = S
    
    return best_set
# EVOLVE-BLOCK-END
