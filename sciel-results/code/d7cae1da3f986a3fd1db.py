"""Optimized program for small maximal Sidon sets in [1,256]."""
import time
import random
import sys

N = 256
_DEADLINE = 55.0  # safety margin before 60s kill


def _is_sidon(S):
    """Check if S is a Sidon set (all a+b for a<=b distinct)."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_unblocked(S, N):
    """
    Returns the set of x in [1, N] \ S that can be added to S while keeping it Sidon.
    If this set is empty, S is maximal.
    """
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    
    unblocked = []
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if adding x creates a collision
        # New sums: 2x, x+s for s in S
        # Collision if 2x in sums OR x+s1 = s2+s3 for s1,s2,s3 in S
        blocked = False
        if 2*x in sums:
            blocked = True
        else:
            for s in S:
                if x + s in sums:
                    blocked = True
                    break
        if not blocked:
            unblocked.append(x)
    return unblocked

def _is_maximal(S, N):
    """Check if S is inclusion-maximal Sidon in [1,N]."""
    if not S:
        return False
    if not _is_sidon(S):
        return False
    return len(_get_unblocked(S, N)) == 0

def _sidon_and_maximal(S, N):
    if not S:
        return False
    if not _is_sidon(S):
        return False
    if not _is_maximal(S, N):
        return False
    return True

def _random_sidon_set(k, seed):
    """Generate a random Sidon set of size k."""
    rng = random.Random(seed)
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    S = []
    sums = set()
    for x in candidates:
        if len(S) == k:
            break
        new_sums = [x + a for a in S] + [2 * x]
        if len(set(new_sums)) == len(new_sums) and not sums.intersection(set(new_sums)):
            S.append(x)
            sums.update(new_sums)
    if len(S) == k:
        S.sort()
        return S
    return None

def _local_search_maximal(S, N, deadline=None):
    """
    Local search to find a maximal Sidon set starting from S.
    Tries to swap elements to reduce the number of unblocked elements, 
    aiming for 0 unblocked (maximal).
    """
    S = list(S)
    if not _is_sidon(S):
        return S
    
    current_unblocked = _get_unblocked(S, N)
    best_unblocked_count = len(current_unblocked)
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        # Try swapping each element in S with an unblocked element
        for i in range(len(S)):
            s_old = S[i]
            # Try replacing s_old with x from unblocked
            for x in current_unblocked:
                if x == s_old:
                    continue
                # Construct new set
                S_new = S[:i] + [x] + S[i+1:]
                S_new.sort()
                if not _is_sidon(S_new):
                    continue
                # Check if this swap improves things
                new_unblocked = _get_unblocked(S_new, N)
                if len(new_unblocked) < best_unblocked_count:
                    S = S_new
                    best_unblocked_count = len(new_unblocked)
                    current_unblocked = new_unblocked
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1

    def elapsed():
        return time.time() - start_time

    # Target sizes for small maximal Sidon sets
    # Start with small sizes and increase if not found
    target_sizes = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    
    found_maximal = False
    
    # Phase 1: Random search for small maximal sets
    for k in target_sizes:
        if elapsed() > _DEADLINE - 10:
            break
        for seed in range(2000):
            if elapsed() > _DEADLINE - 5:
                break
            S = _random_sidon_set(k, seed)
            if S is None:
                continue
            if _is_maximal(S, N):
                best_set = S
                best_size = k
                found_maximal = True
                break
            # Try local search to fix it
            if elapsed() < _DEADLINE - 5:
                S_ls = _local_search_maximal(S, N, deadline=_DEADLINE - 5)
                if _is_maximal(S_ls, N) and len(S_ls) == k:
                    best_set = S_ls
                    best_size = k
                    found_maximal = True
                    break
        if found_maximal:
            break

    if not found_maximal:
        # Fallback: Greedy maximal set (large size)
        # This ensures we return *a* maximal set if no small one is found
        # The greedy set is maximal by construction
        candidates = list(range(1, N + 1))
        random.seed(42)
        random.shuffle(candidates)
        S = []
        sums = set()
        for x in candidates:
            new_sums = [x + a for a in S] + [2 * x]
            if len(set(new_sums)) == len(new_sums) and not sums.intersection(set(new_sums)):
                S.append(x)
                sums.update(new_sums)
        S.sort()
        best_set = S
        best_size = len(S)
        
        # Try to minimize this large set? 
        # As noted, simple removal doesn't work. 
        # But we can try to find a *different* small set in the remaining time.
        if elapsed() < _DEADLINE - 2:
             # Try random small sets again with different seeds
             for k in range(16, best_size):
                 if elapsed() > _DEADLINE - 2:
                     break
                 for seed in range(1000):
                     if elapsed() > _DEADLINE - 2:
                         break
                     S_try = _random_sidon_set(k, seed + 10000)
                     if S_try and _is_maximal(S_try, N):
                         best_set = S_try
                         best_size = k
                         break
                 if best_size == k:
                     break

    return best_set
