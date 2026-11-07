import time
import random
from itertools import combinations

def _check_validity(subset_sums_set):
    """Check that no two distinct values in the set divide each other."""
    sums = sorted(subset_sums_set)
    n = len(sums)
    for i in range(n):
        a = sums[i]
        for j in range(i + 1, n):
            b = sums[j]
            if b % a == 0:
                return False
    return True

def _build_subset_sums(elements):
    """Build the set of all nonempty subset sums."""
    sums = set()
    n = len(elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _is_valid_set(elements):
    """Check if a set of elements is valid (all subset sums pairwise non-dividing)."""
    sums = _build_subset_sums(elements)
    return _check_validity(sums)

def _greedy_build(starting_elements, pool, target_size, deadline):
    """Greedily add elements from pool to starting_elements, maintaining validity."""
    current = list(starting_elements)
    current_sums = _build_subset_sums(current)
    
    for _ in range(target_size - len(current)):
        if time.time() > deadline:
            break
        added = False
        for x in pool:
            if x in current:
                continue
            # Try adding x
            new_elements = current + [x]
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + x)
            new_sums.add(x)
            
            # Check for duplicates (shouldn't happen if elements are distinct and positive,
            # but let's be safe - actually if x > 0 and all elements positive, sums are unique
            # per subset, but different subsets could give same sum)
            if len(new_sums) < (1 << len(new_elements)) - 1:
                # Duplicate sums detected
                continue
            
            if _check_validity(new_sums):
                current = new_elements
                current_sums = new_sums
                added = True
                break
        
        if not added:
            break
    
    return current

def _random_search(deadline, max_attempts=500):
    """Randomized search for large valid sets."""
    best = []
    best_len = 0
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Randomly select a starting pool
        # Use larger numbers to reduce chance of small subset sums dividing others
        base = random.randint(50, 200)
        spread = random.randint(10, 50)
        
        # Generate candidate pool
        candidates = set()
        for i in range(random.randint(15, 30)):
            val = base + i * spread + random.randint(-spread // 2, spread // 2)
            if 1 <= val <= 256:
                candidates.add(val)
        
        if len(candidates) < 5:
            continue
        
        pool = sorted(candidates)
        
        # Try greedy build from a random starting subset
        start_size = random.randint(2, min(5, len(pool)))
        start = random.sample(pool, start_size)
        
        # Quick validity check on start
        if not _is_valid_set(start):
            continue
        
        result = _greedy_build(start, pool, 30, deadline)
        
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    return best

def _structured_search(deadline):
    """Structured search using arithmetic progressions and other patterns."""
    best = []
    best_len = 0
    
    # Try various structured families
    for base in range(10, 200, 5):
        if time.time() > deadline:
            break
        for step in range(3, 50, 2):
            if time.time() > deadline:
                break
            # Arithmetic progression
            ap = []
            val = base
            while val <= 256 and len(ap) < 30:
                ap.append(val)
                val += step
            
            if len(ap) < 3:
                continue
            
            # Try to extend the AP with other elements
            pool = set(range(1, 257)) - set(ap)
            result = _greedy_build(ap, sorted(pool), 30, deadline)
            
            if len(result) > best_len:
                best = result
                best_len = len(result)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10s safety margin
    
    # Start with the known valid set
    current_best = [31, 32, 34]
    best_len = len(current_best)
    
    # Try structured search first (deterministic, may find good patterns)
    structured = _structured_search(deadline)
    if len(structured) > best_len:
        current_best = structured
        best_len = len(current_best)
    
    # Try randomized search
    if time.time() < deadline - 5:
        random.seed(42)
        random_result = _random_search(deadline)
        if len(random_result) > best_len:
            current_best = random_result
            best_len = len(current_best)
    
    # Ensure we return a valid list of Python ints
    return list(current_best)
