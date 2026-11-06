"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time

N = 256

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a Sidon set (all unordered sums distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal: adding any missing element breaks Sidon property."""
    selected_set = set(selected)
    # Precompute all sums
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            sums.add(selected[i] + selected[j])
    
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if adding x would break Sidon property
        # New sums involving x: x+x, x+a for a in selected
        conflict = False
        if 2 * x in sums:
            conflict = True
        else:
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
        if not conflict:
            return False
    return True

def _greedy_max_blocking(N, deadline):
    """
    Greedy Max-Blocking: iteratively select the element that blocks the most
    future candidates (i.e., has the most conflicts with unselected elements).
    This tends to produce smaller maximal sets.
    """
    start_time = time.time()
    selected = []
    selected_set = set()
    sums = set()
    remaining = set(range(1, N + 1))
    
    while remaining:
        if time.time() > deadline:
            break
        
        # For each candidate in remaining, compute how many other remaining elements it would block
        # If we pick x, it blocks y if x+y or 2x conflicts with existing sums, OR if x+y would create
        # a sum that conflicts with another future sum. More precisely, picking x removes from remaining
        # all y such that {x, y} cannot both be in the Sidon set given current sums.
        
        best_candidate = None
        best_block_count = -1
        
        for x in sorted(remaining):
            # Check if x can be added to current selected
            added_sums = [2 * x] + [x + a for a in selected]
            if len(set(added_sums)) != len(added_sums):
                continue
            if sums.intersection(added_sums):
                continue
            
            # Count how many remaining elements y (y != x) would be blocked by adding x
            block_count = 0
            for y in remaining:
                if y == x:
                    continue
                # Check if y could still be added after x is selected
                # New sums would include x+y, x+x, and all existing sums
                # y is blocked if 2y or y+a (for a in selected ∪ {x}) is in sums ∪ {2x, x+a for a in selected}
                new_sums_if_x = sums | set(added_sums)
                y_added = [2 * y] + [y + a for a in selected] + [x + y]
                if len(set(y_added)) != len(y_added):
                    block_count += 1
                    continue
                if new_sums_if_x.intersection(y_added):
                    block_count += 1
            
            if block_count > best_block_count:
                best_block_count = block_count
                best_candidate = x
        
        if best_candidate is None:
            break
        
        # Add best_candidate
        x = best_candidate
        selected.append(x)
        selected_set.add(x)
        remaining.remove(x)
        added_sums = [2 * x] + [x + a for a in selected[:-1]]
        sums.update(added_sums)
    
    return selected

def _greedy_min_cardinality(N, deadline):
    """
    Alternative greedy: try to build a small maximal set by selecting elements
    that are 'expensive' to add (block many others) but also ensure maximality.
    """
    start_time = time.time()
    selected = []
    selected_set = set()
    sums = set()
    remaining = set(range(1, N + 1))
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Find all candidates that can be added
        valid_candidates = []
        for x in remaining:
            added_sums = [2 * x] + [x + a for a in selected]
            if len(set(added_sums)) != len(added_sums):
                continue
            if sums.intersection(added_sums):
                continue
            valid_candidates.append(x)
        
        if not valid_candidates:
            break
        
        # Pick the candidate that blocks the most other valid candidates
        best_candidate = None
        best_score = -1
        
        for x in valid_candidates:
            # Simulate adding x
            new_sums = sums | set([2 * x] + [x + a for a in selected])
            
            # Count how many other valid candidates would be blocked
            blocked = 0
            for y in valid_candidates:
                if y == x:
                    continue
                y_added = [2 * y] + [y + a for a in selected] + [x + y]
                if len(set(y_added)) != len(y_added):
                    blocked += 1
                    continue
                if new_sums.intersection(y_added):
                    blocked += 1
            
            # Score: prefer candidates that block more (to minimize final size)
            score = blocked
            if score > best_score:
                best_score = score
                best_candidate = x
        
        if best_candidate is None:
            break
        
        x = best_candidate
        selected.append(x)
        selected_set.add(x)
        remaining.discard(x)
        added_sums = [2 * x] + [x + a for a in selected[:-1]]
        sums.update(added_sums)
    
    return selected

def _try_random_greedy(N, deadline, seed_offset=0):
    """Randomized greedy with different orderings."""
    import random
    rng = random.Random(42 + seed_offset)
    
    start_time = time.time()
    selected = []
    selected_set = set()
    sums = set()
    remaining = list(range(1, N + 1))
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Find valid candidates
        valid_candidates = []
        for x in remaining:
            added_sums = [2 * x] + [x + a for a in selected]
            if len(set(added_sums)) != len(added_sums):
                continue
            if sums.intersection(added_sums):
                continue
            valid_candidates.append(x)
        
        if not valid_candidates:
            break
        
        # Pick randomly from valid candidates, weighted by blocking power
        # Compute blocking power for each
        scores = []
        for x in valid_candidates:
            new_sums = sums | set([2 * x] + [x + a for a in selected])
            blocked = 0
            for y in valid_candidates:
                if y == x:
                    continue
                y_added = [2 * y] + [y + a for a in selected] + [x + y]
                if len(set(y_added)) != len(y_added):
                    blocked += 1
                    continue
                if new_sums.intersection(y_added):
                    blocked += 1
            scores.append(blocked)
        
        # Pick the one with highest score, with some randomness
        max_score = max(scores)
        top_candidates = [x for x, s in zip(valid_candidates, scores) if s >= max_score - 1]
        x = rng.choice(top_candidates)
        
        selected.append(x)
        selected_set.add(x)
        remaining.remove(x)
        added_sums = [2 * x] + [x + a for a in selected[:-1]]
        sums.update(added_sums)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_result = None
    best_size = float('inf')
    
    # Try multiple strategies
    strategies = [
        ("max_blocking", lambda: _greedy_max_blocking(N, deadline)),
        ("min_card", lambda: _greedy_min_cardinality(N, deadline)),
    ]
    
    # Add several randomized variants
    for i in range(10):
        strategies.append((f"random_{i}", lambda i=i: _try_random_greedy(N, deadline, i)))
    
    for name, func in strategies:
        if time.time() > deadline:
            break
        
        result = func()
        if not result:
            continue
        
        # Verify validity
        if not _is_valid_sidon(result):
            continue
        
        # Check maximality
        if not _is_maximal(result, N):
            continue
        
        if len(result) < best_size:
            best_size = len(result)
            best_result = result
    
    if best_result is None:
        # Fallback: use the simple ascending greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
    
    return best_result

# EVOLVE-BLOCK-END
