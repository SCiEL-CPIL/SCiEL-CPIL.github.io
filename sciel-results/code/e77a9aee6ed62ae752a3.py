import math
import time
import itertools

def _score(u):
    """Compute the score for set u."""
    if not u:
        return 0.0
    max_u = max(u)
    if max_u == 0:
        return 1.0
    diffs = set()
    sums = set()
    for a in u:
        for b in u:
            diffs.add(a - b)
            sums.add(a + b)
    num_diffs = len(diffs)
    num_sums = len(sums)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    denom = 2 * max_u + 1
    if denom <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(denom)

def _build_from_pattern(pattern, offset=0):
    """Build a set from a pattern of relative positions."""
    return sorted(set(offset + x for x in pattern if x >= 0))

def _random_growth(best_u, max_time, rng):
    """
    Try to improve the set by random additions/removals.
    Returns the best set found.
    """
    current = list(best_u)
    best_score = _score(current)
    best_set = list(current)
    
    max_val = max(current) + 100 if current else 100
    max_val = min(max_val, 20000)
    
    while time.time() < max_time:
        # Try a random modification
        op = rng.choice(['add', 'remove', 'swap'])
        new_set = list(current)
        
        if op == 'add':
            candidate = rng.randint(0, max_val)
            if candidate not in new_set:
                new_set.append(candidate)
                new_set.sort()
                if 0 not in new_set:
                    continue
        elif op == 'remove':
            if len(new_set) <= 1:
                continue
            idx = rng.randint(1, len(new_set) - 1)  # Keep 0
            new_set.pop(idx)
        elif op == 'swap':
            if len(new_set) < 2:
                continue
            idx = rng.randint(1, len(new_set) - 1)
            new_val = rng.randint(0, max_val)
            if new_val not in new_set or new_val == new_set[idx]:
                new_set[idx] = new_val
                new_set = sorted(set(new_set))
                if 0 not in new_set:
                    continue
        
        new_score = _score(new_set)
        if new_score > best_score:
            best_score = new_score
            best_set = list(new_set)
            current = list(new_set)
            # Expand max_val if we added a large element
            if max_val < max(new_set) + 100:
                max_val = max(new_set) + 100
                max_val = min(max_val, 50000)
    
    return best_set

def _local_search(best_u, max_time, rng):
    """
    More structured local search: try adding elements in arithmetic progressions
    or geometric progressions to exploit additive structure.
    """
    current = list(best_u)
    best_score = _score(current)
    best_set = list(current)
    
    while time.time() < max_time:
        # Try adding an arithmetic progression starting from a random element
        if current:
            start = rng.choice(current) if len(current) > 1 else 0
            step = rng.randint(1, max(1, len(current) * 2))
            length = rng.randint(2, 5)
            new_elements = [start + i * step for i in range(length)]
            new_set = sorted(set(current + new_elements))
            if 0 not in new_set:
                new_set = sorted(set([0] + new_set))
            new_score = _score(new_set)
            if new_score > best_score:
                best_score = new_score
                best_set = list(new_set)
                current = list(new_set)
        
        # Try removing a random non-zero element
        if len(current) > 1:
            idx = rng.randint(1, len(current) - 1)
            new_set = current[:idx] + current[idx+1:]
            new_score = _score(new_set)
            if new_score > best_score:
                best_score = new_score
                best_set = list(new_set)
                current = list(new_set)
    
    return best_set

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    
    Strategy: Start with a small seed set, then use random growth and local search
    to find a set that maximizes the score.
    """
    start_time = time.time()
    total_time = 120  # Leave margin from 150s limit
    rng = __import__('random').Random(42)
    
    # Initial seed sets to try
    seeds = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 3, 5],
        [0, 1, 3, 4, 6],
        [0, 1, 2, 4, 5],
        [0, 1, 4, 5],
        [0, 1, 2, 5, 6],
    ]
    
    # Evaluate seeds
    best_set = seeds[0]
    best_score = _score(best_set)
    
    for seed in seeds[1:]:
        s = _score(seed)
        if s > best_score:
            best_score = s
            best_set = list(seed)
    
    # Phase 1: Random growth (60% of time)
    time_left = total_time - (time.time() - start_time)
    phase1_time = time_left * 0.5
    if phase1_time > 5:
        best_set = _random_growth(best_set, start_time + time_left * 0.5, rng)
        best_score = _score(best_set)
    
    # Phase 2: Local search with structured moves (30% of time)
    time_left = total_time - (time.time() - start_time)
    phase2_time = time_left * 0.5
    if phase2_time > 5:
        best_set = _local_search(best_set, start_time + time_left * 0.5, rng)
        best_score = _score(best_set)
    
    # Phase 3: Final random polish (10% of time)
    time_left = total_time - (time.time() - start_time)
    if time_left > 2:
        best_set = _random_growth(best_set, start_time + time_left * 0.9, rng)
    
    return best_set
