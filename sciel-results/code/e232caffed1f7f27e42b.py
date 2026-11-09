import time
import random
from itertools import combinations

START_TIME = time.time()
DEADLINE = 55.0  # seconds (safety margin from 60s)

def elapsed():
    return time.time() - START_TIME

def is_valid_set(sums_by_card):
    """Check that no sum appears in two different cardinality sets."""
    seen = {}
    for k, s in sums_by_card.items():
        for val in s:
            if val in seen and seen[val] != k:
                return False
            seen[val] = k
    return True

def build_sums_by_card(elements):
    """Build dictionary: cardinality -> set of achievable sums."""
    n = len(elements)
    sums_by_card = {k: set() for k in range(n + 1)}
    sums_by_card[0].add(0)
    
    for i, elem in enumerate(elements):
        # Update in reverse to avoid using the same element twice
        for k in range(i, -1, -1):
            for s in list(sums_by_card[k]):
                sums_by_card[k + 1].add(s + elem)
    return sums_by_card

def backtracking_search(elements, target_size, max_value=256):
    """
    Backtracking search for a set of target_size distinct integers in 1..max_value
    such that subset sums of different cardinalities are disjoint.
    
    Uses forward checking: after adding an element, verify no cross-cardinality
    sum collision occurs.
    """
    if len(elements) >= target_size:
        return elements[:]
    
    # Determine which values we can still add
    used = set(elements)
    # We want to try values that minimize future collisions
    # Sort candidates: prefer values that create fewer new sum collisions
    
    current_sums = build_sums_by_card(elements)
    n = len(elements)
    
    # Generate candidate values not yet used
    candidates = [v for v in range(1, max_value + 1) if v not in used]
    
    # Score each candidate by how many cross-cardinality collisions it would create
    def score_candidate(v):
        new_sums = build_sums_by_card(elements + [v])
        # Count collisions between different cardinalities
        seen = {}
        collisions = 0
        for k, s in new_sums.items():
            for val in s:
                if val in seen and seen[val] != k:
                    collisions += 1
                seen[val] = k
        return collisions
    
    # Sort candidates by collision count (ascending), then by value
    scored = []
    for v in candidates:
        c = score_candidate(v)
        scored.append((c, v))
    
    # Only consider candidates with 0 collisions first
    zero_collision = [v for c, v in scored if c == 0]
    
    if zero_collision:
        # Try zero-collision candidates first, in descending order (larger values
        # leave more room for smaller values later)
        for v in sorted(zero_collision, reverse=True):
            if elapsed() > DEADLINE:
                return None
            result = backtracking_search(elements + [v], target_size, max_value)
            if result is not None:
                return result
        return None
    else:
        # No zero-collision candidates; try all with increasing collisions
        scored.sort()
        for c, v in scored:
            if elapsed() > DEADLINE:
                return None
            result = backtracking_search(elements + [v], target_size, max_value)
            if result is not None:
                return result
        return None

def greedy_construct(n_target):
    """Greedy construction: add elements one by one, choosing the value
    that causes the fewest cross-cardinality sum collisions."""
    elements = []
    max_value = 256
    
    while len(elements) < n_target:
        used = set(elements)
        candidates = [v for v in range(1, max_value + 1) if v not in used]
        
        if not candidates:
            break
        
        current_sums = build_sums_by_card(elements)
        
        def count_collisions(v):
            new_sums = build_sums_by_card(elements + [v])
            seen = {}
            collisions = 0
            for k, s in new_sums.items():
                for val in s:
                    if val in seen and seen[val] != k:
                        collisions += 1
                    seen[val] = k
            return collisions
        
        scored = [(count_collisions(v), v) for v in candidates]
        scored.sort()
        
        # Pick the best candidate (fewest collisions, then largest value)
        best_c, best_v = scored[0]
        
        if best_c > 0:
            # Adding this element creates collisions; try to see if we can still
            # complete the set, otherwise stop
            # For greedy, we just stop if collisions occur
            break
        
        elements.append(best_v)
        
        if elapsed() > DEADLINE:
            break
    
    return elements

def local_search_improve(elements, target_size, max_value=256):
    """Local search: try swapping elements to increase set size or maintain validity."""
    if len(elements) >= target_size:
        return elements
    
    elements = elements[:]
    used = set(elements)
    
    # Try to add one more element
    best_new = None
    best_score = float('inf')
    
    for v in range(1, max_value + 1):
        if v in used:
            continue
        new_sums = build_sums_by_card(elements + [v])
        seen = {}
        collisions = 0
        for k, s in new_sums.items():
            for val in s:
                if val in seen and seen[val] != k:
                    collisions += 1
                seen[val] = k
        if collisions == 0:
            # Prefer larger values to leave room for smaller ones
            if best_new is None or v > best_new:
                best_new = v
                best_score = 0
    
    if best_new is not None:
        elements.append(best_new)
    
    return elements

def construct():
    """Main entry point: construct the largest valid set possible."""
    if elapsed() > DEADLINE:
        return []
    
    # Start with the known valid construction of size 31
    # {226, 227, ..., 256}
    base = list(range(226, 257))
    best = base
    
    # Try to extend beyond 31 using backtracking
    # First, verify base is valid
    sums = build_sums_by_card(base)
    if not is_valid_set(sums):
        # Shouldn't happen, but fallback
        best = []
    
    # Try backtracking search for larger sets
    # We'll try to find a set of size 32, 33, etc.
    for target in range(len(best) + 1, 40):  # Try up to size 39
        if elapsed() > DEADLINE - 5:
            break
        
        # Use backtracking with the base as a starting hint
        # But backtracking from scratch may be expensive
        # Instead, try greedy extension first
        extended = greedy_construct(target)
        if len(extended) > len(best):
            best = extended
            # Verify
            if is_valid_set(build_sums_by_card(best)):
                continue
            else:
                best = base
        
        # Try local search improvement
        improved = local_search_improve(best, target)
        if len(improved) > len(best) and is_valid_set(build_sums_by_card(improved)):
            best = improved
    
    # Final attempt: try backtracking search for a slightly larger set
    if elapsed() < DEADLINE - 10 and len(best) < 35:
        # Try to find a set of size len(best)+1 using backtracking
        # Start with empty and use the scoring heuristic
        result = backtracking_search([], len(best) + 1, 256)
        if result is not None and len(result) > len(best):
            if is_valid_set(build_sums_by_card(result)):
                best = result
    
    # Ensure we return a valid set
    if best:
        if not is_valid_set(build_sums_by_card(best)):
            # Fallback to base
            best = list(range(226, 257))
    
    return best
