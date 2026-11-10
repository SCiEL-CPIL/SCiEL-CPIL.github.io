"""Rewritten solver for B2[2] (Sidon-like) sets in 1..1000.

Strategy:
- Fast greedy construction with several diverse orderings.
- Local search: try single insertions, single deletions, and 1-for-1 swaps.
- Multiple random restarts with time budget.
- Return the best valid set found within the deadline.
"""

import time
import random
import sys

# Deadline: 150s wall-clock, keep safety margin
DEADLINE = 150.0
SAFETY_MARGIN = 5.0  # seconds before deadline to stop
N = 1000


def _is_valid_insertion(chosen_set, counts, x):
    """Check if x can be added to chosen_set without violating B2[2] property.
    
    chosen_set: set of ints
    counts: dict sum -> count of representations
    x: candidate to add
    
    Returns (True, new_counts) if valid, (False, None) otherwise.
    """
    # New sums: x + y for y in chosen_set, y != x (but we need a<=b so x+y where y in chosen_set)
    # Actually for a+b with a<=b: if x is new, pairs are (x, y) for y in chosen_set with y >= x? No.
    # Pairs involving x: (x, x) gives 2x, and (x, y) for y in chosen_set.
    # But we need a <= b, so if y < x, the pair is (y, x) which is already counted when y was added? 
    # No - when we add x, the new pairs are (x, y) for all y in chosen_set where y >= x? 
    # Actually no. The representation a+b with a<=b: if we add x, new pairs are:
    # - (x, x): sum 2x
    # - (x, y) for y in chosen_set: but we need x <= y for the pair to be (x,y), or if y < x, the pair is (y,x)
    # Wait, the pair is unordered in the sense that a<=b. So adding x creates pairs:
    # (min(x,y), max(x,y)) for each y in chosen_set, plus (x,x).
    # So new sums are: 2x, and x+y for each y in chosen_set.
    # But we must check if any of these sums would exceed count 2.
    
    new_sums = [2 * x] + [x + y for y in chosen_set]
    
    for s in new_sums:
        if counts.get(s, 0) + 1 > 2:
            return False, None
    
    # Valid - compute new counts
    new_counts = dict(counts)
    for s in new_sums:
        new_counts[s] = new_counts.get(s, 0) + 1
    return True, new_counts


def _remove_from_counts(chosen_set, counts, x):
    """Remove x from chosen_set and update counts."""
    new_sums = [2 * x] + [x + y for y in chosen_set if y != x]
    new_counts = dict(counts)
    for s in new_sums:
        new_counts[s] -= 1
        if new_counts[s] == 0:
            del new_counts[s]
    return new_counts


def _greedy_construct(ordering, max_time=None):
    """Greedy construction with given ordering of candidates."""
    chosen = []
    chosen_set = set()
    counts = {}
    
    start_time = time.time()
    for x in ordering:
        if max_time and (time.time() - start_time) > max_time:
            break
        valid, new_counts = _is_valid_insertion(chosen_set, counts, x)
        if valid:
            chosen.append(x)
            chosen_set.add(x)
            counts = new_counts
    
    return chosen


def _local_search(chosen, max_iterations=5000, max_time=None):
    """Local search: try insertions, deletions, and swaps to improve size."""
    chosen_set = set(chosen)
    # Build counts
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            s = a + chosen[j]
            counts[s] = counts.get(s, 0) + 1
    
    current_size = len(chosen)
    start_time = time.time()
    improved = True
    
    while improved and (max_time is None or (time.time() - start_time) < max_time):
        improved = False
        
        # Try single insertions
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in chosen_set:
                continue
            if (time.time() - start_time) > max_time:
                break
            
            valid, new_counts = _is_valid_insertion(chosen_set, counts, x)
            if valid:
                chosen.append(x)
                chosen_set.add(x)
                counts = new_counts
                current_size += 1
                improved = True
                break  # Restart after improvement
        
        if improved:
            continue
        
        # Try 1-for-1 swaps: remove one, add another
        for i in range(len(chosen)):
            if (time.time() - start_time) > max_time:
                break
            
            x_remove = chosen[i]
            # Temporarily remove
            temp_counts = _remove_from_counts(chosen_set, counts, x_remove)
            temp_set = chosen_set - {x_remove}
            
            # Try adding candidates
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
            for x_add in candidates:
                if x_add in temp_set:
                    continue
                
                valid, new_counts = _is_valid_insertion(temp_set, temp_counts, x_add)
                if valid:
                    # Perform swap
                    chosen[i] = x_add
                    chosen_set.remove(x_remove)
                    chosen_set.add(x_add)
                    counts = new_counts
                    improved = True
                    break
            
            if improved:
                break
    
    return chosen


def _try_greedy_with_ordering(ordering, time_budget):
    """Run greedy then local search, return best found."""
    chosen = _greedy_construct(ordering, max_time=time_budget * 0.3)
    remaining = time_budget * 0.7
    chosen = _local_search(chosen, max_time=remaining)
    return chosen


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best = []
    
    # Strategy 1: Ascending greedy
    if time.time() < deadline:
        t0 = time.time()
        chosen = _greedy_construct(range(1, N + 1))
        # Local search
        remaining = deadline - time.time()
        if remaining > 1:
            chosen = _local_search(chosen, max_time=remaining)
        if len(chosen) > len(best):
            best = chosen
    
    # Strategy 2: Descending greedy
    if time.time() < deadline:
        t0 = time.time()
        chosen = _greedy_construct(range(N, 0, -1))
        remaining = deadline - time.time()
        if remaining > 1:
            chosen = _local_search(chosen, max_time=remaining)
        if len(chosen) > len(best):
            best = chosen
    
    # Strategy 3: Random restarts with local search
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline:
        # Create a random ordering with bias
        candidates = list(range(1, N + 1))
        # Weight toward sparser numbers or random
        if attempts % 3 == 0:
            # Sparse bias: prefer numbers with larger gaps
            random.shuffle(candidates)
        elif attempts % 3 == 1:
            # Descending with noise
            candidates = list(range(N, 0, -1))
            for i in range(len(candidates) - 1, 0, -1):
                j = rng.randint(0, i)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        else:
            # Random
            random.shuffle(candidates)
        
        # Quick greedy
        max_greedy_time = min(2.0, (deadline - time.time()) * 0.2)
        chosen = _greedy_construct(candidates, max_time=max_greedy_time)
        
        # Local search with remaining time
        remaining = deadline - time.time()
        if remaining > 1:
            chosen = _local_search(chosen, max_time=remaining)
        
        if len(chosen) > len(best):
            best = chosen
        
        attempts += 1
        # Limit total attempts to avoid infinite loop
        if attempts >= 50:
            break
    
    # Final local search on best if time permits
    if time.time() < deadline and len(best) > 0:
        remaining = deadline - time.time()
        if remaining > 1:
            best = _local_search(best, max_time=remaining)
    
    return best


# Test
if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}")
    # Verify
    from collections import Counter
    c = Counter()
    for i in range(len(result)):
        for j in range(i, len(result)):
            c[result[i] + result[j]] += 1
    max_rep = max(c.values()) if c else 0
    print(f"Max representations: {max_rep}")
    assert max_rep <= 2, "Invalid B2[2] set!"
    assert all(1 <= x <= 1000 for x in result)
    assert len(set(result)) == len(result)
