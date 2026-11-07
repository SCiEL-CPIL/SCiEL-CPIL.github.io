"""Improved program: cardinality-indexed subset-sum DP to maximize set size."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_SECONDS = 55.0  # safety margin before 60s kill


def _build_dp_state(S):
    """
    Build DP state for set S:
      sums_by_k[k] = set of sums achievable with exactly k elements
      sum_to_k[s]  = set of cardinalities k such that s is achievable with k elements
    """
    n = len(S)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in S:
        for k in range(n, 0, -1):
            if sums_by_k[k - 1]:
                for s in sums_by_k[k - 1]:
                    sums_by_k[k].add(s + x)
    
    sum_to_k = defaultdict(set)
    for k in range(n + 1):
        for s in sums_by_k[k]:
            sum_to_k[s].add(k)
    
    return sums_by_k, sum_to_k


def _check_add_valid(sums_by_k, sum_to_k, candidate):
    """
    Check if adding candidate to the set represented by (sums_by_k, sum_to_k) is valid.
    Returns (True, new_sums_by_k, new_sum_to_k) if valid, (False, None, None) otherwise.
    Only checks newly introduced sums for collisions.
    """
    n = len(sums_by_k) - 1
    
    # New sums introduced at each cardinality:
    # At cardinality k (1..n+1), new sums = {s + candidate for s in sums_by_k[k-1]}
    
    # Collect all new sums by cardinality
    new_sums = {}
    for k in range(1, n + 2):
        src = sums_by_k[k - 1]
        if src:
            new_sums[k] = {s + candidate for s in src}
    
    # Check collisions:
    all_new = {}  # sum -> set of cardinalities where it's newly introduced
    for k, sums in new_sums.items():
        for s in sums:
            if s in all_new:
                # Two new sums at different cardinalities collide
                return False, None, None
            all_new[s] = {k}
            # Check against existing sum_to_k
            if s in sum_to_k:
                existing_ks = sum_to_k[s]
                if k not in existing_ks:
                    # Collision with existing sum at different cardinality
                    return False, None, None
    
    # No collisions found; build new state
    new_sums_by_k = [set() for _ in range(n + 2)]
    new_sums_by_k[0].add(0)
    for k in range(1, n + 1):
        new_sums_by_k[k] = set(sums_by_k[k])
        if k in new_sums:
            new_sums_by_k[k].update(new_sums[k])
    if n + 1 in new_sums:
        new_sums_by_k[n + 1] = set(new_sums[n + 1])
    
    # Update sum_to_k
    new_sum_to_k = defaultdict(set)
    for k in range(n + 2):
        for s in new_sums_by_k[k]:
            new_sum_to_k[s].add(k)
    
    return True, new_sums_by_k, new_sum_to_k


def _score_candidate(x, sums_by_k, sum_to_k):
    """
    Heuristic score for candidate x: lower is better.
    Counts how many new sums would collide with existing sums at other cardinalities.
    """
    n = len(sums_by_k) - 1
    collision_count = 0
    for k in range(1, n + 2):
        src = sums_by_k[k - 1]
        if not src:
            continue
        for s in src:
            new_s = s + x
            if new_s in sum_to_k:
                if k not in sum_to_k[new_s]:
                    collision_count += 1
    return collision_count


def _greedy_build(candidates, deadline):
    """Greedily build a valid set from candidates list using incremental DP."""
    current = []
    # Initialize DP state for empty set
    sums_by_k = [set()]
    sums_by_k[0].add(0)
    sum_to_k = defaultdict(set)
    sum_to_k[0].add(0)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(current) >= 60:
            break
        valid, new_sums_by_k, new_sum_to_k = _check_add_valid(sums_by_k, sum_to_k, x)
        if valid:
            current.append(x)
            sums_by_k = new_sums_by_k
            sum_to_k = new_sum_to_k
    return current


def _greedy_build_scored(candidates, deadline):
    """
    Greedy build that scores candidates and picks the least-colliding one each step.
    """
    current = []
    sums_by_k = [set()]
    sums_by_k[0].add(0)
    sum_to_k = defaultdict(set)
    sum_to_k[0].add(0)
    
    remaining = list(candidates)
    
    while remaining and time.time() < deadline and len(current) < 60:
        # Score all remaining candidates
        best_x = None
        best_score = float('inf')
        for x in remaining:
            sc = _score_candidate(x, sums_by_k, sum_to_k)
            if sc < best_score:
                best_score = sc
                best_x = x
            if sc == 0:
                break  # Can't do better than 0
        
        if best_x is None:
            break
        
        # Try to add best_x
        valid, new_sums_by_k, new_sum_to_k = _check_add_valid(sums_by_k, sum_to_k, best_x)
        if valid:
            current.append(best_x)
            sums_by_k = new_sums_by_k
            sum_to_k = new_sum_to_k
            remaining.remove(best_x)
        else:
            # Shouldn't happen if score was 0, but be safe
            remaining.remove(best_x)
    
    return current


def _local_search_improve(best_set, deadline, max_iter=200):
    """
    Try to improve best_set by swapping elements or adding new ones.
    Returns an improved set (possibly same as input).
    """
    current = best_set[:]
    if not current:
        return current
    
    sums_by_k, sum_to_k = _build_dp_state(current)
    in_set = set(current)
    
    # First pass: try to add elements
    remaining = [x for x in range(1, 257) if x not in in_set]
    random.shuffle(remaining)
    
    for x in remaining:
        if time.time() > deadline:
            break
        if len(current) >= 60:
            break
        valid, new_sums_by_k, new_sum_to_k = _check_add_valid(sums_by_k, sum_to_k, x)
        if valid:
            current.append(x)
            sums_by_k = new_sums_by_k
            sum_to_k = new_sum_to_k
            in_set.add(x)
    
    # Second pass: try swaps to potentially enable more additions
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if len(current) >= 60:
            break
        
        # Try swapping each element with a candidate not in set
        current_list = current[:]
        random.shuffle(current_list)
        
        found_swap = False
        for i, elem in enumerate(current_list):
            if time.time() > deadline:
                break
            
            # Remove elem
            # Rebuild DP without elem
            without = [x for x in current if x != elem]
            if len(without) == 0:
                continue
            
            sums_by_k_wo, sum_to_k_wo = _build_dp_state(without)
            
            # Try adding candidates
            candidates_to_try = [x for x in range(1, 257) if x not in set(without)]
            random.shuffle(candidates_to_try)
            
            best_addition = None
            for x in candidates_to_try:
                if time.time() > deadline:
                    break
                valid, new_sums_by_k, new_sum_to_k = _check_add_valid(sums_by_k_wo, sum_to_k_wo, x)
                if valid:
                    # Check if this enables further additions
                    test_set = without + [x]
                    test_sums_by_k, test_sum_to_k = _build_dp_state(test_set)
                    
                    # Try to add more
                    further = 0
                    remaining2 = [y for y in range(1, 257) if y not in set(test_set)]
                    random.shuffle(remaining2)
                    for y in remaining2[:20]:  # Limit to keep fast
                        valid2, _, _ = _check_add_valid(test_sums_by_k, test_sum_to_k, y)
                        if valid2:
                            further += 1
                    
                    if best_addition is None or further > best_addition[1]:
                        best_addition = (x, further)
                
                if time.time() > deadline - 5:
                    break
            
            if best_addition and best_addition[1] > 0:
                # Perform the swap
                current = without + [best_addition[0]]
                sums_by_k, sum_to_k = _build_dp_state(current)
                in_set = set(current)
                found_swap = True
                break
        
        if not found_swap:
            break
    
    return current


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Deterministic ascending with incremental DP
    candidates = list(range(1, 257))
    current = _greedy_build(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 2: Deterministic descending
    candidates = list(range(256, 0, -1))
    current = _greedy_build(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 3: Scored greedy (pick least-colliding candidate each step)
    candidates = list(range(1, 257))
    current = _greedy_build_scored(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 4: Random restarts with various biases
    random.seed(42)
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Various randomization strategies
        if attempt % 4 == 0:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            current = _greedy_build(candidates, deadline)
        elif attempt % 4 == 1:
            # Bias toward smaller values
            candidates = list(range(1, 129))
            random.shuffle(candidates)
            candidates += list(range(129, 257))
            random.shuffle(candidates)
            current = _greedy_build(candidates, deadline)
        elif attempt % 4 == 2:
            # Mix of small and large
            small = list(range(1, 65))
            large = list(range(192, 257))
            random.shuffle(small)
            random.shuffle(large)
            candidates = small + large
            random.shuffle(candidates)
            current = _greedy_build(candidates, deadline)
        else:
            # Scored greedy on random order
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            current = _greedy_build_scored(candidates, deadline)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # If we've found a large set, try to improve it further
        if best_size >= 20 and time.time() < deadline - 5:
            # Try adding remaining candidates not in best_set
            in_set = set(best_set)
            remaining = [x for x in range(1, 257) if x not in in_set]
            random.shuffle(remaining)
            
            # Build DP state for best_set
            sums_by_k, sum_to_k = _build_dp_state(best_set)
            
            improved = best_set[:]
            for x in remaining:
                if time.time() > deadline:
                    break
                if len(improved) >= 60:
                    break
                valid, new_sums_by_k, new_sum_to_k = _check_add_valid(sums_by_k, sum_to_k, x)
                if valid:
                    improved.append(x)
                    sums_by_k = new_sums_by_k
                    sum_to_k = new_sum_to_k
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved[:]
    
    # Final local search improvement
    if best_set and time.time() < deadline - 3:
        improved = _local_search_improve(best_set, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    # Final: ensure all elements are Python int
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
