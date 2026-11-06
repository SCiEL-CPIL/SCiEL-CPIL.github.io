"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    N = 256
    # Precompute which sums are perfect squares
    # a+b ranges from 1+1=2 to 256+256=512
    # Perfect squares in that range: 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484
    square_sums = set()
    for k in range(2, 23):  # k^2 up to 22^2=484
        sq = k * k
        if 2 <= sq <= 512:
            square_sums.add(sq)
    
    # Precompute for each a, which b values would conflict (a+b is square)
    # conflict[a] = set of b such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):  # b >= a to handle a<=b
            if a + b in square_sums:
                conflict[a].add(b)
                if b != a:
                    conflict[b].add(a)
    
    # Also need to check: for a in set, is 2*a a perfect square? (diagonal)
    # This is already covered: conflict[a] contains a if 2*a is square
    
    def is_valid_subset(subset_set):
        """Check if subset is valid (no pair a<=b has a+b square)."""
        for a in subset_set:
            for b in conflict[a]:
                if b in subset_set:
                    return False
        return True
    
    def count_conflicts(subset_set, candidate):
        """Count how many elements in subset_set conflict with candidate."""
        cnt = 0
        for b in conflict[candidate]:
            if b in subset_set:
                cnt += 1
        return cnt
    
    def greedy_min_degree():
        """Greedy: pick elements with minimum conflicts among remaining."""
        remaining = set(range(1, N + 1))
        result = set()
        while remaining:
            best_elem = None
            best_conf = None
            for elem in remaining:
                conf_count = len(conflict[elem] & result)
                if best_conf is None or conf_count < best_conf:
                    best_conf = conf_count
                    best_elem = elem
            if best_conf == 0:
                result.add(best_elem)
                remaining.discard(best_elem)
            else:
                # If all remaining have conflicts, pick the one with min conflict
                result.add(best_elem)
                remaining.discard(best_elem)
        return result
    
    def local_search_improve(subset_set, time_limit):
        """Try to improve by adding elements or swapping."""
        subset_set = set(subset_set)
        end_time = time.time() + time_limit
        improved = True
        while improved and time.time() < end_time:
            improved = False
            # Try adding elements
            for cand in range(1, N + 1):
                if cand in subset_set:
                    continue
                if count_conflicts(subset_set, cand) == 0:
                    subset_set.add(cand)
                    improved = True
                    break
            # Try swapping: remove one, add two
            if not improved:
                removed_list = list(subset_set)
                random.shuffle(removed_list)
                for rem in removed_list:
                    if time.time() >= end_time:
                        break
                    temp_set = subset_set - {rem}
                    # Try to add two elements
                    added = []
                    for cand in range(1, N + 1):
                        if cand in temp_set:
                            continue
                        if count_conflicts(temp_set, cand) == 0:
                            temp_set.add(cand)
                            added.append(cand)
                            if len(added) >= 2:
                                break
                    if len(added) >= 2:
                        subset_set = temp_set
                        improved = True
                        break
                    elif len(added) == 1 and False:
                        pass
                    else:
                        temp_set.difference_update(added)
        return subset_set
    
    def local_search_swap(subset_set, time_limit):
        """Local search with single swaps and additions."""
        subset_set = set(subset_set)
        end_time = time.time() + time_limit
        improved = True
        while improved and time.time() < end_time:
            improved = False
            # Try adding
            for cand in range(1, N + 1):
                if cand not in subset_set and count_conflicts(subset_set, cand) == 0:
                    subset_set.add(cand)
                    improved = True
                    break
            # Try 1-for-1 swap
            if not improved:
                for rem in list(subset_set):
                    if time.time() >= end_time:
                        break
                    temp = subset_set - {rem}
                    for cand in range(1, N + 1):
                        if cand not in temp and count_conflicts(temp, cand) == 0:
                            # Check if this is actually an improvement (size stays same, but maybe enables more)
                            # For now, only do swap if it allows subsequent additions
                            subset_set = temp | {cand}
                            improved = True
                            break
                    if improved:
                        break
        return subset_set
    
    # Strategy: multiple random restarts with greedy + local search
    best_size = 0
    best_set = None
    
    # First, try a deterministic greedy
    greedy_set = greedy_min_degree()
    greedy_set = local_search_improve(greedy_set, 2.0)
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set
    
    # Random restarts
    random.seed(42)
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # Create a random partial set and extend greedily
        subset = set()
        # Randomly seed with some elements
        n_seed = random.randint(0, 20)
        for _ in range(n_seed):
            cand = random.randint(1, N)
            if count_conflicts(subset, cand) == 0:
                subset.add(cand)
        
        # Greedy extend
        remaining = set(range(1, N + 1)) - subset
        while remaining:
            best_elem = None
            best_conf = None
            for elem in remaining:
                conf_count = len(conflict[elem] & subset)
                if best_conf is None or conf_count < best_conf:
                    best_conf = conf_count
                    best_elem = elem
            if best_conf == 0:
                subset.add(best_elem)
                remaining.discard(best_elem)
            else:
                # Pick min conflict even if > 0
                subset.add(best_elem)
                remaining.discard(best_elem)
        
        # Local search
        subset = local_search_improve(subset, 1.0)
        
        if len(subset) > best_size:
            best_size = len(subset)
            best_set = subset
        
        # Also try starting from best_set and perturbing
        if best_set and random.random() < 0.5:
            perturbed = set(best_set)
            # Remove 1-3 random elements
            n_remove = random.randint(1, 3)
            elems = list(perturbed)
            random.shuffle(elems)
            for _ in range(min(n_remove, len(elems))):
                perturbed.discard(elems[_])
            # Re-extend
            remaining = set(range(1, N + 1)) - perturbed
            while remaining:
                best_elem = None
                best_conf = None
                for elem in remaining:
                    conf_count = len(conflict[elem] & perturbed)
                    if best_conf is None or conf_count < best_conf:
                        best_conf = conf_count
                        best_elem = elem
                if best_conf == 0:
                    perturbed.add(best_elem)
                    remaining.discard(best_elem)
                else:
                    perturbed.add(best_elem)
                    remaining.discard(best_elem)
            perturbed = local_search_improve(perturbed, 1.0)
            if len(perturbed) > best_size:
                best_size = len(perturbed)
                best_set = perturbed
    
    # Final: ensure best_set is valid
    if best_set is None:
        best_set = greedy_min_degree()
    
    # Verify validity
    if not is_valid_subset(best_set):
        # Fallback: build valid set greedily
        best_set = set()
        for a in range(1, N + 1):
            if count_conflicts(best_set, a) == 0:
                best_set.add(a)
    
    return list(best_set)

# EVOLVE-BLOCK-END
