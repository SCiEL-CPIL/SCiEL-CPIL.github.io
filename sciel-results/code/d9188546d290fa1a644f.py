import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _precompute_conflicts():
    """Precompute all conflict triples (a,b,c) where 1/a = 1/b + 1/c with distinct a,b,c in [1,180]."""
    triple_set = set()
    
    for a in range(1, 181):
        a2 = a * a
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                for d_val in [d, a2 // d]:
                    if d_val == a:
                        continue
                    b = a + d_val
                    c = a + a2 // d_val
                    if b == c:
                        continue
                    if b < 1 or b > 180 or c < 1 or c > 180:
                        continue
                    if b == a or c == a:
                        continue
                    triple = tuple(sorted((a, b, c)))
                    triple_set.add(triple)
    
    conflicts_with = {i: set() for i in range(1, 181)}
    for (a, b, c) in triple_set:
        conflicts_with[a].add(frozenset((b, c)))
        conflicts_with[b].add(frozenset((a, c)))
        conflicts_with[c].add(frozenset((a, b)))
    
    # Compute degrees
    degrees = {i: len(conflicts_with[i]) for i in range(1, 181)}
    
    return triple_set, conflicts_with, degrees

def _can_add(element, current_set, conflicts_with):
    """Check if element can be added to current_set without creating a conflict triple."""
    for pair in conflicts_with[element]:
        y, z = tuple(pair)
        if y in current_set and z in current_set:
            return False
    return True

def _greedy_build(start_set, candidates, conflicts_with, time_limit):
    """Greedily add candidates to start_set if validity is maintained."""
    current_set = set(start_set)
    deadline = time.time() + time_limit
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in current_set:
            continue
        if _can_add(cand, current_set, conflicts_with):
            current_set.add(cand)
    
    return list(current_set)

def _local_search_1swap_improved(current_set, all_elements, conflicts_with, degrees, time_limit):
    """
    Improved local search:
    1. Try to add any element (increase size).
    2. If no add possible, try to swap u in S with v not in S to reduce total degree.
    """
    deadline = time.time() + time_limit
    current = set(current_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add an element
        for add_elem in all_elements:
            if add_elem in current:
                continue
            if _can_add(add_elem, current, conflicts_with):
                current.add(add_elem)
                improved = True
                break
        
        if improved:
            continue
            
        # Phase 2: No add possible. Try to swap to reduce conflict density
        # Calculate current total degree
        current_deg_sum = sum(degrees[x] for x in current)
        
        best_swap = None
        best_new_deg_sum = current_deg_sum
        
        for remove_elem in current:
            temp_set = current - {remove_elem}
            # Try adding elements not in temp_set
            for add_elem in all_elements:
                if add_elem in temp_set:
                    continue
                if _can_add(add_elem, temp_set, conflicts_with):
                    new_set = temp_set | {add_elem}
                    new_deg_sum = sum(degrees[x] for x in new_set)
                    if new_deg_sum < best_new_deg_sum:
                        best_new_deg_sum = new_deg_sum
                        best_swap = (remove_elem, add_elem)
            
        if best_swap:
            remove_elem, add_elem = best_swap
            current.remove(remove_elem)
            current.add(add_elem)
            improved = True
    
    return current

def _local_search_2swap(current_set, all_elements, conflicts_with, time_limit):
    """Try 2-vertex removals with 2-vertex additions."""
    deadline = time.time() + time_limit
    current = set(current_set)
    current_list = list(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing pairs and adding pairs
        for i in range(len(current_list)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(current_list)):
                if time.time() > deadline:
                    break
                remove_pair = {current_list[i], current_list[j]}
                temp_set = current - remove_pair
                # Try adding pairs of elements not in temp_set
                for add_elem in all_elements:
                    if add_elem in temp_set:
                        continue
                    if not _can_add(add_elem, temp_set, conflicts_with):
                        continue
                    temp_with_one = temp_set | {add_elem}
                    for add_elem2 in all_elements:
                        if add_elem2 in temp_with_one or add_elem2 == add_elem:
                            continue
                        if _can_add(add_elem2, temp_with_one, conflicts_with):
                            new_set = temp_with_one | {add_elem2}
                            if len(new_set) > len(current):
                                current = new_set
                                current_list = list(current)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    return current

def construct():
    deadline = time.time() + 85  # 85s limit, 5s margin
    
    # Precompute conflicts
    triple_set, conflicts_with, degrees = _precompute_conflicts()
    all_elements = list(range(1, 181))
    
    random.seed(42)
    
    # Start with high numbers
    best = set(range(91, 181))
    best_size = len(best)
    
    # Greedy add from low numbers in various orders
    low_candidates = list(range(1, 91))
    
    # Order 1: ascending
    result = _greedy_build(best, sorted(low_candidates), conflicts_with, 3)
    if len(result) > best_size:
        best = set(result)
        best_size = len(best)
    
    # Order 2: descending
    result = _greedy_build(best, sorted(low_candidates, reverse=True), conflicts_with, 3)
    if len(result) > best_size:
        best = set(result)
        best_size = len(best)
        
    # Order 3: by degree (ascending)
    sorted_by_degree = sorted(all_elements, key=lambda x: degrees[x])
    result = _greedy_build(set(), sorted_by_degree, conflicts_with, 3)
    if len(result) > best_size:
        best = set(result)
        best_size = len(best)
        
    # Random orders
    for i in range(30):
        if time.time() > deadline - 5:
            break
        shuffled = low_candidates[:]
        random.shuffle(shuffled)
        result = _greedy_build(best, shuffled, conflicts_with, 1)
        if len(result) > best_size:
            best = set(result)
            best_size = len(best)
    
    # Local search: 1-swap improved
    if time.time() < deadline - 30:
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            result = _local_search_1swap_improved(best, all_elements, conflicts_with, degrees, 5)
            if len(result) > best_size:
                best = result
                best_size = len(best)
                improved = True
    
    # Local search: 2-swap
    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            result = _local_search_2swap(best, all_elements, conflicts_with, 5)
            if len(result) > best_size:
                best = result
                best_size = len(best)
                improved = True
    
    # Random restarts with perturbation
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        num_restarts = min(20, int(remaining_time / 3))
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Perturb: start from best, remove some random elements
            perturbed = set(best)
            num_remove = random.randint(3, min(15, len(perturbed) // 3))
            to_remove = random.sample(list(perturbed), min(num_remove, len(perturbed)))
            for elem in to_remove:
                perturbed.discard(elem)
            
            # Greedy fill
            candidates = [x for x in all_elements if x not in perturbed]
            random.shuffle(candidates)
            result = _greedy_build(perturbed, candidates, conflicts_with, 2)
            
            # Local search on result
            if len(result) > best_size or random.random() < 0.3:
                result_set = _local_search_1swap_improved(result, all_elements, conflicts_with, degrees, 2)
                if len(result_set) > best_size:
                    best = result_set
                    best_size = len(best)
                result_set = _local_search_2swap(best, all_elements, conflicts_with, 2)
                if len(result_set) > best_size:
                    best = result_set
                    best_size = len(best)
    
    # Final polish: try all single additions
    if time.time() < deadline - 2:
        for elem in all_elements:
            if elem in best:
                continue
            if _can_add(elem, best, conflicts_with):
                best.add(elem)
                best_size = len(best)
    
    return list(best)

# EVOLVE-BLOCK-END
