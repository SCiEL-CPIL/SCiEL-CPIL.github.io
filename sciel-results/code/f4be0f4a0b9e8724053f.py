import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _precompute_conflicts():
    """Precompute all conflict triples (a,b,c) where 1/a = 1/b + 1/c with distinct a,b,c in [1,180]."""
    # For each a, find all pairs (b,c) with b != c, b != a, c != a, and 1/a = 1/b + 1/c
    # This means b*c = a*(b+c), so b*c - a*b - a*c = 0, (b-a)(c-a) = a^2
    # Let d = b-a, then c-a = a^2/d, so d must divide a^2
    # b = a+d, c = a + a^2/d
    # Need b != c => d != a^2/d => d^2 != a^2 => d != a (since d>0)
    # Also need b,c in [1,180] and distinct from a
    conflicts = {}  # maps element -> set of (other1, other2) pairs that conflict with it
    triple_set = set()
    
    for a in range(1, 181):
        a2 = a * a
        # Find all divisors d of a^2 where d != a
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
                    # Store the triple
                    triple = tuple(sorted((a, b, c)))
                    triple_set.add(triple)
    
    # Build conflict graph: for each element, which other elements form a conflict triple with it
    # Actually, we need: for each pair (x,y), does there exist z such that {x,y,z} is a conflict triple?
    # More useful: for each element a, the set of elements that cannot be co-present with a given another element
    # Let's build: conflict_pairs[a] = set of (b,c) such that {a,b,c} is a conflict triple
    # And: incompatible[a][b] = True if a and b cannot both be in a valid set (i.e., there exists c with {a,b,c} conflict)
    
    # Actually for incremental checking, we want: given a current set S, can we add element x?
    # We need to check: for every pair (y,z) in S, is {x,y,z} a conflict triple?
    # So we precompute for each x, the set of pairs (y,z) such that {x,y,z} is a conflict triple.
    
    # conflicts_with[x] = set of frozenset({y,z}) such that {x,y,z} is a conflict triple
    conflicts_with = {i: set() for i in range(1, 181)}
    for (a, b, c) in triple_set:
        conflicts_with[a].add(frozenset((b, c)))
        conflicts_with[b].add(frozenset((a, c)))
        conflicts_with[c].add(frozenset((a, b)))
    
    return triple_set, conflicts_with

def _can_add(element, current_set, conflicts_with):
    """Check if element can be added to current_set without creating a conflict triple."""
    # For each pair (y,z) in current_set, check if {element, y, z} is a conflict triple
    # Equivalently: check if any (y,z) in conflicts_with[element] has both y and z in current_set
    for pair in conflicts_with[element]:
        y, z = tuple(pair)
        if y in current_set and z in current_set:
            return False
    return True

def _fast_check_validity(s):
    """Faster validity check using precomputed conflicts."""
    s_set = set(s)
    for a in s_set:
        a2 = a * a
        for b in s_set:
            if b <= a:
                continue
            d = b - a
            if a2 % d != 0:
                continue
            c = a + a2 // d
            if c == a or c == b:
                continue
            if c in s_set:
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

def _local_search_1swap(current_set, all_elements, conflicts_with, time_limit):
    """Try 1-vertex swaps: remove one element, try to add others."""
    deadline = time.time() + time_limit
    current = set(current_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element and adding a different one
        for remove_elem in list(current):
            if time.time() > deadline:
                break
            temp_set = current - {remove_elem}
            # Try adding elements not in temp_set
            for add_elem in all_elements:
                if add_elem in temp_set:
                    continue
                if _can_add(add_elem, temp_set, conflicts_with):
                    # Check if this actually improves (adds an element we didn't have)
                    new_set = temp_set | {add_elem}
                    if len(new_set) > len(current):
                        current = new_set
                        improved = True
                        break
                    elif add_elem not in current_set:
                        # Same size but different composition - might help future swaps
                        # For now, only accept if strictly better or same size with new element
                        pass
    
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

def _random_restart_greedy(all_elements, conflicts_with, time_limit):
    """Start from empty set, greedily add in random order."""
    deadline = time.time() + time_limit
    current_set = set()
    candidates = all_elements[:]
    random.shuffle(candidates)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if _can_add(cand, current_set, conflicts_with):
            current_set.add(cand)
    
    return current_set

def construct():
    deadline = time.time() + 85  # 85s limit, 5s margin
    
    # Precompute conflicts
    triple_set, conflicts_with = _precompute_conflicts()
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
    
    # Local search: 1-swap
    if time.time() < deadline - 30:
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            result = _local_search_1swap(best, all_elements, conflicts_with, 5)
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
                result_set = _local_search_1swap(result, all_elements, conflicts_with, 2)
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
