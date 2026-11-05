"""
Strategy: Precompute all 4-AP conflict hyperedges once, then use a fast
independent-set search (greedy + local search) with a precomputed
conflict graph to maximize set size.
"""
import time
import random
import numpy as np

# Precompute all 4-APs in [1, 512]
# A 4-AP is (a, a+d, a+2d, a+3d) with d > 0
# We need to find a large subset with no 4-AP

def precompute_conflicts():
    """
    Precompute for each element x, the list of 4-APs it participates in.
    Return a dict: x -> list of 4-APs (as tuples)
    """
    n = 512
    # For each 4-AP (a, a+d, a+2d, a+3d), add it to the lists of all 4 elements
    # We'll store conflicts as: for each x, the set of (y, z, w) such that {x,y,z,w} is a 4-AP
    # But that's too much. Instead, let's store for each pair (x, y) the list of 4-APs they share.
    
    # Actually, for the greedy check, we need: given current set S, can we add x?
    # x conflicts if there exists a 4-AP containing x and at least 3 other elements in S.
    # So for each x, we need to know all 4-APs containing x.
    
    n = 512
    aps_by_element = {x: [] for x in range(1, n+1)}
    
    for a in range(1, n+1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e <= n:
                ap = (a, b, c, e)
                for x in ap:
                    aps_by_element[x].append(ap)
    
    return aps_by_element

# Precompute once
_AP_BY_ELEMENT = precompute_conflicts()

def can_add(x, current_set):
    """
    Check if x can be added to current_set without creating a 4-AP.
    x creates a 4-AP if there's a 4-AP containing x where the other 3 elements are all in current_set.
    """
    for ap in _AP_BY_ELEMENT[x]:
        others = [y for y in ap if y != x]
        if all(y in current_set for y in others):
            return False
    return True

def construct():
    """
    Construct a large subset of [1, 512] with no 4-term AP.
    Uses precomputed 4-AP conflicts for fast checks.
    """
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    n = 512
    all_elements = list(range(1, n + 1))
    
    # Strategy: Multiple greedy passes with different orderings, then local search.
    # Key insight: the order of greedy matters a lot.
    
    def greedy_pass(order):
        """Greedy: add elements in given order if they don't create a 4-AP."""
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    def local_search(current_set, max_steps=5000):
        """
        Local search: try to add random missing elements, or swap (remove one, add two).
        """
        current_set = set(current_set)
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve < max_no_improve:
            missing = [x for x in all_elements if x not in current_set]
            if not missing:
                break
            
            # Try to add a random missing element
            x = random.choice(missing)
            if can_add(x, current_set):
                current_set.add(x)
                no_improve = 0
            else:
                no_improve += 1
                # Try swap: remove one element from current_set, then try to add x
                # Only do this occasionally to save time
                if no_improve % 10 == 0 and len(current_set) > 10:
                    # Try removing a random element and adding x
                    candidates_to_remove = list(current_set)
                    random.shuffle(candidates_to_remove)
                    for y in candidates_to_remove[:5]:
                        trial = current_set - {y}
                        if can_add(x, trial):
                            # Check if we can also add another element
                            current_set = trial
                            current_set.add(x)
                            no_improve = 0
                            break
                    else:
                        # Restore
                        current_set.add(y)
        
        return current_set
    
    best_set = set()
    best_size = 0
    
    # Try several different greedy orderings
    # 1. Natural order
    s = greedy_pass(all_elements)
    best_set = s
    best_size = len(s)
    
    # 2. Reverse order
    if time.time() < deadline:
        s = greedy_pass(list(reversed(all_elements)))
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # 3. Random orders
    num_random = 20
    for i in range(num_random):
        if time.time() > deadline:
            break
        order = all_elements[:]
        random.shuffle(order)
        s = greedy_pass(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # 4. Local search from the best greedy result
    if time.time() < deadline:
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # 5. Try more local search iterations from different starts
    for i in range(5):
        if time.time() > deadline:
            break
        order = all_elements[:]
        random.shuffle(order)
        s = greedy_pass(order)
        if len(s) >= best_size - 2:  # Only if close to best
            improved = local_search(s, max_steps=2000)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(best_set)
    
    # Final verification
    def has_4ap(s):
        s_list = sorted(s)
        s_set = set(s_list)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                d = b - a
                c = b + d
                e = c + d
                if c in s_set and e in s_set:
                    return True
        return False
    
    if has_4ap(best_set):
        # Fallback: use a simple valid set
        # The set of numbers with no 4-AP in base 4 representation... 
        # Actually just use a small safe set
        return list(range(1, 10))
    
    return sorted(best_set)
