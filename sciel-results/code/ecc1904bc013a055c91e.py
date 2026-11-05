"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin
    
    N = 180
    all_elements = list(range(1, N + 1))
    
    # Precompute all forbidden triples: (a, b, c) where a < b < c and 1/a = 1/b + 1/c
    # This means c = a*b/(b-a), so b-a must divide a*b
    forbidden = []  # list of (a, b, c) tuples
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            diff = b - a
            if a * b % diff == 0:
                c = a * b // diff
                if c > b and c <= N:
                    forbidden.append((a, b, c))
    
    # Build for each element, the list of triples it participates in
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(forbidden):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
    
    # Start with all elements, then iteratively remove the worst offender
    included = set(all_elements)
    
    def count_violations():
        count = 0
        for a, b, c in forbidden:
            if a in included and b in included and c in included:
                count += 1
        return count
    
    # Greedy removal: repeatedly remove the element in the most violated triples
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Count how many violated triples each element is in
        viol_count = {i: 0 for i in included}
        for a, b, c in forbidden:
            if a in included and b in included and c in included:
                viol_count[a] += 1
                viol_count[b] += 1
                viol_count[c] += 1
        
        if max(viol_count.values()) == 0:
            break
        
        # Remove the element with the highest violation count
        worst = max(included, key=lambda x: viol_count[x])
        included.remove(worst)
        improved = True
    
    # Now try to add back elements that were removed
    removed = set(all_elements) - included
    
    # Try adding back each removed element if it doesn't create a violation
    for elem in sorted(removed, reverse=True):
        if (time.time() - start_time) > deadline - 5:
            break
        # Check if adding this element creates any new violation
        can_add = True
        for idx in elem_triples[elem]:
            a, b, c = forbidden[idx]
            others = [x for x in (a, b, c) if x != elem]
            if all(o in included for o in others):
                can_add = False
                break
        if can_add:
            included.add(elem)
    
    # Local search: try swapping - remove one, add another
    best_size = len(included)
    
    # Multiple passes of local search
    for iteration in range(20):
        if (time.time() - start_time) > deadline - 5:
            break
        improved = False
        for elem in list(included):
            if (time.time() - start_time) > deadline - 5:
                break
            included.remove(elem)
            # Try to add a removed element
            for add_elem in sorted(removed, reverse=True):
                can_add = True
                for idx in elem_triples[add_elem]:
                    a, b, c = forbidden[idx]
                    others = [x for x in (a, b, c) if x != add_elem]
                    if all(o in included for o in others):
                        can_add = False
                        break
                if can_add:
                    included.add(add_elem)
                    removed.discard(add_elem)
                    removed.add(elem)
                    improved = True
                    break
            if not improved:
                included.add(elem)
                removed.discard(elem)
    
    # Final verification
    for a, b, c in forbidden:
        if a in included and b in included and c in included:
            # This shouldn't happen, but if it does, remove one
            included.remove(a)
    
    return sorted(included)

# EVOLVE-BLOCK-END
