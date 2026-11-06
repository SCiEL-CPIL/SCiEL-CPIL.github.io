import time
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 60.0

    nums = list(range(1, 129))
    
    # Precompute LCMs
    lcm_cache = {}
    for i in range(1, 129):
        for j in range(i + 1, 129):
            lcm_cache[(i, j)] = lcm(i, j)

    # Identify bad triples
    bad_triples = set()
    for a, b, c in combinations(nums, 3):
        if lcm_cache[(a, b)] == lcm_cache[(a, c)] == lcm_cache[(b, c)]:
            bad_triples.add(frozenset((a, b, c)))

    # Map element to triples containing it
    triples_for_x = {x: [] for x in nums}
    for t in bad_triples:
        for x in t:
            triples_for_x[x].append(t)

    def is_valid_addition(x, current_set):
        for t in triples_for_x[x]:
            others = [y for y in t if y != x]
            if others[0] in current_set and others[1] in current_set:
                return False
        return True

    # Initial Greedy
    selected = []
    for x in nums:
        if all(not (frozenset((a, b, x)) in bad_triples) for a, b in combinations(selected, 2)):
            selected.append(x)
            
    best_set = set(selected)
    
    # Greedy Addition of remaining
    remaining = [x for x in nums if x not in best_set]
    for x in remaining:
        if is_valid_addition(x, best_set):
            best_set.add(x)

    # Local Search: Try to swap elements to increase size?
    # Actually, the greedy addition should find a maximal independent set.
    # To improve, we can try randomized restarts or perturbation.
    
    best_size = len(best_set)
    best_result = list(best_set)
    
    # Simple Local Search: Randomly remove an element, try to re-add many
    # Given time constraints, a single good greedy pass with sorted order might be enough.
    # Let's try sorting by degree (number of conflicts) to pack more.
    
    # Degree of x = number of triples containing x
    degrees = {x: len(triples_for_x[x]) for x in nums}
    
    # Try greedy with different orderings
    orders = [
        list(range(1, 129)),
        sorted(nums, key=lambda x: degrees[x]),
        sorted(nums, key=lambda x: -degrees[x]),
        sorted(nums, reverse=True)
    ]
    
    for order in orders:
        if time.time() > deadline - 5.0:
            break
        curr_set = set()
        for x in order:
            if is_valid_addition(x, curr_set):
                curr_set.add(x)
        if len(curr_set) > best_size:
            best_size = len(curr_set)
            best_result = list(curr_set)

    return best_result
