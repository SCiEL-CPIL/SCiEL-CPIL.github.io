"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 512
    n = N  # We are working with [1, N]
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) where 1 <= a, a+3d <= N
    # For efficiency, we'll build a conflict graph: for each number x, 
    # which other numbers y form a 4-term AP with x?
    # Actually, a 4-term AP is forbidden if all 4 terms are in the set.
    # So the "conflict" is not pairwise. This is a 4-uniform hypergraph.
    # We want a subset with no hyperedge fully contained.
    
    # Let's use a greedy approach with backtracking or local search.
    # Given the size (512), a good heuristic is to use a random greedy 
    # construction and then try to improve it via local search (add/remove/swap).
    
    # However, for 4-AP-free sets, known constructions give density ~ n^(1/3) or similar?
    # Actually, for 3-AP-free, density is ~ n / log(n). For 4-AP-free, it's smaller.
    # But we just need a good heuristic.
    
    # Let's try a randomized greedy with multiple restarts and local improvement.
    
    # First, let's try a simple greedy: iterate 1..N, add if no 4-AP is completed.
    # But the order matters. Let's try different orderings.
    
    # Better: use a SAT-like approach or integer programming? Too slow.
    
    # Let's use a local search:
    # 1. Start with a random subset or a greedy subset.
    # 2. Try to add elements that don't create a 4-AP.
    # 3. Try to swap elements to increase size.
    
    # Precompute for each 4-AP (a, b, c, d) the set of elements.
    # There are O(N^2) such APs.
    
    # Let's represent the set as a boolean array of length N+1 (1-indexed).
    # For checking if adding x creates a 4-AP, we need to check if there exists
    # a 4-AP containing x where the other 3 are already in the set.
    
    # For a given x, the 4-APs containing x are:
    # Case 1: x = a (first term): x, x+d, x+2d, x+3d
    # Case 2: x = a+d (second term): x-d, x, x+d, x+2d
    # Case 3: x = a+2d (third term): x-2d, x-d, x, x+d
    # Case 4: x = a+3d (fourth term): x-3d, x-2d, x-d, x
    
    # So for each x, we can precompute the list of "triples" (other 3 elements)
    # such that {x} U triple forms a 4-AP.
    
    # Let's build this index.
    ap_triples = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            # The AP is (a, b, c, e)
            ap_triples[a].append((b, c, e))
            ap_triples[b].append((a, c, e))
            ap_triples[c].append((a, b, e))
            ap_triples[e].append((a, b, c))
    
    # Now, a greedy construction:
    def greedy_construct(order):
        selected = [False] * (N + 1)
        count = 0
        for x in order:
            # Check if adding x creates a 4-AP
            can_add = True
            for (y, z, w) in ap_triples[x]:
                if selected[y] and selected[z] and selected[w]:
                    can_add = False
                    break
            if can_add:
                selected[x] = True
                count += 1
        return selected, count
    
    # Try different orderings:
    # 1. Natural order
    # 2. Random order
    # 3. Order by number of conflicts (fewest first)
    
    best_set = None
    best_count = 0
    
    # Natural order
    order = list(range(1, N + 1))
    selected, count = greedy_construct(order)
    if count > best_count:
        best_count = count
        best_set = selected[:]
    
    # Try random orders
    random.seed(42)
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        selected, count = greedy_construct(order)
        if count > best_count:
            best_count = count
            best_set = selected[:]
    
    # Local search improvement:
    # Try to add elements that were not added, or swap to increase size.
    # Actually, the greedy is maximal (no element can be added without creating a 4-AP).
    # So we need to try swaps: remove one element, add two others?
    # Or remove one, add one? That won't increase size.
    # So we need to remove k elements and add k+1 elements.
    
    # Let's try a simple local search:
    # 1. Start with best_set.
    # 2. Try to find an element x not in the set that can be added if we remove some elements.
    # This is complex. Let's try a different approach.
    
    # Alternative: use a simulated annealing or tabu search.
    # But given the time, let's stick with multiple greedy restarts and a simple local improvement.
    
    # Let's try to improve by removing one element and adding two.
    # For each element x in the set, remove it, then try to add elements not in the set.
    # This is O(N^3) which might be too slow.
    
    # Let's try a simpler heuristic: for each element not in the set, check if it can be added
    # by removing one element from the set.
    
    # Actually, let's just do more random restarts and keep the best.
    
    # Also, try a greedy that prioritizes elements with fewer conflicts.
    # Compute the number of 4-APs each element is in.
    conflict_count = [len(ap_triples[i]) for i in range(N + 1)]
    
    # Order by conflict count (ascending)
    order = sorted(range(1, N + 1), key=lambda x: conflict_count[x])
    selected, count = greedy_construct(order)
    if count > best_count:
        best_count = count
        best_set = selected[:]
    
    # Try reverse conflict count
    order = sorted(range(1, N + 1), key=lambda x: -conflict_count[x])
    selected, count = greedy_construct(order)
    if count > best_count:
        best_count = count
        best_set = selected[:]
    
    # Try a few more random orders
    for _ in range(500):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        selected, count = greedy_construct(order)
        if count > best_count:
            best_count = count
            best_set = selected[:]
    
    # Convert best_set to a sorted list
    result = [i for i in range(1, N + 1) if best_set[i]]
    
    # Verify no 4-AP
    # (Optional, but good for correctness)
    # for a in range(1, N + 1):
    #     for d in range(1, (N - a) // 3 + 1):
    #         if best_set[a] and best_set[a+d] and best_set[a+2*d] and best_set[a+3*d]:
    #             # Found a 4-AP, which should not happen
    #             pass
    
    return result

# EVOLVE-BLOCK-END
