"""Solve equal-LCM-triple avoidance via maximum independent set on conflict graph."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all pairwise LCMS
    lcm_table = {}
    for i in range(n):
        for j in range(i + 1, n):
            lcm_table[(i, j)] = _lcm(elements[i], elements[j])
    
    # Find all "bad triples": three distinct elements where all 3 pairwise LCMS are equal
    # For each pair (i,j), find all k such that lcm(i,j) == lcm(i,k) == lcm(j,k)
    # Group by LCM value: for each LCM value L, find all pairs with that LCM
    # Then any triple within the same LCM group where all 3 pairs have LCM = L is bad
    
    # More efficient: for each LCM value, collect all pairs with that LCM
    # A triple (a,b,c) is bad if lcm(a,b)=lcm(a,c)=lcm(b,c)=L for some L
    # This means all three pairs are in the same LCM group
    
    # Build: for each LCM value, the set of pairs that have that LCM
    from collections import defaultdict
    lcm_to_pairs = defaultdict(list)
    for (i, j), L in lcm_table.items():
        lcm_to_pairs[L].append((i, j))
    
    # For each LCM value L, find all triples where all 3 pairs have LCM = L
    # This means we need a triangle in the graph of pairs-with-LCM-L
    # Build adjacency: for each LCM value, which pairs share endpoints
    
    # For each LCM value L, build a graph where vertices are the elements involved
    # and edges are the pairs with LCM = L. Then find all triangles.
    bad_triples = set()
    for L, pairs in lcm_to_pairs.items():
        if len(pairs) < 3:
            continue
        # Build adjacency map for this LCM group
        adj = defaultdict(set)
        for i, j in pairs:
            adj[i].add(j)
            adj[j].add(i)
        # Find triangles: for each pair (i,j) in this group, check if there's a k
        # such that (i,k) and (j,k) are also in this group
        for i, j in pairs:
            common = adj[i] & adj[j]
            for k in common:
                if k != i and k != j:
                    triple = tuple(sorted((i, j, k)))
                    bad_triples.add(triple)
    
    # Now build conflict graph: two elements conflict if they are part of a bad triple together
    # Actually, we need: a set S is valid iff it contains no bad triple
    # This is a 3-uniform hypergraph independent set problem
    # But we can reduce to a graph: if (a,b,c) is a bad triple, then no two of {a,b,c} can both be in S
    # Wait, that's not right. We can have two of them, just not all three.
    # So it's a hypergraph independent set, not a graph independent set.
    
    # Alternative approach: use the fact that if (a,b,c) is bad, then for any pair among them,
    # the third is "forbidden" given the pair. We can model this as: for each pair (i,j),
    # the set of k that form a bad triple with i,j.
    
    # Let's use a different approach: branch and bound on the hypergraph
    # But first, let's try a greedy + local search approach which was measured to work well
    
    # Greedy: add elements in order, skip if it creates a bad triple
    def is_valid_addition(candidate, current_set, bad_triple_set):
        """Check if adding candidate to current_set creates any bad triple."""
        # A bad triple is created if there exist a, b in current_set such that
        # (a, b, candidate) is a bad triple
        for a in current_set:
            for b in current_set:
                if a < b:
                    triple = tuple(sorted((a, b, candidate)))
                    if triple in bad_triple_set:
                        return False
        return True
    
    # Greedy construction
    selected = set()
    for i in range(n):
        if is_valid_addition(i, selected, bad_triple_set):
            selected.add(i)
    
    best_size = len(selected)
    best_set = set(selected)
    
    # Local search / simulated annealing to improve
    import random
    random.seed(42)
    
    current = set(best_set)
    current_size = len(current)
    
    # Precompute: for each pair in current, which elements would form a bad triple
    def get_conflicts_with_pair(pair, all_elements, bad_triple_set):
        """For a pair (a,b), find all c such that (a,b,c) is a bad triple."""
        a, b = pair
        conflicts = set()
        for c in all_elements:
            if c != a and c != b:
                triple = tuple(sorted((a, b, c)))
                if triple in bad_triple_set:
                    conflicts.add(c)
        return conflicts
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    max_iters = 50000
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        if T < T_min:
            break
            
        # Try a random move: add an element, remove an element, or swap
        move_type = random.random()
        
        if move_type < 0.33:
            # Try to add a random element not in current
            not_in = [i for i in range(n) if i not in current]
            if not not_in:
                continue
            candidate = random.choice(not_in)
            # Check if adding would create a bad triple
            valid = True
            for a in current:
                for b in current:
                    if a < b:
                        triple = tuple(sorted((a, b, candidate)))
                        if triple in bad_triple_set:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                new_current = current | {candidate}
                delta = 1
                if delta > 0 or random.random() < (1 - T) / (T + 1e-10):
                    current = new_current
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
            else:
                # Can't add, try removing one of the conflicting elements
                # Find which pairs conflict
                for a in current:
                    for b in current:
                        if a < b:
                            triple = tuple(sorted((a, b, candidate)))
                            if triple in bad_triple_set:
                                # Remove a or b
                                remove_choice = random.choice([a, b])
                                new_current = current - {remove_choice}
                                delta = -1
                                if random.random() < (1 - T) / (T + 1e-10):
                                    current = new_current
                                break
                    else:
                        continue
                    break
        
        elif move_type < 0.66:
            # Try to remove a random element
            if len(current) > 1:
                to_remove = random.choice(list(current))
                new_current = current - {to_remove}
                delta = -1
                if random.random() < (1 - T) / (T + 1e-10):
                    current = new_current
        
        else:
            # Swap: remove one, add one
            if len(current) > 0:
                to_remove = random.choice(list(current))
                not_in = [i for i in range(n) if i not in current]
                if not_in:
                    to_add = random.choice(not_in)
                    new_current = (current - {to_remove}) | {to_add}
                    # Check validity
                    valid = True
                    for a in new_current:
                        for b in new_current:
                            if a < b:
                                triple = tuple(sorted((a, b, to_add)))
                                if to_add in new_current and triple in bad_triple_set:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        delta = 0
                        if random.random() < (1 - T) / (T + 1e-10):
                            current = new_current
                            if len(current) > best_size:
                                best_size = len(current)
                                best_set = set(current)
        
        T *= cooling
    
    # Final greedy improvement: try to add any remaining elements
    for i in range(n):
        if i not in best_set:
            valid = True
            for a in best_set:
                for b in best_set:
                    if a < b:
                        triple = tuple(sorted((a, b, i)))
                        if triple in bad_triple_set:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                best_set.add(i)
    
    # Convert to list of actual values
    result = sorted(elements[i] for i in best_set)
    
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
