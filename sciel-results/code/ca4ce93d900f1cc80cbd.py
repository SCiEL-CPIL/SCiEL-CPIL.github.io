"""Improved program: precomputed conflict hypergraph + degree-ordered greedy + local search."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd
    from itertools import combinations

    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Precompute lcm
    def lcm(a, b):
        return a * b // gcd(a, b)

    # Precompute conflict hypergraph: for each triple (a,b,c) with a<b<c,
    # check if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Store conflicts as: for each element x, set of (i,j) pairs that conflict with x
    # i.e., if x is added, it conflicts with pair (i,j)
    
    # Better approach: for each pair (a,b), find all c such that lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Then the conflict hypergraph edges are {a,b,c}
    
    # Let's build: conflict_pairs[x] = set of frozenset({a,b}) such that {a,b,x} is a bad triple
    # Actually simpler: bad_triples = list of (a,b,c)
    # For greedy: when considering adding x, check if there exist a,b in selected such that (a,b,x) is bad
    
    # Precompute all bad triples
    bad_triples = set()
    for a in range(1, 129):
        for b in range(a+1, 129):
            lab = lcm(a, b)
            for c in range(b+1, 129):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    bad_triples.add((a, b, c))
    
    # For efficient lookup: for each x, store the set of pairs (a,b) with a<b<x such that (a,b,x) is bad
    # And pairs (a,b) with x<a<b such that (a,b,x) is bad -> store as (min(a,b), max(a,b))
    # Actually let's just store for each x: set of frozenset pairs
    conflicts_with = [set() for _ in range(129)]  # conflicts_with[x] = set of (i,j) tuples
    for (a, b, c) in bad_triples:
        # triple (a,b,c) with a<b<c
        conflicts_with[a].add((b, c))
        conflicts_with[b].add((a, c))
        conflicts_with[c].add((a, b))
    
    # Degree: number of pairs that would conflict
    degrees = [len(conflicts_with[x]) for x in range(1, 129)]
    
    # Greedy: process elements in order of increasing degree (fewest conflicts first)
    # This is a known heuristic for maximum independent set in hypergraphs
    order = sorted(range(1, 129), key=lambda x: degrees[x])
    
    selected = []
    selected_set = set()
    
    for x in order:
        # Check if adding x creates any bad triple with existing pairs
        ok = True
        for (a, b) in conflicts_with[x]:
            if a in selected_set and b in selected_set:
                ok = False
                break
        if ok:
            selected.append(x)
            selected_set.add(x)
    
    # Local search: try to add more elements or swap
    # Try adding any element not in selected
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each non-selected element
        for x in range(1, 129):
            if x in selected_set:
                continue
            # Check if x can be added
            can_add = True
            for (a, b) in conflicts_with[x]:
                if a in selected_set and b in selected_set:
                    can_add = False
                    break
            if can_add:
                selected.append(x)
                selected_set.add(x)
                improved = True
                break  # restart check after adding one
        
        if not improved:
            # Try swap: remove one element, add another
            for x in range(1, 129):
                if x in selected_set:
                    continue
                # Try removing each selected element y and adding x
                for y in list(selected):
                    new_set = selected_set - {y}
                    # Check if x can be added to new_set
                    can_add = True
                    for (a, b) in conflicts_with[x]:
                        if a in new_set and b in new_set:
                            can_add = False
                            break
                    if can_add:
                        # Also need to verify the rest is still valid (it should be since we removed y)
                        selected_set.remove(y)
                        selected.remove(y)
                        selected.append(x)
                        selected_set.add(x)
                        improved = True
                        break
                if improved:
                    break
    
    return sorted(selected)
# EVOLVE-BLOCK-END
