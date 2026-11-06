import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    N = 180
    # Precompute all forbidden triples: (a, b, c) with a < b < c, 1/a = 1/b + 1/c
    # a = bc/(b+c), so a must divide bc and a < b
    triples = []  # list of (a, b, c)
    triple_count = {}  # element -> number of triples it's in
    for a in range(1, N + 1):
        # Find b, c with b < c, b > a, c > a, and a = bc/(b+c)
        # bc = a(b+c) => bc - ab - ac = 0 => (b-a)(c-a) = a^2
        a2 = a * a
        # Factor pairs of a^2
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            # b = a + d, c = a + e (or b = a + e, c = a + d)
            b1 = a + d
            c1 = a + e
            if b1 > a and c1 > a and b1 != c1 and b1 <= N and c1 <= N:
                if b1 < c1:
                    triples.append((a, b1, c1))
                    triple_count[a] = triple_count.get(a, 0) + 1
                    triple_count[b1] = triple_count.get(b1, 0) + 1
                    triple_count[c1] = triple_count.get(c1, 0) + 1
            # Also check if d != e gives another pair (already covered by symmetry)
    
    # Greedy construction: start with all elements, iteratively remove
    # the element involved in the most remaining triples
    in_set = [True] * (N + 1)  # 1-indexed
    # Count active triples
    active_triples = set(triples)
    
    # Better approach: greedy add from large to small
    # Start empty, add elements in decreasing order
    # When adding x, check if it forms a triple with any two existing elements
    S = set()
    # For each x, precompute which pairs (b,c) in potential S would conflict
    # Simpler: iterate x from N down to 1, add x if no conflict
    
    # Precompute for each x, the set of (b,c) pairs that would conflict
    # x conflicts with (b,c) if x = bc/(b+c), i.e., (b-x)(c-x) = x^2
    # Or if b = x*c/(c-x) for c > x, etc.
    
    # Actually, let's do: for each x, find all pairs (b,c) with b,c != x, b<c, 
    # such that {x,b,c} is a forbidden triple
    conflict_pairs = {x: [] for x in range(1, N+1)}
    for (a, b, c) in triples:
        # a is the "small" one
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    # Greedy: add from large to small
    S = set()
    for x in range(N, 0, -1):
        # Check if adding x creates any forbidden triple
        can_add = True
        for (b, c) in conflict_pairs[x]:
            if b in S and c in S:
                can_add = False
                break
        if can_add:
            S.add(x)
    
    # Local search: try to improve by swapping
    # Try removing one element and adding another
    best_S = set(S)
    best_size = len(S)
    
    # Try all single removals and see if we can add something back
    for _ in range(50):  # limited iterations
        if time.time() > deadline:
            break
        improved = False
        for remove_elem in list(best_S):
            new_S = best_S - {remove_elem}
            # Try to add elements not in new_S
            for add_elem in range(1, N+1):
                if add_elem in new_S:
                    continue
                # Check if adding add_elem to new_S is valid
                valid = True
                for (b, c) in conflict_pairs[add_elem]:
                    if b in new_S and c in new_S:
                        valid = False
                        break
                if valid:
                    if len(new_S) + 1 > len(best_S):
                        best_S = new_S | {add_elem}
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    result = sorted(best_S)
    return result
