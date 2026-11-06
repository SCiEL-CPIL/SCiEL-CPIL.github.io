import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 180
    
    # Precompute all forbidden triples: (a, b, c) with distinct elements, 1/a = 1/b + 1/c
    # From 1/a = 1/b + 1/c => a = bc/(b+c) => (b-a)(c-a) = a^2
    # For each a, find factor pairs (d, e) of a^2 with d*e = a^2, then b=a+d, c=a+e
    # All three must be distinct and in [1, N]
    
    # For each element, store the set of other elements it conflicts with (in some triple)
    # Actually, we need: for each element x, which pairs (y,z) form a triple with x?
    # Better: for each element x, store the set of elements that cannot coexist with x
    # in a way that would create a triple. But triples involve 3 elements.
    
    # Let's precompute all triples first
    triples = []  # list of frozensets or sorted tuples (a,b,c)
    triple_set = set()
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find all factor pairs (d, e) with d * e = a2, d <= e
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d > e:
                continue
            b = a + d
            c = a + e
            # Check all three are in [1, N] and distinct
            if 1 <= a <= N and 1 <= b <= N and 1 <= c <= N:
                if a != b and a != c and b != c:
                    triple = tuple(sorted((a, b, c)))
                    if triple not in triple_set:
                        triple_set.add(triple)
                        triples.append(triple)
    
    # For greedy approach: for each element, count how many triples it's in
    # and for each pair of elements, check if they're in a triple together
    
    # Build conflict structure: for each element, which other elements can form a triple with it
    # Actually for greedy: when we add element x, we need to check that no existing pair (y,z) 
    # in our set forms a triple with x.
    
    # Better approach: greedy by adding elements in order of increasing "danger"
    # danger = number of triples the element participates in
    
    element_triple_count = [0] * (N + 1)
    for (a, b, c) in triples:
        element_triple_count[a] += 1
        element_triple_count[b] += 1
        element_triple_count[c] += 1
    
    # Sort elements by triple count (ascending) - add "safe" elements first
    elements_by_safety = sorted(range(1, N + 1), key=lambda x: element_triple_count[x])
    
    # For efficient checking, maintain for each pair of selected elements whether they conflict
    # Actually, let's just do a simple greedy: try to add each element in order, check validity
    
    selected = set()
    
    # To check if adding x is valid: for every pair (y, z) in selected, 
    # check that (x, y, z) is not a forbidden triple
    # This is O(|selected|^2) per insertion, which could be slow
    
    # Alternative: for each element x, precompute the set of "forbidden pairs" 
    # that would create a triple with x. I.e., pairs (y,z) such that {x,y,z} is a triple.
    
    # forbidden_pairs_for[x] = set of frozensets {y,z} where {x,y,z} is a triple
    forbidden_pairs_for = [set() for _ in range(N + 1)]
    for (a, b, c) in triples:
        # For element a: pairs (b,c)
        forbidden_pairs_for[a].add(frozenset((b, c)))
        forbidden_pairs_for[b].add(frozenset((a, c)))
        forbidden_pairs_for[c].add(frozenset((a, b)))
    
    # Greedy: add elements in order of increasing danger
    for x in elements_by_safety:
        if time.time() > deadline:
            break
        # Check if adding x would create a triple with any existing pair
        valid = True
        for pair in forbidden_pairs_for[x]:
            # pair is a frozenset of two elements
            if len(pair) == 2:
                y, z = tuple(pair)
                if y in selected and z in selected:
                    valid = False
                    break
        if valid:
            selected.add(x)
    
    result = sorted(selected)
    if not result:
        # Fallback: return the original safe set
        result = list(range(91, 181))
    
    return result
