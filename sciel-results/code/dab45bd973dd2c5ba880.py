"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # We need a subset of [1, 180] with no three distinct elements a, b, c
    # such that 1/a = 1/b + 1/c.
    # This is equivalent to: 1/a = (b+c)/(bc) => bc = a(b+c) => bc - ab - ac = 0
    # => bc - ab - ac + a^2 = a^2 => (b-a)(c-a) = a^2
    # Let x = b-a, y = c-a. Then xy = a^2, and b = a+x, c = a+y.
    # Since b, c are distinct from a and each other, x != 0, y != 0, x != y (if x=y then b=c, not allowed).
    # Also b, c must be in [1, 180].
    
    # For a given a, the forbidden pairs (b, c) with b != c, b, c != a are those where
    # (b-a)(c-a) = a^2.
    # So for each divisor d of a^2, we can set b-a = d, c-a = a^2/d.
    # Then b = a + d, c = a + a^2/d.
    # We need b, c in [1, 180], b != c, b != a, c != a.
    # Since d divides a^2 and d != 0, b != a and c != a automatically.
    # b != c means d != a^2/d => d^2 != a^2 => d != a and d != -a.
    # Since we're looking at positive divisors for positive b-a, c-a, we assume d > 0.
    # Then b = a+d > a, c = a + a^2/d > a.
    # b != c iff d != a^2/d iff d^2 != a^2 iff d != a.
    
    # So for each a in [1, 180], we generate forbidden pairs (b, c) with b < c (wlog)
    # where (b-a)(c-a) = a^2, b, c in [1, 180], b != c.
    
    # Actually, we should also consider negative differences?
    # If b < a, then b-a < 0, so c-a must be negative too, meaning c < a.
    # Then (b-a)(c-a) = positive = a^2.
    # Let b = a - x, c = a - y where x, y > 0 and xy = a^2.
    # Then b = a-x, c = a-y.
    # b != c iff x != y iff x^2 != a^2 iff x != a.
    
    # So forbidden triples are:
    # 1. a < b, a < c, (b-a)(c-a) = a^2
    # 2. a > b, a > c, (a-b)(a-c) = a^2
    # 3. b < a < c or c < a < b: then one difference is positive, one negative, product negative, can't equal a^2 > 0.
    
    # So all forbidden triples have all three elements on the same side of a? No, a is the "small" one in case 1 and "large" one in case 2.
    # In case 1: a < b, a < c.
    # In case 2: b < a, c < a.
    
    # Let's build a graph where vertices are numbers 1..180, and we have hyperedges (triples) that are forbidden.
    # We want a maximum independent set in this 3-uniform hypergraph.
    # This is NP-hard, so we use heuristics.
    
    # Strategy: Greedy with random restarts and local search.
    # Or: Start with a large set and remove elements that cause conflicts.
    
    # Let's precompute all forbidden triples.
    N = 180
    forbidden = set()
    for a in range(1, N+1):
        a2 = a * a
        # Case 1: b > a, c > a
        for d in range(1, a2+1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d == e:  # b == c
                continue
            b = a + d
            c = a + e
            if b > N or c > N:
                continue
            if b == c:
                continue
            # b, c, a are distinct
            triple = tuple(sorted((a, b, c)))
            forbidden.add(triple)
        # Case 2: b < a, c < a
        for d in range(1, a2+1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d == e:
                continue
            b = a - d
            c = a - e
            if b < 1 or c < 1:
                continue
            if b == c:
                continue
            triple = tuple(sorted((a, b, c)))
            forbidden.add(triple)
    
    # Now we have the set of forbidden triples.
    # We want to maximize the size of a subset S of [1, 180] such that no forbidden triple is contained in S.
    
    # Heuristic: Greedy removal.
    # Start with all numbers, remove elements that are in many forbidden triples.
    # Or: Greedy addition. Start empty, add elements if they don't create a forbidden triple.
    
    # Let's try a greedy approach: sort elements by some criterion and add them if possible.
    # Or: start with all, and iteratively remove the element that is in the most forbidden triples.
    
    # Let's try: start with all elements. While there's a forbidden triple in the set, remove one of its elements.
    # To maximize size, remove the element that is in the most forbidden triples.
    
    # But this is slow. Let's precompute for each element how many forbidden triples it's in.
    
    # Actually, let's use a different approach: 
    # The seed program returns [91, 180], size 90.
    # Can we do better?
    
    # Let's try a greedy independent set algorithm for hypergraphs.
    # 1. Start with all vertices.
    # 2. While there's a hyperedge fully contained in the set:
    #    - Pick a hyperedge.
    #    - Remove the vertex from it that has the smallest degree (number of hyperedges it's in).
    
    # But this might not be optimal. Let's try random restarts with this heuristic.
    
    def greedy_remove(vertices, forbidden_triples):
        vertices = set(vertices)
        # Build a map from vertex to list of triples containing it
        vertex_triples = defaultdict(list)
        for t in forbidden_triples:
            for v in t:
                if v in vertices:
                    vertex_triples[v].append(t)
        
        # We'll maintain a set of active triples (those with all vertices in vertices)
        active_triples = set()
        for t in forbidden_triples:
            if all(v in vertices for v in t):
                active_triples.add(t)
        
        while active_triples:
            # Pick a triple with the minimum total degree of its vertices
            min_deg = float('inf')
            chosen_triple = None
            for t in active_triples:
                deg = sum(len(vertex_triples[v]) for v in t)
                if deg < min_deg:
                    min_deg = deg
                    chosen_triple = t
            
            if chosen_triple is None:
                break
            
            # Remove the vertex in chosen_triple with the smallest degree
            min_v_deg = float('inf')
            remove_v = None
            for v in chosen_triple:
                if v in vertices:
                    d = len(vertex_triples[v])
                    if d < min_v_deg:
                        min_v_deg = d
                        remove_v = v
            
            if remove_v is None:
                break
            
            # Remove remove_v
            vertices.remove(remove_v)
            # Update active_triples and vertex_triples
            # This is inefficient, but let's try
            
            # Remove all triples containing remove_v from active_triples
            new_active = set()
            for t in active_triples:
                if remove_v in t:
                    continue
                if all(v in vertices for v in t):
                    new_active.add(t)
            active_triples = new_active
            
            # Update vertex_triples
            for v in list(vertex_triples.keys()):
                new_list = [t for t in vertex_triples[v] if all(u in vertices for u in t)]
                vertex_triples[v] = new_list
                if not new_list:
                    del vertex_triples[v]
        
        return vertices
    
    # This is O(n^3) or worse. Let's try a simpler approach.
    
    # Alternative: Start with all, and for each forbidden triple, remove one element.
    # Use a priority queue of triples by the minimum degree of their elements.
    
    # Let's try a different heuristic: 
    # Sort numbers 1..180. 
    # We know that [91, 180] works. 
    # Maybe we can include some numbers below 91?
    
    # Let's check: if a < 91, can we include it?
    # For a in [1, 90], the forbidden triples involve b, c > a or b, c < a.
    # If we include a, we must exclude all b, c such that (b-a)(c-a) = a^2.
    
    # Let's try a greedy construction:
    # Start with empty set.
    # For a in 1..180 (or in some order), add a if it doesn't create a forbidden triple.
    
    # Let's try multiple orders and take the best.
    
    def greedy_add(order, forbidden_triples):
        S = set()
        # For each a in order:
        #   Check if adding a creates a forbidden triple with elements already in S.
        #   A forbidden triple containing a is (a, b, c) where (b-a)(c-a) = a^2.
        #   We need to check if there exist b, c in S such that (a, b, c) is forbidden.
        
        # Precompute for each a, the set of (b, c) pairs that form a forbidden triple with a.
        # Actually, we can just check all forbidden triples containing a.
        
        # Let's build a map: a -> list of (b, c) such that (a, b, c) is forbidden.
        # But we already have forbidden_triples as a set of sorted triples.
        
        # For efficiency, let's build:
        # for each a, list of other two elements that with a form a forbidden triple.
        a_pairs = defaultdict(list)
        for t in forbidden_triples:
            for i in range(3):
                a = t[i]
                others = [t[j] for j in range(3) if j != i]
                a_pairs[a].append(tuple(others))
        
        for a in order:
            # Check if adding a creates a forbidden triple
            ok = True
            for (b, c) in a_pairs[a]:
                if b in S and c in S:
                    ok = False
                    break
            if ok:
                S.add(a)
        
        return S
    
    # Let's try different orders:
    # 1. Natural order 1..180
    # 2. Reverse order 180..1
    # 3. Random orders
    
    best = set()
    best_size = 0
    
    # Try natural order
    S = greedy_add(list(range(1, N+1)), forbidden)
    if len(S) > best_size:
        best = S
        best_size = len(S)
    
    # Try reverse order
    S = greedy_add(list(range(N, 0, -1)), forbidden)
    if len(S) > best_size:
        best = S
        best_size = len(S)
    
    # Try random orders
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        S = greedy_add(order, forbidden)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Also try the seed solution
    seed = set(range(91, 181))
    if len(seed) > best_size:
        best = seed
        best_size = len(seed)
    
    # Try to improve by local search: remove an element, see if we can add two others?
    # Or: start with best, and try to swap elements.
    
    # Let's try a simple local search: for each element in best, try removing it and see if we can add a new element.
    # This is complex. Let's just return the best we found.
    
    return list(best)
# EVOLVE-BLOCK-END
