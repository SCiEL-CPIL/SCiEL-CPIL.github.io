import time
import random

def construct():
    N = 180
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Generate all forbidden triples
    # a = bc / (b+c) => a(b+c) = bc => ab + ac - bc = 0
    # Alternatively, iterate a, then find b, c
    # 1/b + 1/c = 1/a => (b-a)(c-a) = a^2
    # Let x = b-a, y = c-a. xy = a^2.
    # b = a+x, c = a+y.
    # We need b, c <= N, b != c, a != b, a != c.
    # x, y > 0.
    # x != y.
    
    triples = []
    # For each a, find divisors of a^2
    for a in range(1, N + 1):
        a_sq = a * a
        # Find divisors of a_sq
        divs = []
        i = 1
        while i * i <= a_sq:
            if a_sq % i == 0:
                divs.append(i)
                if i != a_sq // i:
                    divs.append(a_sq // i)
            i += 1
        
        # Pairs (x, y) from divs such that x * y = a_sq and x < y
        # Actually, if x is a divisor, y = a_sq // x.
        # We need x < y.
        for x in divs:
            y = a_sq // x
            if x >= y:
                continue
            b = a + x
            c = a + y
            if b <= N and c <= N:
                # Ensure distinctness
                # a, b, c distinct?
                # b = a+x > a. c = a+y > a.
                # b != c because x != y.
                triples.append((a, b, c))
                
    # Remove duplicates?
    # The generation is unique for each a, x.
    # But a triple {a,b,c} is identified by its smallest element a.
    # So no duplicates.
    
    # 2. Identify elements in any triple
    elements_in_triples = set()
    for t in triples:
        elements_in_triples.update(t)
        
    U = [i for i in range(1, N + 1) if i not in elements_in_triples]
    V = [i for i in range(1, N + 1) if i in elements_in_triples]
    
    # Map element to index in V for faster lookup?
    # Or just use sets.
    
    # 3. Greedy Independent Set on V
    # We want to select max subset of V with no triple.
    
    def solve_greedy(elements, triples, seed=0):
        rng = random.Random(seed)
        S = set(elements)
        
        # Precompute for each element in S, the list of triples containing it
        # And for each triple, the elements
        # We need to efficiently find conflicts and remove.
        
        # active_triples: set of indices of triples where all 3 elements are in S
        # Initially, all triples where all 3 elements are in V (which is all of them, since triples are on V)
        # Wait, triples are on [1, N]. But we only care about those fully in V?
        # Yes, because U elements are never in a triple.
        # So all triples are fully in V.
        
        active_triples = set(range(len(triples)))
        
        # Count conflicts for each element
        # conflict_count[v] = number of active triples containing v
        conflict_count = {v: 0 for v in elements}
        for idx in active_triples:
            a, b, c = triples[idx]
            conflict_count[a] += 1
            conflict_count[b] += 1
            conflict_count[c] += 1
            
        # Mapping from element to list of triple indices containing it
        elem_to_triples = {v: [] for v in elements}
        for idx, t in enumerate(triples):
            for v in t:
                elem_to_triples[v].append(idx)
                
        while active_triples:
            if time.time() > deadline:
                break
                
            # Find element with max conflict count
            # Only consider elements in S
            max_count = -1
            best_v = None
            # To add randomness, pick from top few
            # But for now, max count
            candidates = []
            for v in S:
                if conflict_count[v] > 0:
                    candidates.append((conflict_count[v], v))
            
            if not candidates:
                break # Should not happen if active_triples not empty
                
            # Sort by count desc
            candidates.sort(reverse=True)
            
            # Pick one
            # If we want randomness, pick from top 5
            top_k = min(5, len(candidates))
            best_v = rng.choice([v for _, v in candidates[:top_k]])
            
            # Remove best_v
            S.remove(best_v)
            
            # Update active_triples and conflict_count
            # All triples containing best_v are now inactive
            for idx in elem_to_triples[best_v]:
                if idx in active_triples:
                    active_triples.remove(idx)
                    a, b, c = triples[idx]
                    # Decrease count for other elements
                    for v in (a, b, c):
                        if v != best_v:
                            conflict_count[v] -= 1
                            
        return S
        
    best_S_V = set()
    best_size = 0
    
    # Try multiple seeds
    for seed in range(20):
        S_V = solve_greedy(V, triples, seed)
        if len(S_V) > best_size:
            best_size = len(S_V)
            best_S_V = S_V
        if time.time() > deadline:
            break
            
    S = set(U) | best_S_V
    
    # 4. Try to add back elements from V \ best_S_V
    # And maybe even from U? No, U is already all included.
    # We can try to add elements from V \ best_S_V
    
    removed_V = [v for v in V if v not in best_S_V]
    
    # Precompute conflicts for each u in removed_V
    # conflicts[u] = list of pairs (v, w) such that {u, v, w} is a triple
    # We need to check if {v, w} subset of S
    
    # Actually, we can just check validity for each u
    # For u in removed_V:
    #   Check if S U {u} is valid
    #   This means no triple {u, v, w} with v, w in S
    
    # To speed up, precompute for each u, the triples containing u
    # And for each such triple, the other two elements
    
    u_to_other_pairs = {v: [] for v in removed_V}
    for t in triples:
        a, b, c = t
        for u in (a, b, c):
            if u in u_to_other_pairs:
                others = [x for x in (a, b, c) if x != u]
                u_to_other_pairs[u].append(tuple(others))
                
    # Iterate adding
    changed = True
    while changed:
        changed = False
        for u in removed_V:
            if u in S:
                continue
            # Check if we can add u
            can_add = True
            for v, w in u_to_other_pairs[u]:
                if v in S and w in S:
                    can_add = False
                    break
            if can_add:
                S.add(u)
                changed = True
                # Optional: restart loop?
                # Since adding u might block others, or open up?
                # Adding u only adds conflicts, doesn't remove.
                # So if we can add u now, we should.
                # But adding u might prevent adding another u' later?
                # Yes. So order matters.
                # We can try to add all possible in one pass.
                # Then repeat until no more can be added.
                
    return list(S)
