import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 80
    
    N = 180
    
    # Precompute all forbidden triples (a, b, c) distinct in [1, N] with 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c => (b-a)*c = a*b => c = a*b/(b-a) for b > a
    # Or equivalently: a = b*c/(b+c)
    # We'll enumerate all valid triples
    triples = []
    seen = set()
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    t = tuple(sorted([a, b, c]))
                    if t not in seen:
                        seen.add(t)
                        triples.append(t)
    
    # Build adjacency: for each element, list of triple indices containing it
    element_triples = defaultdict(list)
    for i, t in enumerate(triples):
        for x in t:
            element_triples[x].append(i)
    
    # Count how many triples each element participates in
    triple_count = [0] * (N + 1)
    for t in triples:
        for x in t:
            triple_count[x] += 1
    
    def is_valid(S):
        S_set = set(S)
        for t in triples:
            if all(x in S_set for x in t):
                return False
        return True
    
    def count_violations(S):
        S_set = set(S)
        count = 0
        for t in triples:
            if all(x in S_set for x in t):
                count += 1
        return count
    
    # Strategy 1: Greedy removal from full set, removing element in most violated triples
    S = set(range(1, N + 1))
    
    # Find all violated triples
    violated = set()
    for i, t in enumerate(triples):
        if all(x in S for x in t):
            violated.add(i)
    
    while violated and time.time() < deadline - 5:
        elem_viol = [0] * (N + 1)
        for i in violated:
            t = triples[i]
            for x in t:
                if x in S:
                    elem_viol[x] += 1
        
        max_count = 0
        best_elem = -1
        for x in range(1, N + 1):
            if x in S and elem_viol[x] > max_count:
                max_count = elem_viol[x]
                best_elem = x
        
        if best_elem == -1:
            break
        
        S.discard(best_elem)
        
        violated = set()
        for i, t in enumerate(triples):
            if all(x in S for x in t):
                violated.add(i)
    
    best_set = set(S)
    best_size = len(S)
    
    # Try to add back elements that don't create violations
    removed = set(range(1, N + 1)) - S
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in list(removed):
            if time.time() > deadline - 5:
                break
            ok = True
            for i in element_triples[x]:
                t = triples[i]
                if all(y in S for y in t if y != x):
                    ok = False
                    break
            if ok:
                S.add(x)
                removed.discard(x)
                improved = True
                best_size = len(S)
    
    # Strategy 2: Randomized greedy construction
    random.seed(12345)
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        S2 = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if time.time() > deadline - 10:
                break
            ok = True
            for i in element_triples[x]:
                t = triples[i]
                if all(y in S2 for y in t if y != x):
                    ok = False
                    break
            if ok:
                S2.add(x)
        
        # Try to add back
        remaining = set(range(1, N + 1)) - S2
        improved2 = True
        while improved2 and time.time() < deadline - 10:
            improved2 = False
            for x in list(remaining):
                if time.time() > deadline - 10:
                    break
                ok = True
                for i in element_triples[x]:
                    t = triples[i]
                    if all(y in S2 for y in t if y != x):
                        ok = False
                        break
                if ok:
                    S2.add(x)
                    remaining.discard(x)
                    improved2 = True
        
        if len(S2) > best_size:
            best_set = set(S2)
            best_size = len(S2)
    
    # Strategy 3: Weighted greedy - prefer elements with fewer constraints
    for restart in range(100):
        if time.time() > deadline - 10:
            break
        
        S3 = set()
        # Start with elements having lowest triple count, with some randomness
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        candidates.sort(key=lambda x: (triple_count[x], random.random()))
        
        for x in candidates:
            if time.time() > deadline - 10:
                break
            ok = True
            for i in element_triples[x]:
                t = triples[i]
                if all(y in S3 for y in t if y != x):
                    ok = False
                    break
            if ok:
                S3.add(x)
        
        remaining = set(range(1, N + 1)) - S3
        improved3 = True
        while improved3 and time.time() < deadline - 10:
            improved3 = False
            for x in list(remaining):
                if time.time() > deadline - 10:
                    break
                ok = True
                for i in element_triples[x]:
                    t = triples[i]
                    if all(y in S3 for y in t if y != x):
                        ok = False
                        break
                if ok:
                    S3.add(x)
                    remaining.discard(x)
                    improved3 = True
        
        if len(S3) > best_size:
            best_set = set(S3)
            best_size = len(S3)
    
    # Strategy 4: Hill-climbing local search with single add/remove moves
    if time.time() < deadline - 15:
        current_set = set(best_set)
        
        # Precompute for each element the set of "other elements" it conflicts with
        # An element x conflicts with pair (y,z) if {x,y,z} forms a forbidden triple
        # For local search: removing x frees up all triples containing x
        # Adding x is valid iff no triple containing x has both other elements in S
        
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            
            # Try adding each removed element
            removed_elems = [x for x in range(1, N + 1) if x not in current_set]
            for x in removed_elems:
                if time.time() > deadline - 15:
                    break
                ok = True
                for i in element_triples[x]:
                    t = triples[i]
                    if all(y in current_set for y in t if y != x):
                        ok = False
                        break
                if ok:
                    current_set.add(x)
                    improved = True
            
            if not improved:
                break
            
            # Try removing each element and see if we can add more back
            # Only do this if we haven't gained size yet in this round
            if not improved:
                break
            
            # Now try: remove one element, then greedily add back as many as possible
            # This is a 1-removal + multi-add move
            for x in list(current_set):
                if time.time() > deadline - 15:
                    break
                
                # Remove x
                current_set.discard(x)
                
                # Try to add back any removed elements (including x)
                removed_after = [y for y in range(1, N + 1) if y not in current_set]
                added_count = 0
                for y in removed_after:
                    if time.time() > deadline - 15:
                        break
                    ok = True
                    for i in element_triples[y]:
                        t = triples[i]
                        if all(z in current_set for z in t if z != y):
                            ok = False
                            break
                    if ok:
                        current_set.add(y)
                        added_count += 1
                
                # Net change: removed 1, added added_count
                # If added_count >= 1, we didn't lose size (or gained if >1)
                # But we might have swapped: removed x, added different elements
                # Check if size improved
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                    improved = True
                elif len(current_set) == best_size:
                    # Same size but different composition - keep for diversity
                    # Only accept if it allows further improvement potential
                    pass
                else:
                    # Size decreased - revert this specific move
                    # We need to undo: remove the newly added elements, re-add x
                    # Simpler: just revert to previous state
                    # But we already modified current_set... let's track
                    # Actually, let's just not do this complex revert
                    # Instead, only do removals if we can immediately gain
                    pass
        
    # Strategy 5: Final polish - iterate add-then-remove-then-add until stable
    if time.time() < deadline - 5:
        for polish_round in range(50):
            if time.time() > deadline - 5:
                break
            
            # Phase 1: Add all possible
            changed = False
            for x in range(1, N + 1):
                if x not in best_set:
                    ok = True
                    for i in element_triples[x]:
                        t = triples[i]
                        if all(y in best_set for y in t if y != x):
                            ok = False
                            break
                    if ok:
                        best_set.add(x)
                        changed = True
            
            # Phase 2: Try removing each element and re-adding
            for x in list(best_set):
                if time.time() > deadline - 5:
                    break
                best_set.discard(x)
                # Try to add back everything possible
                for y in range(1, N + 1):
                    if y not in best_set:
                        ok = True
                        for i in element_triples[y]:
                            t = triples[i]
                            if all(z in best_set for z in t if z != y):
                                ok = False
                                break
                        if ok:
                            best_set.add(y)
                            changed = True
                # If we didn't get x back or gained others, keep state
                # If size decreased, we need to check
                if len(best_set) < best_size:
                    # Revert: this is tricky. Let's just track best
                    pass
            
            if not changed:
                break
    
    return list(best_set)
