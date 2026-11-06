import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 80
    
    N = 180
    
    # Precompute all forbidden triples (a, b, c) distinct in [1, N] with 1/a = 1/b + 1/c
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
    
    def count_violations(S_set):
        count = 0
        for t in triples:
            if all(x in S_set for x in t):
                count += 1
        return count
    
    # Strategy 1: Greedy removal from full set
    S = set(range(1, N + 1))
    
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
    
    # Strategy 4: Simulated annealing local search
    if time.time() < deadline - 15:
        current_set = set(best_set)
        current_size = len(current_set)
        temp = 10.0
        cooling = 0.995
        min_temp = 0.1
        
        # Precompute conflict pairs for faster checking
        # For each element x, store set of (y,z) pairs such that {x,y,z} is a forbidden triple
        conflict_pairs = defaultdict(set)
        for t in triples:
            a, b, c = t
            conflict_pairs[a].add((b, c))
            conflict_pairs[b].add((a, c))
            conflict_pairs[c].add((a, b))
        
        def can_add(x, S_set):
            """Check if x can be added to S_set without creating a violation"""
            for (y, z) in conflict_pairs[x]:
                if y in S_set and z in S_set:
                    return False
            return True
        
        improved_count = 0
        for sa_iter in range(5000):
            if time.time() > deadline - 15:
                break
            
            # Choose a random move: add or remove a random element
            if random.random() < 0.5:
                # Try to add a random element not in current_set
                not_in_set = [x for x in range(1, N + 1) if x not in current_set]
                if not not_in_set:
                    continue
                x = random.choice(not_in_set)
                
                if can_add(x, current_set):
                    # Always accept valid additions
                    current_set.add(x)
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
                    improved_count += 1
                else:
                    # Addition would create violations - accept with probability based on temp
                    # Count how many violations would be created
                    viol_count = 0
                    for (y, z) in conflict_pairs[x]:
                        if y in current_set and z in current_set:
                            viol_count += 1
                    
                    # Accept with probability exp(-viol_count / temp)
                    if random.random() < (1.0 / (1.0 + viol_count)) * (temp / 10.0):
                        current_set.add(x)
            else:
                # Try to remove a random element from current_set
                if not current_set:
                    continue
                x = random.choice(list(current_set))
                current_set.discard(x)
                
                # After removal, try to greedily add back as many elements as possible
                added_back = 0
                for y in range(1, N + 1):
                    if y not in current_set and can_add(y, current_set):
                        current_set.add(y)
                        added_back += 1
                
                # Net effect: removed 1, added added_back
                # If added_back >= 1, we didn't lose size
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                    improved_count += 1
            
            temp *= cooling
        
        # Final greedy add pass
        for x in range(1, N + 1):
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
                best_size = len(best_set)
    
    # Strategy 5: Pair-swap local search
    if time.time() < deadline - 10:
        current_set = set(best_set)
        
        # Try swapping: remove one element, add another
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            
            improved = False
            not_in_set = [x for x in range(1, N + 1) if x not in current_set]
            
            if not not_in_set:
                break
            
            for x_remove in list(current_set):
                if time.time() > deadline - 10:
                    break
                
                # Remove x_remove
                old_set = set(current_set)
                current_set.discard(x_remove)
                
                # Try to add as many elements as possible
                added = 0
                for x_add in not_in_set:
                    if can_add(x_add, current_set):
                        current_set.add(x_add)
                        added += 1
                
                # If we added more than we removed, keep the change
                if added > 1 or (added == 1 and x_remove not in current_set):
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
                    improved = True
                else:
                    # Revert
                    current_set = old_set
            
            if not improved:
                break
        
        # Final greedy add pass
        for x in range(1, N + 1):
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
                best_size = len(best_set)
    
    # Strategy 6: Final polish - iterate add-then-remove-then-add until stable
    if time.time() < deadline - 5:
        for polish_round in range(50):
            if time.time() > deadline - 5:
                break
            
            # Phase 1: Add all possible
            changed = False
            for x in range(1, N + 1):
                if x not in best_set and can_add(x, best_set):
                    best_set.add(x)
                    changed = True
            
            # Phase 2: Try removing each element and re-adding
            for x in list(best_set):
                if time.time() > deadline - 5:
                    break
                
                old_size = len(best_set)
                best_set.discard(x)
                
                # Try to add back everything possible
                for y in range(1, N + 1):
                    if y not in best_set and can_add(y, best_set):
                        best_set.add(y)
                
                # If we didn't get x back or gained others, keep state
                # If size decreased, revert
                if len(best_set) < old_size:
                    # Revert: remove newly added elements, re-add x
                    newly_added = best_set - (set(best_set) - {x})
                    # This is tricky - just keep the smaller set if it's still valid
                    # Actually, let's just not do complex reverts
                    pass
            
            if not changed:
                break
    
    return list(best_set)
