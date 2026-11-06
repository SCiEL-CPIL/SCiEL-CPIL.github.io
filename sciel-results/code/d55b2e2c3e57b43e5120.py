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
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                    improved = True
        
        # Update best if current improved
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
    
    # Strategy 5: Simulated Annealing local search
    # Objective: maximize |S| subject to no forbidden triples
    # We use a penalty-based approach: fitness = |S| - penalty * (#violations)
    # SA accepts worse solutions with probability exp(-delta/T)
    
    if time.time() < deadline - 5:
        # Start from best_set
        sa_set = set(best_set)
        best_sa_set = set(best_set)
        best_sa_size = len(best_set)
        
        # Parameters
        T_start = 5.0
        T_end = 0.01
        cooling_rate = 0.999
        
        # Compute current fitness
        def get_fitness(S):
            S_set = set(S)
            viol = 0
            for t in triples:
                if all(x in S_set for x in t):
                    viol += 1
            return len(S) - 2 * viol
        
        current_fitness = get_fitness(sa_set)
        best_fitness = current_fitness
        
        T = T_start
        max_steps = 200000
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            in_set = [x for x in range(1, N + 1) if x in sa_set]
            out_set = [x for x in range(1, N + 1) if x not in sa_set]
            
            if move_type < 0.4 and out_set:
                # Add a random element from outside
                x = random.choice(out_set)
                sa_set.add(x)
                new_fitness = get_fitness(sa_set)
                delta = new_fitness - current_fitness
                if delta >= 0 or random.random() < pow(2.718281828, -delta / max(T, 0.001)):
                    current_fitness = new_fitness
                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        best_sa_set = set(sa_set)
                        if len(best_sa_set) > best_sa_size:
                            best_sa_size = len(best_sa_set)
                else:
                    sa_set.discard(x)
                    
            elif move_type < 0.7 and in_set:
                # Remove a random element from inside
                x = random.choice(in_set)
                sa_set.discard(x)
                new_fitness = get_fitness(sa_set)
                delta = new_fitness - current_fitness
                if delta >= 0 or random.random() < pow(2.718281828, -delta / max(T, 0.001)):
                    current_fitness = new_fitness
                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        best_sa_set = set(sa_set)
                        if len(best_sa_set) > best_sa_size:
                            best_sa_size = len(best_sa_set)
                else:
                    sa_set.add(x)
                    
            else:
                # Swap: remove one from in, add one from out
                if in_set and out_set:
                    x_in = random.choice(in_set)
                    x_out = random.choice(out_set)
                    sa_set.discard(x_in)
                    sa_set.add(x_out)
                    new_fitness = get_fitness(sa_set)
                    delta = new_fitness - current_fitness
                    if delta >= 0 or random.random() < pow(2.718281828, -delta / max(T, 0.001)):
                        current_fitness = new_fitness
                        if current_fitness > best_fitness:
                            best_fitness = current_fitness
                            best_sa_set = set(sa_set)
                            if len(best_sa_set) > best_sa_size:
                                best_sa_size = len(best_sa_set)
                    else:
                        sa_set.discard(x_out)
                        sa_set.add(x_in)
            
            # Cool down
            T *= cooling_rate
        
        # Update global best
        if len(best_sa_set) > best_size:
            best_set = set(best_sa_set)
            best_size = len(best_set)
    
    # Strategy 6: Final polish - iterate add-then-remove-then-add until stable
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
                # If size decreased, we need to check
                if len(best_set) < best_size:
                    pass
            
            if not changed:
                break
    
    return list(best_set)

