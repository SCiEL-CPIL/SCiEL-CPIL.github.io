import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    n = 180
    
    # Generate all triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = kxy, b = kx(x+y), c = ky(x+y) with gcd(x,y)=1
    triples = []
    # x < y, gcd(x,y)=1
    # b = kx(x+y) <= 180
    # c = ky(x+y) <= 180
    # a = kxy <= 180
    
    for x in range(1, n + 1):
        for y in range(x + 1, n + 1):
            if __import__('math').gcd(x, y) != 1:
                continue
            s = x + y
            # k * y * s <= 180 => k <= 180 / (y*s)
            max_k = 180 // (y * s)
            if max_k < 1:
                continue
            for k in range(1, max_k + 1):
                a = k * x * y
                b = k * x * s
                c = k * y * s
                if a <= n and b <= n and c <= n:
                    triples.append((a, b, c))
    
    # Build adjacency: for each vertex, list of triples it is in
    vertex_to_triples = {i: [] for i in range(1, n + 1)}
    for i, (a, b, c) in enumerate(triples):
        vertex_to_triples[a].append(i)
        vertex_to_triples[b].append(i)
        vertex_to_triples[c].append(i)
    
    # Greedy Hitting Set
    # State: removed set R (boolean array), active_triples (boolean array)
    # We want to minimize |R|
    
    def greedy_hitting_set():
        removed = [False] * (n + 1)
        active = [True] * len(triples)
        
        # Precompute for each vertex the list of active triples it covers
        # This needs to be updated dynamically or we recompute counts
        
        # Count how many active triples each vertex covers
        counts = [0] * (n + 1)
        for i, (a, b, c) in enumerate(triples):
            if active[i]:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
        
        while any(active):
            # Find vertex with max count > 0
            max_count = 0
            best_v = -1
            for v in range(1, n + 1):
                if counts[v] > max_count:
                    max_count = counts[v]
                    best_v = v
            
            if best_v == -1:
                break
            
            # Remove best_v
            removed[best_v] = True
            
            # Update active and counts
            for idx in vertex_to_triples[best_v]:
                if active[idx]:
                    active[idx] = False
                    a, b, c = triples[idx]
                    for v in (a, b, c):
                        if v != best_v:
                            counts[v] -= 1
            counts[best_v] = 0 # No longer needed
            
        return removed
    
    # Run greedy multiple times with random tie-breaking?
    # Or just once, then local search.
    
    best_removed = greedy_hitting_set()
    best_size = n - sum(best_removed)
    
    # Local Search
    # Try to remove an element from R (add to selected)
    # This might violate some triples.
    # If it violates 0, accept.
    # If it violates 1, can we fix it by adding one element to R?
    # If yes, size stays same. We might accept to explore.
    # If it violates > 1, skip.
    
    # We will iterate until no improvement or time limit
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try to remove each element from R
        for u in range(1, n + 1):
            if not best_removed[u]:
                continue
            
            # Tentatively remove u
            best_removed[u] = False
            
            # Check which triples are now violated
            # A triple is violated if it contains u and no other element in R
            violated = []
            for idx in vertex_to_triples[u]:
                a, b, c = triples[idx]
                # Check if any other element in (a,b,c) is in R
                is_covered = False
                for v in (a, b, c):
                    if v != u and best_removed[v]:
                        is_covered = True
                        break
                if not is_covered:
                    violated.append(idx)
            
            if len(violated) == 0:
                # Accept, size increased
                best_size += 1
                improved = True
                continue
            
            # If len(violated) == 1, try to find a vertex v not in R that covers it
            # Adding v to R will cover this violated triple and possibly others
            # Size change: -1 (removed u) + 1 (added v) = 0
            # We accept if it leads to a better configuration? 
            # For now, let's just accept if we can swap, to allow further moves.
            # But we must ensure we don't loop.
            # Let's stick to size-increasing moves for now to be safe.
            # Or allow size-neutral moves with a limit.
            
            # Let's try size-neutral moves only if we have time and no size-increasing moves found in this pass?
            # Simpler: just accept size-increasing moves.
            
            # Revert
            best_removed[u] = True
            
            # Try to find a swap: add v to R to cover violated
            # If we add v, we cover 'violated' and any other triples containing v.
            # We must ensure that adding v doesn't break anything else? 
            # No, adding to R only removes triples from 'active' (in the hitting set sense).
            # Wait, the logic is:
            # R is the set of removed vertices.
            # Validity: No triple has all 3 vertices NOT in R.
            # Equivalently: Every triple has at least 1 vertex in R.
            
            # When we remove u from R, some triples might lose their last cover (u).
            # Those are 'violated'.
            # To fix, we must add some vertices to R to cover them.
            # If we add 1 vertex v, and v covers all 'violated' triples, we can swap.
            
            if len(violated) == 1:
                idx = violated[0]
                a, b, c = triples[idx]
                # Candidates for v are a, b, c (excluding u)
                # Also, v can be any vertex that is in this triple? No, v must be in the triple to cover it.
                # So v must be one of the other two elements in the triple.
                candidates = [v for v in (a, b, c) if v != u]
                
                # Check if adding any candidate v to R is valid (it is, by definition of hitting set)
                # But we must check if v is already in R? 
                # If v is in R, then the triple was covered by v, so it wouldn't be violated.
                # So candidates are not in R.
                
                for v in candidates:
                    if not best_removed[v]:
                        # Swap u out, v in
                        # This keeps size same.
                        # Do we want to do this?
                        # It might allow future size-increasing moves.
                        # Let's do it with a small probability or if it reduces "potential conflicts"?
                        # For simplicity, let's skip size-neutral swaps to avoid complexity/loops.
                        pass
            
            # Revert is already done above? No, I reverted immediately after checking len(violated)==0.
            # I need to structure the code better.

    # The loop above was flawed. Let's rewrite the local search properly.
    
    # Reset to best_removed from greedy
    # Actually, let's just run a few iterations of the greedy with randomization.
    
    # Better approach:
    # 1. Greedy to get R.
    # 2. Repeat:
    #    - Try to remove each u in R.
    #    - If it causes 0 violations, accept.
    #    - If it causes 1 violation, try to swap with a candidate v.
    #      If swap is possible, accept (size same, but structure changed).
    #    - If no changes in a full pass, break.
    
    # Let's implement this robustly.
    
    def local_search(removed):
        size = n - sum(removed)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try to remove elements
            for u in range(1, n + 1):
                if not removed[u]:
                    continue
                
                # Tentatively remove u
                removed[u] = False
                
                violated = []
                for idx in vertex_to_triples[u]:
                    a, b, c = triples[idx]
                    covered = False
                    for v in (a, b, c):
                        if v != u and removed[v]:
                            covered = True
                            break
                    if not covered:
                        violated.append(idx)
                
                if len(violated) == 0:
                    # Accept
                    size += 1
                    improved = True
                    continue
                
                # Try swap if 1 violation
                if len(violated) == 1:
                    idx = violated[0]
                    a, b, c = triples[idx]
                    candidates = [v for v in (a, b, c) if v != u and not removed[v]]
                    # Any candidate works to cover this triple.
                    # We pick one randomly or the one that covers most other "potential" violations?
                    # For now, pick the first one.
                    if candidates:
                        v = candidates[0]
                        # Swap
                        removed[v] = True
                        # Size remains same
                        # Mark improved to continue search
                        improved = True
                        # Note: we don't revert u, we keep u removed and v added.
                        # This is a valid move.
                        continue
                
                # Revert if not accepted
                removed[u] = True
        
        return size

    # Run greedy
    removed = greedy_hitting_set()
    size = local_search(removed)
    
    # Try a few random restarts if time permits
    # The greedy step is deterministic. We can randomize the tie-breaking.
    
    best_size = size
    best_removed = removed[:]
    
    # Randomized Greedy
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
        
        # Randomized Greedy
        removed_rand = [False] * (n + 1)
        active = [True] * len(triples)
        counts = [0] * (n + 1)
        for i, (a, b, c) in enumerate(triples):
            if active[i]:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
        
        while any(active):
            max_count = 0
            best_candidates = []
            for v in range(1, n + 1):
                if counts[v] > max_count:
                    max_count = counts[v]
                    best_candidates = [v]
                elif counts[v] == max_count and max_count > 0:
                    best_candidates.append(v)
            
            if not best_candidates:
                break
            
            best_v = random.choice(best_candidates)
            removed_rand[best_v] = True
            
            for idx in vertex_to_triples[best_v]:
                if active[idx]:
                    active[idx] = False
                    a, b, c = triples[idx]
                    for v in (a, b, c):
                        if v != best_v:
                            counts[v] -= 1
            counts[best_v] = 0
        
        size_rand = local_search(removed_rand)
        if size_rand > best_size:
            best_size = size_rand
            best_removed = removed_rand[:]
    
    # Final check and return
    # Construct the set
    selected = [i for i in range(1, n + 1) if not best_removed[i]]
    
    # Verify
    for a, b, c in triples:
        if a in selected and b in selected and c in selected:
            # This shouldn't happen if logic is correct
            # But if it does, remove one
            selected.remove(c)
    
    return sorted(selected)

