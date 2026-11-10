import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    
    # Precompute conflicts
    # conflicts[i] is a set of j such that i and j conflict
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            x = i // g
            y = j // g
            if g % (x + y) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Initial set: all odds
    S = set(range(1, N + 1, 2))
    
    # List of evens not in S
    evens = [i for i in range(2, N + 1, 2) if i not in S]
    
    # Greedy addition of non-conflicting evens
    # We repeatedly scan and add any even that has no conflict with current S
    # To speed up, we can just iterate until no change
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate over a copy since we modify evens
        for e in list(evens):
            if time.time() > deadline:
                break
            # Check if e conflicts with any element in S
            # conflicts[e] contains all numbers that conflict with e
            # We need conflicts[e] to be disjoint from S
            if not conflicts[e].intersection(S):
                S.add(e)
                evens.remove(e)
                changed = True
                # Optional: break and restart loop to handle new conflicts?
                # No, adding e might create conflicts for other evens, but we check against S dynamically.
                # So we can continue scanning.
    
    # Local Search: try to improve by swapping elements
    # Strategy: try removing one element from S and see if we can add multiple evens
    # that were previously blocked by that element.
    
    def try_local_search(S, evens, deadline):
        """Try to improve S by local search (swaps)."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try each element in S for removal
            for s_elem in list(S):
                if time.time() > deadline:
                    return S, evens
                
                # If we remove s_elem, which evens become available?
                # An even e is available if:
                # 1. e is not in S
                # 2. e doesn't conflict with any element in S \ {s_elem}
                # 3. e doesn't conflict with s_elem (we're removing it, so this is fine)
                
                # Find all evens that only conflicted with s_elem (among elements in S)
                # i.e., conflicts[e] & (S - {s_elem}) == empty
                removable = S - {s_elem}
                
                # Find evens that are now free
                free_evens = []
                for e in evens:
                    if e in S:
                        continue
                    # Check if e conflicts with any element in removable
                    if not conflicts[e].intersection(removable):
                        free_evens.append(e)
                
                # Try to greedily add as many free evens as possible
                # Sort by number of conflicts to prioritize "easy" ones
                free_evens.sort(key=lambda x: len(conflicts[x]))
                
                new_S = set(removable)
                new_evens = list(evens)
                added = 0
                for e in free_evens:
                    if time.time() > deadline:
                        break
                    if e in new_S:
                        continue
                    if not conflicts[e].intersection(new_S):
                        new_S.add(e)
                        added += 1
                
                # If we added more elements than we removed (1), it's an improvement
                if added > 1:
                    S = new_S
                    evens = [e for e in evens if e not in new_S]
                    improved = True
                    break  # Restart the loop with the new S
        
        return S, evens
    
    S, evens = try_local_search(S, evens, deadline)
    
    # Second pass: try 1-for-k swaps more aggressively
    # Also try removing pairs of elements to free up more evens
    def try_pair_removal(S, evens, deadline):
        """Try removing pairs of elements to free up more evens."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each pair of elements
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline:
                    return S, evens
                for j in range(i + 1, len(S_list)):
                    if time.time() > deadline:
                        return S, evens
                    
                    s1, s2 = S_list[i], S_list[j]
                    removable = S - {s1, s2}
                    
                    # Find evens that are now free
                    free_evens = []
                    for e in evens:
                        if e in S:
                            continue
                        if not conflicts[e].intersection(removable):
                            free_evens.append(e)
                    
                    # Try to greedily add as many free evens as possible
                    free_evens.sort(key=lambda x: len(conflicts[x]))
                    
                    new_S = set(removable)
                    added = 0
                    for e in free_evens:
                        if time.time() > deadline:
                            break
                        if e in new_S:
                            continue
                        if not conflicts[e].intersection(new_S):
                            new_S.add(e)
                            added += 1
                    
                    # If we added more elements than we removed (2), it's an improvement
                    if added > 2:
                        S = new_S
                        evens = [e for e in evens if e not in new_S]
                        improved = True
                        break
                if improved:
                    break
        
        return S, evens
    
    S, evens = try_pair_removal(S, evens, deadline)
    
    # Final greedy pass after local search
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for e in list(evens):
            if time.time() > deadline:
                break
            if not conflicts[e].intersection(S):
                S.add(e)
                evens.remove(e)
                changed = True
    
    return list(S)
# EVOLVE-BLOCK-END
