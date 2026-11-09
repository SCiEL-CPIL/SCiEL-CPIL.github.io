import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 88.0
    n = 180
    
    # Precompute triples (a,b,c) such that 1/a = 1/b + 1/c
    # a < b < c
    # (b-a)(c-a) = a^2
    triples = []
    for a in range(1, n + 1):
        a2 = a * a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d < e:
                    b = a + d
                    c = a + e
                    if c <= n:
                        triples.append((a, b, c))
            d += 1
    
    # Incidence list: for each index i (0 to 179, value i+1),
    # store list of tuples (j, k) such that (i+1, j+1, k+1) is a triple.
    conflicts = [[] for _ in range(n)]
    for (a, b, c) in triples:
        conflicts[a - 1].append((b - 1, c - 1))
        conflicts[b - 1].append((a - 1, c - 1))
        conflicts[c - 1].append((a - 1, b - 1))
        
    def try_add(S, a):
        """Try to add element a (0-indexed) to set S. Returns True if successful."""
        for (u, v) in conflicts[a]:
            if S[u] and S[v]:
                return False
        S[a] = True
        return True

    def local_search_aggressive(S, deadline):
        """Aggressive local search: greedy add, then swap-based improvements."""
        cur_size = sum(S)
        
        # Phase 1: Greedy add all possible
        changed = True
        while changed:
            changed = False
            for a in range(n):
                if not S[a]:
                    if try_add(S, a):
                        cur_size += 1
                        changed = True
        
        # Phase 2: Try removing one element and adding multiple others
        improved = True
        while improved and time.time() < deadline:
            improved = False
            present = [i for i in range(n) if S[i]]
            random.shuffle(present)
            for a in present:
                if time.time() > deadline:
                    break
                # Temporarily remove a
                S[a] = False
                cur_size -= 1
                # Try to add all missing elements
                missing = [i for i in range(n) if not S[i]]
                random.shuffle(missing)
                added_count = 0
                added_list = []
                for b in missing:
                    if try_add(S, b):
                        added_count += 1
                        added_list.append(b)
                # If we added more than we removed, keep the change
                if added_count > 1:
                    cur_size += added_count
                    improved = True
                else:
                    # Restore a
                    S[a] = True
                    cur_size += 1
                    # Undo the additions
                    for b in added_list:
                        S[b] = False
                        cur_size -= 1
        
        # Phase 3: Single swap - remove one, try to add a different one
        improved = True
        while improved and time.time() < deadline:
            improved = False
            present = [i for i in range(n) if S[i]]
            random.shuffle(present)
            for a in present:
                if time.time() > deadline:
                    break
                missing = [i for i in range(n) if not S[i]]
                random.shuffle(missing)
                for b in missing:
                    # Check if swapping a out for b is valid
                    # First check if b can be added if a is removed
                    S[a] = False
                    can_add_b = True
                    for (u, v) in conflicts[b]:
                        if S[u] and S[v]:
                            can_add_b = False
                            break
                    if can_add_b:
                        S[a] = True
                        S[b] = True
                        cur_size += 1  # net +1 since we added b
                        improved = True
                        break
                    else:
                        S[a] = True
        
        # Phase 4: Final greedy add
        for a in range(n):
            if not S[a]:
                try_add(S, a)
        
        return sum(S)

    best_set = None
    best_size = 0
    
    # Strategy 1: Suffix + Greedy Add
    for K in range(1, n + 1):
        if time.time() - t0 > DEADLINE - 20:
            break
            
        S = [False] * n
        for i in range(K - 1, n):
            S[i] = True
            
        # Check if this suffix is valid on its own
        valid_suffix = True
        for (a, b, c) in triples:
            if a >= K and b >= K and c >= K:
                valid_suffix = False
                break
        
        if not valid_suffix:
            continue
            
        cur_size = n - K + 1
        if cur_size > best_size:
            best_size = cur_size
            best_set = S[:]
            
        # Greedy add smaller numbers from K-1 down to 1
        for a in range(K - 1, 0, -1):
            if S[a - 1]:
                continue
            can_add = True
            for (u, v) in conflicts[a - 1]:
                if S[u] and S[v]:
                    can_add = False
                    break
                    
            if can_add:
                S[a - 1] = True
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = S[:]
                    
    # Strategy 2: Greedy vertex cover approach
    # Start with all elements, remove those that break the most triples
    if time.time() - t0 < DEADLINE - 15:
        S = [True] * n
        cur_size = n
        
        # Compute for each element how many triples it's in
        triple_count = [0] * n
        for (a, b, c) in triples:
            triple_count[a - 1] += 1
            triple_count[b - 1] += 1
            triple_count[c - 1] += 1
        
        # Greedily remove elements that are in the most active triples
        # Keep removing until no triple is fully present
        improved = True
        while improved and time.time() - t0 < DEADLINE - 10:
            improved = False
            # Find the most violated triple
            max_violations = 0
            worst_elem = -1
            # For each element, count how many triples it's part of that are "almost complete"
            # i.e., the other two elements are in S
            for a in range(n):
                if not S[a]:
                    continue
                count = 0
                for (u, v) in conflicts[a]:
                    if S[u] and S[v]:
                        count += 1
                if count > max_violations:
                    max_violations = count
                    worst_elem = a
            
            if worst_elem >= 0 and max_violations > 0:
                S[worst_elem] = False
                cur_size -= 1
                improved = True
        
        # Now try to add back elements
        for a in range(n):
            if not S[a]:
                if try_add(S, a):
                    cur_size += 1
        
        if cur_size > best_size:
            best_size = cur_size
            best_set = S[:]
            
        # Run aggressive local search on this result
        if time.time() - t0 < DEADLINE - 5:
            S2 = S[:]
            cur_size2 = local_search_aggressive(S2, DEADLINE - 3)
            if cur_size2 > best_size:
                best_size = cur_size2
                best_set = S2[:]

    # Strategy 3: Local search from best set found so far
    if best_set is not None and time.time() - t0 < DEADLINE - 5:
        S2 = best_set[:]
        cur_size = local_search_aggressive(S2, DEADLINE - 3)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S2[:]

    # Strategy 4: Multiple random restarts with aggressive local search
    if time.time() - t0 < DEADLINE - 10:
        for restart in range(30):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            # Random starting point
            K = random.randint(30, 120)
            S = [False] * n
            for i in range(K - 1, n):
                S[i] = True
                
            valid = True
            for (a, b, c) in triples:
                if a >= K and b >= K and c >= K:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Greedy add
            for a in range(K - 1, 0, -1):
                if S[a - 1]:
                    continue
                can_add = True
                for (u, v) in conflicts[a - 1]:
                    if S[u] and S[v]:
                        can_add = False
                        break
                if can_add:
                    S[a - 1] = True
            
            # Aggressive local search
            cur_size = local_search_aggressive(S, DEADLINE - 3)
            
            if cur_size > best_size:
                best_size = cur_size
                best_set = S[:]

    if best_set is None:
        best_set = [False] * n
        best_set[n-1] = True
        best_size = 1

    out = [i + 1 for i in range(n) if best_set[i]]
    return out
