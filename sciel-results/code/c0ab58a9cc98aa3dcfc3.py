"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit to be safe (90s wall-clock)
    n = 256
    
    # Build conflict graph: edge (i,j) exists if (i+1)*(j+1) % ((i+1)+(j+1)) == 0
    # Use 0-indexed internally, values are 1..256
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Convert to bitmask for fast operations
    conflict_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in conflicts[i]:
            mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Greedy initialization: start with all odds (they never conflict with each other)
    # then greedily add compatible evens in conflict-count order
    S = set(range(1, 257, 2))  # 1-indexed: all odd numbers
    
    evens = list(range(2, 257, 2))
    # Sort by conflict count ascending (less conflicting first)
    evens.sort(key=lambda e: len(conflicts[e-1]))
    
    for e in evens:
        if time.time() > deadline - 5.0:
            break
        can_add = True
        for o in S:
            if (e * o) % (e + o) == 0:
                can_add = False
                break
        if can_add:
            S.add(e)
    
    # Convert to 0-indexed set
    S0 = set(v - 1 for v in S)
    best_size = len(S0)
    best_set = set(S0)
    
    def can_add_to_set(v, S_set):
        """Check if v can be added to S_set without creating a conflict."""
        for u in S_set:
            if (1 << u) & conflict_mask[v]:
                return False
        return True
    
    def local_search(S_set, max_time):
        """Local search with 1-for-1, 1-for-2, and 3-for-k swaps."""
        nonlocal best_size, best_set
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Try to add any single element
            for x in range(n):
                if x not in S_set:
                    if can_add_to_set(x, S_set):
                        S_set.add(x)
                        improved = True
                        break
            
            if not improved:
                # Try 1-for-1 swaps: remove a, add b
                for a in list(S_set):
                    S_set.remove(a)
                    for b in range(n):
                        if b not in S_set:
                            if can_add_to_set(b, S_set):
                                S_set.add(b)
                                improved = True
                                break
                    if not improved:
                        S_set.add(a)  # restore
                if improved:
                    continue
            
            if not improved:
                # Try 1-for-2: remove a, add b and c
                for a in list(S_set):
                    S_set.remove(a)
                    found = False
                    for b in range(n):
                        if b not in S_set and not ((1 << b) & conflict_mask[a]):
                            if can_add_to_set(b, S_set):
                                S_set.add(b)
                                for c in range(n):
                                    if c not in S_set and c != b:
                                        if can_add_to_set(c, S_set):
                                            S_set.add(c)
                                            found = True
                                            improved = True
                                            break
                                if found:
                                    break
                    if not found:
                        S_set.remove(b) if b in S_set else None
                        S_set.add(a)  # restore
                if improved:
                    continue
            
            if not improved:
                # Try 3-for-k: remove 3 elements, add k elements
                S_list = list(S_set)
                for i in range(min(len(S_list), 50)):
                    for j in range(i + 1, min(len(S_list), 50)):
                        for k in range(j + 1, min(len(S_list), 50)):
                            if time.time() > max_time:
                                return S_set
                            a, b, c = S_list[i], S_list[j], S_list[k]
                            S_set.discard(a)
                            S_set.discard(b)
                            S_set.discard(c)
                            added = []
                            for x in range(n):
                                if x not in S_set:
                                    if can_add_to_set(x, S_set):
                                        S_set.add(x)
                                        added.append(x)
                            if len(added) > 3:
                                improved = True
                            else:
                                S_set.discard(added[0]) if added else None
                                S_set.discard(added[1]) if len(added) > 1 else None
                                S_set.discard(added[2]) if len(added) > 2 else None
                                S_set.add(a)
                                S_set.add(b)
                                S_set.add(c)
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
        return S_set
    
    # Run local search
    S0 = local_search(S0, deadline - 5.0)
    
    # Try multiple random restarts if time permits
    if time.time() < deadline - 15.0:
        random.seed(123)
        for restart in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Random greedy initialization
            S_restart = set()
            order = list(range(n))
            random.shuffle(order)
            for v in order:
                if can_add_to_set(v, S_restart):
                    S_restart.add(v)
            
            S_restart = local_search(S_restart, deadline - 5.0)
    
    # Convert back to 1-indexed sorted list
    result = sorted(v + 1 for v in best_set)
    
    # Verify validity
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                # Fallback to greedy set if invalid
                return sorted(S)
    
    return result
# EVOLVE-BLOCK-END
