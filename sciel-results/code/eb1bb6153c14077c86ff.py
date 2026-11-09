import time
import random
import math
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = 180
    
    # Precompute triples (a,b,c) such that 1/a = 1/b + 1/c
    # a < b < c
    # (b-a)(c-a) = a^2
    triples = []
    for a in range(1, n + 1):
        a2 = a * a
        # Find divisors d of a2
        # d and e such that d*e = a2
        # b = a + d, c = a + e
        # We need b < c => d < e
        # b, c <= n => a + d <= n, a + e <= n
        # So d <= n - a, e <= n - a
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
            
    # Build conflict graph or list of pairs for each vertex
    # Incidence list: for each index i (0 to 179, value i+1),
    # store list of tuples (j, k) such that (i+1, j+1, k+1) is a triple.
    
    conflicts = [[] for _ in range(n)]
    for (a, b, c) in triples:
        # a is smallest
        conflicts[a - 1].append((b - 1, c - 1))
        # b is middle
        conflicts[b - 1].append((a - 1, c - 1))
        # c is largest
        conflicts[c - 1].append((a - 1, b - 1))
        
    best_set = None
    best_size = 0
    
    def try_add(S, a):
        """Try to add element a (0-indexed) to set S. Returns True if successful."""
        for (u, v) in conflicts[a]:
            if S[u] and S[v]:
                return False
        S[a] = True
        return True

    def local_search(S, max_iters=10000):
        """Improve set S using greedy local search."""
        cur_size = sum(S)
        improved = True
        iter_count = 0
        while improved and cur_size > 0 and iter_count < max_iters:
            improved = False
            iter_count += 1
            # Try to add each missing element in random order
            missing = [i for i in range(n) if not S[i]]
            random.shuffle(missing)
            for a in missing:
                if try_add(S, a):
                    cur_size += 1
                    improved = True
            # Try to remove and re-add elements to escape local optima
            present = [i for i in range(n) if S[i]]
            random.shuffle(present)
            for a in present:
                # Try removing a and see if we can add more
                S[a] = False
                cur_size -= 1
                # Try to add all missing elements
                missing2 = [i for i in range(n) if not S[i]]
                random.shuffle(missing2)
                added = 0
                for b in missing2:
                    if try_add(S, b):
                        added += 1
                # If we added more than we removed, keep the change
                if added > 1:  # net gain
                    cur_size += added
                    improved = True
                else:
                    # Restore a
                    try_add(S, a)
                    cur_size += 1
        return cur_size

    def greedy_hitting_set(triples, n):
        """Find a small set R to remove such that no triple is fully in [1,n]\R."""
        # Map element -> list of triple indices
        elem_to_triples = {i: [] for i in range(1, n+1)}
        for idx, (a,b,c) in enumerate(triples):
            elem_to_triples[a].append(idx)
            elem_to_triples[b].append(idx)
            elem_to_triples[c].append(idx)
            
        active = set(range(len(triples)))
        R = set()
        
        while active:
            best_elem = -1
            best_count = -1
            
            # Find element with max degree in active triples
            for x in range(1, n+1):
                if x in R:
                    continue
                count = 0
                for t_idx in elem_to_triples[x]:
                    if t_idx in active:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_elem = x
            
            if best_elem == -1:
                break
                
            R.add(best_elem)
            
            # Remove all triples containing best_elem from active
            to_remove = set()
            for t_idx in elem_to_triples[best_elem]:
                if t_idx in active:
                    to_remove.add(t_idx)
            active -= to_remove
            
        # Construct S
        S = [False] * n
        for i in range(1, n+1):
            if i not in R:
                S[i-1] = True
        return S

    def csp_backtracking(S, remaining_elements, current_size, deadline):
        """
        CSP-style backtracking solver with dynamic variable ordering and forward checking.
        S: current assignment (list of bools)
        remaining_elements: list of elements not yet decided
        current_size: current number of elements in S
        deadline: time limit
        Returns: (best_size, best_set) found
        """
        nonlocal best_size, best_set
        
        # Base case: no more elements to decide
        if not remaining_elements:
            if current_size > best_size:
                best_size = current_size
                best_set = S[:]
            return current_size, S[:]
        
        # Check time
        if time.time() > deadline:
            return current_size, S[:]
        
        # Dynamic variable ordering: select variable with fewest remaining valid values
        # For each remaining element, count how many of its conflict pairs are both in S
        # Fewer conflicts means it's "easier" to add, so we should consider harder ones first
        # Actually, we want to select the element that is most constrained
        # An element is constrained if adding it would conflict with many existing pairs
        
        # Score each remaining element: number of conflict pairs where both are in S
        # Higher score = more constrained = should be decided first
        scored = []
        for elem in remaining_elements:
            conflict_count = 0
            for (u, v) in conflicts[elem]:
                if S[u] and S[v]:
                    conflict_count += 1
            scored.append((conflict_count, elem))
        
        # Sort by conflict count descending (most constrained first)
        scored.sort(reverse=True)
        
        # Try adding the most constrained element first
        # If it can't be added, skip it (set to False)
        # If it can be added, branch: try adding it, then try not adding it
        
        best_conflict_count, first_elem = scored[0]
        
        # Option 1: Try to add first_elem
        if best_conflict_count == 0:
            # Can add it
            S[first_elem] = True
            new_remaining = [e for e in remaining_elements if e != first_elem]
            size1, set1 = csp_backtracking(S, new_remaining, current_size + 1, deadline)
            
            # Option 2: Don't add it
            S[first_elem] = False
            size2, set2 = csp_backtracking(S, new_remaining, current_size, deadline)
            
            # Restore
            S[first_elem] = False
            
            # Return the better of the two
            if size1 >= size2:
                return size1, set1
            else:
                return size2, set2
        else:
            # Cannot add it, must skip
            S[first_elem] = False
            new_remaining = [e for e in remaining_elements if e != first_elem]
            return csp_backtracking(S, new_remaining, current_size, deadline)

    def csp_with_pruning(S, remaining_elements, current_size, deadline, max_depth=50):
        """
        CSP solver with depth limit and pruning for efficiency.
        """
        nonlocal best_size, best_set
        
        if not remaining_elements or current_size <= best_size - len(remaining_elements):
            if current_size > best_size:
                best_size = current_size
                best_set = S[:]
            return current_size, S[:]
        
        if time.time() > deadline:
            return current_size, S[:]
        
        # Dynamic variable ordering
        scored = []
        for elem in remaining_elements:
            conflict_count = 0
            for (u, v) in conflicts[elem]:
                if S[u] and S[v]:
                    conflict_count += 1
            scored.append((conflict_count, elem))
        
        scored.sort(reverse=True)
        
        best_conflict_count, first_elem = scored[0]
        new_remaining = [e for e in remaining_elements if e != first_elem]
        
        if best_conflict_count == 0:
            # Can add it
            S[first_elem] = True
            size1, set1 = csp_with_pruning(S, new_remaining, current_size + 1, deadline, max_depth)
            
            # Don't add it
            S[first_elem] = False
            size2, set2 = csp_with_pruning(S, new_remaining, current_size, deadline, max_depth)
            
            S[first_elem] = False
            
            if size1 >= size2:
                return size1, set1
            else:
                return size2, set2
        else:
            # Cannot add it
            S[first_elem] = False
            return csp_with_pruning(S, new_remaining, current_size, deadline, max_depth)

    # Strategy 1: Suffix + Greedy Add
    for K in range(1, n + 1):
        if time.time() - t0 > DEADLINE - 30:
            break
            
        # Initialize S with {K, ..., n}
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
            
        # Greedy add smaller numbers
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
                    
    # Strategy 2: Greedy Hitting Set
    if time.time() - t0 < DEADLINE - 20:
        S_ghs = greedy_hitting_set(triples, n)
        cur_size = sum(S_ghs)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S_ghs[:]
            
        # Local search on GHS result
        if time.time() - t0 < DEADLINE - 10:
            S_ghs_ls = S_ghs[:]
            cur_size = local_search(S_ghs_ls)
            if cur_size > best_size:
                best_size = cur_size
                best_set = S_ghs_ls[:]

    # Strategy 3: CSP Backtracking from best set found
    if best_set is not None and time.time() - t0 < DEADLINE - 15:
        # Use CSP to try to improve the best set
        S_csp = best_set[:]
        remaining = [i for i in range(n) if not S_csp[i]]
        current_size = sum(S_csp)
        
        # Use CSP with pruning
        size_csp, set_csp = csp_with_pruning(S_csp, remaining, current_size, DEADLINE - 5, max_depth=30)
        
        if size_csp > best_size:
            best_size = size_csp
            best_set = set_csp[:]

    # Strategy 4: Local search from best set found
    if best_set is not None and time.time() - t0 < DEADLINE - 5:
        S2 = best_set[:]
        cur_size = local_search(S2)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S2[:]

    # Strategy 5: Multiple random restarts with local search
    if time.time() - t0 < DEADLINE - 10:
        for restart in range(30):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            # Random starting point: pick a random threshold
            K = random.randint(20, 140)
            S = [False] * n
            for i in range(K - 1, n):
                S[i] = True
                
            # Check validity
            valid = True
            for (a, b, c) in triples:
                if a >= K and b >= K and c >= K:
                    valid = False
                    break
            
            if not valid:
                continue
            
            cur_size = n - K + 1
            
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
                    cur_size += 1
            
            # Local search
            cur_size = local_search(S)
            
            if cur_size > best_size:
                best_size = cur_size
                best_set = S[:]

    if best_set is None:
        best_set = [False] * n
        best_set[n-1] = True # [180]
        best_size = 1

    # Final polish
    if time.time() - t0 < DEADLINE - 2:
        S_final = best_set[:]
        cur_size = local_search(S_final)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S_final[:]

    # Convert to list of values
    out = [i + 1 for i in range(n) if best_set[i]]
    
    return out
