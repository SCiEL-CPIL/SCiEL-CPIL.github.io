"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random

def _triple_sums_with_new(x, S):
    """Compute all triple sums that include x, given existing set S."""
    sums = []
    n = len(S)
    sums.append(3 * x)
    for s in S:
        sums.append(2 * x + s)
        sums.append(x + 2 * s)
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + S[i] + S[j])
    return sums

def _all_triple_sums(S):
    """Compute all triple sums for set S."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S[i] + S[j] + S[k])
    return sums

def _is_valid_triples(S):
    """Check if all unordered triple sums (with repetition) are unique."""
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                if s in seen:
                    return False
                seen.add(s)
    return True

def _greedy_linear(start=1, end=4096, deadline=None):
    """Linear greedy scan."""
    selected = []
    sums_set = set()
    for x in range(start, end + 1):
        if deadline and time.time() > deadline:
            break
        new_sums = _triple_sums_with_new(x, selected)
        if not any(s in sums_set for s in new_sums):
            selected.append(x)
            sums_set.update(new_sums)
    return selected, sums_set

def _greedy_degree_pruned(start=1, end=4096, deadline=None):
    """
    Spectral graph approach: build conflict graph, then greedily select
    nodes with fewest conflicts (minimum degree first) to maximize packing density.
    """
    if deadline and time.time() > deadline:
        return [], set()
    
    # Phase 1: Greedy build to get a good initial set
    selected, sums_set = _greedy_linear(start, end, deadline)
    if not selected:
        return selected, sums_set
    
    # Phase 2: Try to add more using degree-based heuristic
    # For each candidate, count how many existing sums it conflicts with
    current = list(selected)
    current_sums = set(sums_set)
    
    if deadline and time.time() > deadline:
        return current, current_sums
    
    # Repeatedly try to add the candidate with minimum conflict count
    improved = True
    while improved and not (deadline and time.time() > deadline):
        improved = False
        best_candidate = None
        best_conflicts = float('inf')
        best_new_sums = None
        
        for x in range(1, end + 1):
            if x in current:
                continue
            new_sums = _triple_sums_with_new(x, current)
            conflicts = sum(1 for s in new_sums if s in current_sums)
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_candidate = x
                best_new_sums = new_sums
            if conflicts == 0:
                break  # Can't do better than 0 conflicts
        
        if best_candidate is not None and best_conflicts == 0:
            current.append(best_candidate)
            current_sums.update(best_new_sums)
            improved = True
    
    return current, current_sums

def _local_search_improve(selected, sums_set, deadline=None):
    """
    Local search: try swapping elements to enable adding more.
    Uses degree-based pruning: prefer swaps that reduce total conflict count.
    """
    if deadline and time.time() > deadline:
        return selected
    
    current = list(selected)
    current_sums = set(sums_set)
    best_size = len(current)
    
    for _ in range(200):
        if deadline and time.time() > deadline:
            break
        if len(current) >= 80:
            break
        
        # Try to add any valid element
        found_add = False
        for x in range(1, 4097):
            if x in current:
                continue
            new_sums = _triple_sums_with_new(x, current)
            if not any(s in current_sums for s in new_sums):
                current.append(x)
                current_sums.update(new_sums)
                found_add = True
                break
        
        if found_add:
            continue
        
        # No direct addition possible; try a swap
        best_swap = None
        best_swap_score = -1
        
        # Sample a few positions to try swapping
        if len(current) >= 2:
            for _sample in range(min(20, len(current))):
                pos = random.randint(0, len(current) - 1)
                removed = current[pos]
                
                # Compute sums without this element
                remaining = current[:pos] + current[pos+1:]
                rem_sums = _all_triple_sums(remaining)
                
                # Try adding different elements
                for x in range(1, 4097):
                    if x in remaining:
                        continue
                    new_sums = _triple_sums_with_new(x, remaining)
                    conflicts = sum(1 for s in new_sums if s in rem_sums)
                    if conflicts == 0:
                        # Valid swap
                        # Score: prefer larger x to leave room for smaller
                        score = x
                        if score > best_swap_score:
                            best_swap_score = score
                            best_swap = (pos, removed, x)
                    if best_swap_score > 4000:
                        break
                
                if best_swap_score > 4000:
                    break
        
        if best_swap:
            pos, removed, new_elem = best_swap
            current[pos] = new_elem
            # Recompute sums
            current_sums = _all_triple_sums(current)
            best_size = max(best_size, len(current))
    
    return current

def _backtrack_with_bytearray(deadline):
    """
    Recursive backtracking using bytearray for O(1) sum lookups.
    Explores smallest valid numbers first (greedy path) for fast pruning.
    """
    MAX_SUM = 3 * 4096  # Maximum possible triple sum
    
    def backtrack(selected, sums_bytearray, target_size, start_val):
        if len(selected) >= target_size:
            return list(selected)
        if time.time() > deadline:
            return list(selected)
        
        for x in range(start_val, 4097):
            if time.time() > deadline:
                break
            
            # Check if x can be added
            new_sums = _triple_sums_with_new(x, selected)
            valid = True
            for s in new_sums:
                if s <= MAX_SUM and sums_bytearray[s]:
                    valid = False
                    break
            
            if valid:
                # Mark new sums
                for s in new_sums:
                    if s <= MAX_SUM:
                        sums_bytearray[s] = 1
                
                selected.append(x)
                result = backtrack(selected, sums_bytearray, target_size, x + 1)
                
                if len(result) >= target_size:
                    return result
                
                selected.pop()
                # Unmark
                for s in new_sums:
                    if s <= MAX_SUM:
                        sums_bytearray[s] = 0
        
        return list(selected)
    
    return backtrack([], bytearray(MAX_SUM + 1), 80, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best_result = []
    best_size = 0
    
    # Strategy 1: Linear greedy
    selected, sums_set = _greedy_linear(1, 4096, deadline)
    if len(selected) > best_size:
        best_size = len(selected)
        best_result = list(selected)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 2: Degree-pruned greedy (spectral approach)
    selected2, sums_set2 = _greedy_degree_pruned(1, 4096, deadline)
    if len(selected2) > best_size:
        best_size = len(selected2)
        best_result = list(selected2)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 3: Local search improvement
    current = list(best_result)
    current_sums = _all_triple_sums(current)
    improved = _local_search_improve(current, current_sums, deadline)
    if len(improved) > best_size:
        best_size = len(improved)
        best_result = list(improved)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 4: Backtracking with bytearray for maximum cardinality
    if len(best_result) < 80 and time.time() < deadline - 5:
        bt_result = _backtrack_with_bytearray(deadline - 2)
        if len(bt_result) > best_size:
            best_size = len(bt_result)
            best_result = bt_result
    
    # Final validation
    if best_result and len(best_result) > 0:
        if not _is_valid_triples(best_result):
            # Fallback to linear greedy result
            selected_fb, _ = _greedy_linear(1, 4096, deadline)
            best_result = selected_fb
            best_size = len(best_result)
    
    return best_result
