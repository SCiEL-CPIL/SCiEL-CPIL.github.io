import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    N = 284
    MAX_SUM = 2 * N

    # Pre-filter candidates that are compatible with prescribed set
    initial_used = bytearray(MAX_SUM + 1)
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            initial_used[prescribed[i] + prescribed[j]] = 1

    all_candidates = []
    for x in range(1, N + 1):
        if x in prescribed_set:
            continue
        valid = True
        if initial_used[2 * x]:
            valid = False
        else:
            for y in prescribed:
                if initial_used[x + y]:
                    valid = False
                    break
        if valid:
            all_candidates.append(x)

    # Precompute conflict graph for candidates
    # conflict[x] = set of candidates that conflict with x (cannot be in same Sidon set)
    # Two candidates x, y conflict if:
    # 1. x == y (not relevant since distinct)
    # 2. x + y is a sum already in the set, or
    # 3. 2*x == 2*y (i.e., x == y)
    # Actually, for the graph, we need to know which pairs of candidates cannot coexist.
    # Two candidates x and y (x != y) cannot coexist if x + y is a sum that is already "blocked"
    # by the prescribed elements, OR if x + y equals some other sum that would be created?
    # No, the condition for a Sidon set is that all pairwise sums are distinct.
    # So x and y can coexist if x+y is not equal to any other sum a+b (a<=b) in the set.
    # In the conflict graph for the *remaining* candidates, we can only precompute conflicts
    # between candidate pairs based on the prescribed set.
    # If x+y is a sum of prescribed elements, they conflict.
    # Also, if 2*x is a sum of prescribed elements, x conflicts with itself? No, 2*x is a sum.
    # If x is a candidate, 2*x must not be a sum of prescribed elements (already checked in pre-filter).
    # If x and y are both candidates, and x+y is a sum of prescribed elements, they conflict.
    # Additionally, if x+y equals some other sum that will be formed by other selected candidates,
    # that's a dynamic conflict. But for the *static* graph, we only know about prescribed sums.
    
    # Let's build a graph where edges represent "cannot be together" based on prescribed sums.
    # This is a necessary condition, not sufficient.
    
    conflict = {}
    for x in all_candidates:
        conflict[x] = set()
        for y in all_candidates:
            if x == y:
                continue
            if initial_used[x + y]:
                conflict[x].add(y)
                conflict[y].add(x)
            # Also, if 2*x == 2*y? No, x!=y implies 2*x != 2*y.
            # What if x+y == 2*z for some z? That's a dynamic constraint.

    # Also, a candidate x conflicts with itself if 2*x is a prescribed sum (already filtered).
    
    # Let's refine: we want to find a large independent set in a graph where edges represent
    # "cannot coexist". The graph edges are:
    # 1. x-y edge if x+y is a prescribed sum.
    # 2. x-y edge if x+y is a sum of two other prescribed elements? No, that's covered by 1.
    # 3. x-y edge if 2*x == 2*y? No.
    # 4. x-y edge if x+y == 2*z for some z in the set? This is dynamic.
    
    # So the static graph is just based on prescribed sums. This is a weak constraint.
    # However, we can use it for a heuristic: candidates with fewer static conflicts are "better".

    # Let's also consider: if x and y are both selected, then x+y must not equal 2*z for any z in the set.
    # This is a "3-sum" type constraint.
    
    # For now, let's stick with the static conflict graph for ordering.
    
    # Compute degree in static conflict graph
    degree = {x: len(conflict[x]) for x in all_candidates}

    best_size = 0
    best_set = list(prescribed)

    def get_sums_set(lst):
        """Return a bytearray of used sums for a given list of elements."""
        used = bytearray(MAX_SUM + 1)
        n = len(lst)
        for i in range(n):
            for j in range(i, n):
                used[lst[i] + lst[j]] = 1
        return used

    def is_valid(x, current_set, used):
        """Check if x can be added to current_set given used sums."""
        if used[2 * x]:
            return False
        for y in current_set:
            if used[x + y]:
                return False
        return True

    def greedy_fill(current_set, used, candidates, time_limit=0.5):
        """Greedily add elements to maximize size."""
        start = time.time()
        cands = list(candidates)
        random.shuffle(cands)
        
        # Sort by "safety" - elements that block fewer other candidates?
        # Use static degree as a heuristic: lower degree is better.
        cands.sort(key=lambda x: degree[x])
        
        for x in cands:
            if time.time() - start > time_limit:
                break
            if x in current_set:
                continue
            if is_valid(x, current_set, used):
                current_set.add(x)
                used[2 * x] = 1
                for y in current_set:
                    if y != x:
                        used[x + y] = 1
        return current_set, used

    def backtrack_search(current_set, used, candidates, target_size, time_limit=2.0):
        """
        Try to extend current_set to target_size using backtracking.
        Returns (True, new_set) if successful, else (False, best_found_set).
        """
        start = time.time()
        best_found = set(current_set)
        
        def dfs(sel, used_arr, cand_list, depth):
            nonlocal best_found
            if time.time() - start > time_limit:
                return False
            
            if len(sel) >= target_size:
                best_found = set(sel)
                return True
            
            if len(cand_list) < target_size - len(sel):
                return False

            # Heuristic: Try candidates with lower static degree first
            # Also, filter out candidates that are already in sel or invalid
            # To speed up, we can pre-filter candidates that are valid
            valid_cands = []
            for x in cand_list:
                if x in sel:
                    continue
                if not is_valid(x, sel, used_arr):
                    continue
                valid_cands.append(x)
            
            if not valid_cands:
                return False
            
            # Sort by static degree
            valid_cands.sort(key=lambda x: degree[x])
            
            for x in valid_cands:
                if time.time() - start > time_limit:
                    return False
                
                # Add x
                sel.add(x)
                used_arr[2 * x] = 1
                for y in sel:
                    if y != x:
                        used_arr[x + y] = 1
                
                # Recurse
                if dfs(sel, used_arr, cand_list, depth + 1):
                    return True
                
                # Undo
                sel.remove(x)
                used_arr[2 * x] = 0
                for y in sel:
                    used_arr[x + y] = 0

            return False

        # Prepare candidates for DFS
        dfs_candidates = [x for x in candidates if x not in current_set]
        random.shuffle(dfs_candidates)
        
        # Run DFS
        dfs(set(current_set), used, dfs_candidates, 0)
        
        return best_found

    # Main Loop
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        
        # 1. Initialize
        current_set = set(prescribed_set)
        current_used = get_sums_set(prescribed)
        
        # 2. Greedy Fill
        current_set, current_used = greedy_fill(current_set, current_used, all_candidates, 0.5)
        
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
            if best_size >= 20:
                break

        # 3. Try to reach 20
        target = 20
        
        # If greedy got us close, try backtracking
        if current_size < target:
            remaining = deadline - time.time()
            bt_time = min(2.0, max(0.5, remaining * 0.05))
            
            new_set = backtrack_search(current_set, current_used, all_candidates, target, bt_time)
            
            if len(new_set) > current_size:
                current_set = new_set
                current_used = get_sums_set(list(current_set))
                current_size = len(current_set)
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_set)
                    if best_size >= 20:
                        break

        # 4. Kick
        if current_size < 20:
            removable = [x for x in current_set if x not in prescribed_set]
            if removable:
                kick_size = 1 if random.random() < 0.7 else 2
                if len(removable) >= kick_size:
                    to_remove = random.sample(removable, kick_size)
                    for x in to_remove:
                        current_set.remove(x)
                    current_used = get_sums_set(list(current_set))
                    
                    # Try to refill
                    current_set, current_used = greedy_fill(current_set, current_used, all_candidates, 0.5)
                    current_size = len(current_set)
                    
                    if current_size < 20:
                        remaining = deadline - time.time()
                        if remaining > 1:
                            bt_time = min(1.5, remaining * 0.05)
                            new_set = backtrack_search(current_set, current_used, all_candidates, 20, bt_time)
                            if len(new_set) > current_size:
                                current_set = new_set
                                current_used = get_sums_set(list(current_set))
                                current_size = len(current_set)
                                if current_size > best_size:
                                    best_size = current_size
                                    best_set = list(current_set)
                                    if best_size >= 20:
                                        break

    # Final Polish: 1-for-2 swap
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            removable = [x for x in best_set if x not in prescribed_set]
            for x_remove in removable:
                if time.time() >= deadline - 2:
                    break
                temp_set = set(best_set)
                temp_set.remove(x_remove)
                temp_sums = get_sums_set(list(temp_set))
                
                found_pair = False
                for y_add in all_candidates:
                    if y_add in temp_set:
                        continue
                    if is_valid(y_add, temp_set, temp_sums):
                        temp_set.add(y_add)
                        temp_sums[2 * y_add] = 1
                        for y in temp_set:
                            if y != y_add:
                                temp_sums[y_add + y] = 1
                                
                        for z_add in all_candidates:
                            if z_add in temp_set:
                                continue
                            if is_valid(z_add, temp_set, temp_sums):
                                best_set = list(temp_set)
                                best_size = len(best_set)
                                improved = True
                                found_pair = True
                                break
                        
                        if found_pair:
                            break
                        
                        temp_set.remove(y_add)
                        temp_sums[2 * y_add] = 0
                        for y in temp_set:
                            temp_sums[y_add + y] = 0
                        
                        if found_pair:
                            break
                            
                if found_pair:
                    break

    # Validation
    final_sums = get_sums_set(best_set)
    expected_sums = len(best_set) * (len(best_set) + 1) // 2
    actual_sums = sum(final_sums)
    if actual_sums != expected_sums:
        return sorted(prescribed)

    return sorted(best_set)
