"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 130
    prescribed = [1, 2, 5, 11]
    N = 284
    
    def is_valid_sidon(arr):
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    # CSP with forward checking
    def solve_csp(target_k, time_budget):
        """CSP solver with forward checking. Returns a valid Sidon set of size target_k or None."""
        start_time = time.time()
        prescribed_set = set(prescribed)
        
        # State: assigned values, used sums (including 2*x), remaining candidates per position
        # We assign positions 0..target_k-1, where first 4 are prescribed
        # Position order: assign in order of most-constrained first (prescribed first, then others)
        
        # For CSP, we need to track which sums are used
        # A sum s = a + b where a <= b, a and b in the set
        # When we assign a new value x to position i, we add sums x + y for all already assigned y, and 2*x
        
        # We'll use backtracking with forward checking
        # Domain for each unassigned position: all values in 1..N not yet used
        
        # Sort assignment order: prescribed first (fixed), then others
        # Actually, prescribed are fixed, so we just need to assign target_k - 4 more values
        
        num_extra = target_k - len(prescribed)
        if num_extra <= 0:
            return list(prescribed)
        
        # Initial state
        assigned = list(prescribed)
        used_sums = set()
        used_values = set(prescribed)
        
        # Compute initial sums from prescribed
        for i in range(len(prescribed)):
            for j in range(i, len(prescribed)):
                used_sums.add(prescribed[i] + prescribed[j])
        
        # For forward checking, we need to know for each candidate value x,
        # whether it would conflict with any already assigned value
        # x conflicts if for any y in assigned: x + y in used_sums (and x != y or 2*x in used_sums)
        # Actually: when adding x, new sums are {x + y for y in assigned} ∪ {2*x}
        # These must all be new (not in used_sums) and distinct from each other
        
        # We'll do depth-first search with MRV heuristic
        # But with forward checking, we can prune early
        
        # Let's use a simpler approach: iterative deepening or just DFS with pruning
        # Since target_k is at most 20, and N is 284, we need to be smart
        
        # Strategy: assign one value at a time, maintaining used_sums and used_values
        # For each new value x, check that all new sums are valid
        
        # To make this efficient, we'll precompute which values are "compatible" with current state
        # A value x is compatible if:
        # 1. x not in used_values
        # 2. For all y in assigned: x + y not in used_sums (if x != y) or 2*x not in used_sums
        # 3. All new sums are distinct: for all y1, y2 in assigned with y1 < y2: x + y1 != x + y2 (always true if y1 != y2)
        #    and for all y in assigned: x + y != 2*x, i.e., y != x (guaranteed by condition 1)
        
        # So the key check is: for all y in assigned, x + y not in used_sums, and 2*x not in used_sums
        
        # Let's implement DFS with recursion limit
        import sys
        sys.setrecursionlimit(10000)
        
        best_solution = None
        
        def dfs(depth, assigned_list, used_sums_set, used_values_set):
            nonlocal best_solution
            if time.time() - start_time > time_budget:
                return None
            
            if depth == target_k:
                best_solution = list(assigned_list)
                return best_solution
            
            # MRV: choose the next position to assign
            # For simplicity, we assign in order, but we can use heuristic
            # Positions 0-3 are prescribed, so we start from position 4
            # Actually, let's just assign the next free position
            
            # Try candidate values in a good order
            # Heuristic: try values that create fewer conflicts
            candidates = []
            for x in range(1, N + 1):
                if x in used_values_set:
                    continue
                # Check if x is compatible
                compatible = True
                if 2 * x in used_sums_set:
                    compatible = False
                else:
                    for y in assigned_list:
                        if x + y in used_sums_set:
                            compatible = False
                            break
                if compatible:
                    candidates.append(x)
            
            if not candidates:
                return None
            
            # Order candidates: prefer those that leave more options for future
            # Simple heuristic: try middle values first (more flexible)
            # Or: try values that minimize new sum conflicts
            # For now, just try in order, but we can improve
            
            # To make search faster, limit candidates to a subset
            # Use randomization with restarts
            random.shuffle(candidates)
            # Limit to top K candidates to keep search tractable
            K = min(len(candidates), 50)
            candidates = candidates[:K]
            
            for x in candidates:
                if time.time() - start_time > time_budget:
                    return None
                
                # Assign x
                new_used_sums = set(used_sums_set)
                new_used_values = set(used_values_set)
                new_assigned = list(assigned_list)
                
                # Add new sums
                conflict = False
                for y in assigned_list:
                    s = x + y
                    if s in new_used_sums:
                        conflict = True
                        break
                    new_used_sums.add(s)
                if not conflict:
                    s = 2 * x
                    if s in new_used_sums:
                        conflict = True
                    else:
                        new_used_sums.add(s)
                
                if conflict:
                    continue
                
                new_used_values.add(x)
                new_assigned.append(x)
                
                result = dfs(depth + 1, new_assigned, new_used_sums, new_used_values)
                if result is not None:
                    return result
            
            return None
        
        # Multiple restarts with different random seeds
        for attempt in range(20):
            if time.time() - start_time > time_budget:
                break
            random.seed(42 + attempt)
            result = dfs(4, list(prescribed), set(used_sums), set(prescribed))
            if result is not None:
                return result
        
        return None
    
    # First try CSP for target size 20
    target_sizes = [20, 19, 18, 17, 16]
    best_a = None
    
    for target_k in target_sizes:
        if time.time() > deadline - 30:
            break
        if target_k <= len(prescribed):
            continue
        
        # Allocate time budget for this target
        remaining_time = deadline - time.time() - 30
        time_budget = min(20, remaining_time * 0.5)  # Spend at most 20s per target
        
        result = solve_csp(target_k, time_budget)
        if result is not None and is_valid_sidon(result):
            best_a = result
            break
    
    # If CSP didn't find a solution, fall back to SA
    if best_a is None:
        # Greedy initial solution
        a = list(prescribed)
        sums = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                sums.add(x + y)
        for x in range(12, N + 1):
            new = {x + y for y in a} | {2 * x}
            if len(new) == len(a) + 1 and not new & sums:
                a.append(x)
                sums |= new
        best_size = len(a)
        best_a = list(a)
        
        # SA with incremental O(k) conflict updates
        def build_diff_counts(arr):
            max_diff = N
            diff_counts = [0] * (max_diff + 1)
            num_conflicts = 0
            n = len(arr)
            for i in range(n):
                for j in range(i + 1, n):
                    d = arr[j] - arr[i]
                    diff_counts[d] += 1
                    if diff_counts[d] > 1:
                        num_conflicts += 1
            return diff_counts, num_conflicts
        
        def apply_move(arr, diff_counts, num_conflicts, idx, new_val):
            n = len(arr)
            old_val = arr[idx]
            for j in range(n):
                if j == idx:
                    continue
                if arr[j] > old_val:
                    d = arr[j] - old_val
                else:
                    d = old_val - arr[j]
                diff_counts[d] -= 1
                if diff_counts[d] == 1:
                    num_conflicts -= 1
            for j in range(n):
                if j == idx:
                    continue
                if arr[j] > new_val:
                    d = arr[j] - new_val
                else:
                    d = new_val - arr[j]
                diff_counts[d] += 1
                if diff_counts[d] == 2:
                    num_conflicts += 1
            arr[idx] = new_val
            return num_conflicts
        
        def revert_move(arr, diff_counts, num_conflicts, idx, new_val, old_val):
            n = len(arr)
            for j in range(n):
                if j == idx:
                    continue
                if arr[j] > new_val:
                    d = arr[j] - new_val
                else:
                    d = new_val - arr[j]
                diff_counts[d] -= 1
                if diff_counts[d] == 1:
                    num_conflicts -= 1
            for j in range(n):
                if j == idx:
                    continue
                if arr[j] > old_val:
                    d = arr[j] - old_val
                else:
                    d = old_val - arr[j]
                diff_counts[d] += 1
                if diff_counts[d] == 2:
                    num_conflicts += 1
            arr[idx] = old_val
            return num_conflicts
        
        random.seed(42)
        
        for target_k in [min(20, N), best_size + 1, best_size]:
            if time.time() > deadline - 15:
                break
            if target_k <= len(prescribed):
                continue
            
            remaining = target_k - len(prescribed)
            greedy_extra = [x for x in a if x not in prescribed]
            chosen_extra = list(greedy_extra[:remaining]) if len(greedy_extra) >= remaining else list(greedy_extra)
            if len(chosen_extra) < remaining:
                available = [x for x in range(1, N + 1) if x not in prescribed and x not in chosen_extra]
                random.shuffle(available)
                chosen_extra += available[:remaining - len(chosen_extra)]
            
            sol = list(prescribed) + list(chosen_extra)
            diff_counts, num_conflicts = build_diff_counts(sol)
            
            if num_conflicts == 0:
                if target_k > best_size:
                    best_size = target_k
                    best_a = list(sol)
                break
            
            T = 5.0
            T_min = 0.01
            cooling = 0.9999
            max_iters = 200000
            
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                if num_conflicts == 0:
                    if target_k > best_size:
                        best_size = target_k
                        best_a = list(sol)
                    break
                
                idx = random.randint(4, target_k - 1)
                old_val = sol[idx]
                new_val = random.randint(1, N)
                
                valid = True
                for i in range(target_k):
                    if i != idx and sol[i] == new_val:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_conflicts = apply_move(sol, diff_counts, num_conflicts, idx, new_val)
                delta = new_conflicts - num_conflicts
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    num_conflicts = new_conflicts
                else:
                    num_conflicts = revert_move(sol, diff_counts, num_conflicts, idx, new_val, old_val)
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            if num_conflicts == 0 and target_k > best_size:
                best_size = target_k
                best_a = list(sol)
    
    # Verify and return
    if best_a is not None and is_valid_sidon(best_a):
        return sorted(best_a)
    return sorted(prescribed)
# EVOLVE-BLOCK-END
