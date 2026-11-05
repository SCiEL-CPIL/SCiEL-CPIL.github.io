import time
import sys

# Increase recursion limit just in case, though depth is max 32
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin for safety
    
    # Target differences
    target_diffs = set(range(1, 31))
    
    # Initial state
    # We must include 0.
    # We want to minimize the max element.
    # We will use DFS to find a valid Sidon set.
    # To find a *small* max element, we should try small values for new elements first.
    
    best_solution = None
    best_max_val = float('inf')
    
    def is_valid_solution(current_set, found_diffs):
        if not found_diffs.issuperset(target_diffs):
            return False
        if len(current_set) > 32:
            return False
        # Check Sidon property again? No, we maintain it incrementally.
        # But we should ensure all diffs are distinct.
        # The incremental check ensures this.
        return True

    def dfs(current_set, used_diffs, found_diffs, depth):
        nonlocal best_solution, best_max_val
        
        if time.time() > deadline:
            return False
        
        # Check if we have a solution
        if found_diffs.issuperset(target_diffs):
            # We have a valid solution. 
            # Since we are doing DFS with increasing x, the first one found 
            # at a certain depth might not be the global min max element, 
            # but it's a valid candidate.
            # Actually, to minimize max element, we should probably 
            # do a branch and bound or just accept the first one if it's small enough.
            # Given the small size, the first valid solution found by 
            # trying small x's is likely quite good.
            current_max = max(current_set)
            if current_max < best_max_val:
                best_max_val = current_max
                best_solution = list(current_set)
            # We can continue searching for a better one if time permits,
            # but usually the first one is good. Let's try to prune.
            # If we found a solution, we can stop this branch.
            # But maybe a smaller set (fewer elements) with same max? 
            # The task says "at most 32". It doesn't require minimizing size.
            # It minimizes largest element.
            # So if we found a solution with max M, we only care about finding 
            # another with max < M.
            return True # Signal that this branch yielded a solution
        
        if len(current_set) >= 32:
            return False
            
        if time.time() > deadline:
            return False

        last_val = current_set[-1]
        
        # Heuristic: We need to fill in missing differences.
        # missing_diffs = target_diffs - found_diffs
        # If missing_diffs is empty, we would have returned above.
        
        # Try next elements
        # We can bound the next element. 
        # If we have k elements left, and we need to realize D missing diffs,
        # the max element must be at least something.
        # For now, just iterate.
        
        # To make it faster, we can prioritize x that creates new found diffs.
        # But simple iteration from last_val + 1 is a good baseline for "minimize max".
        
        # We can add a bound: if last_val + 1 is already larger than best_max_val, 
        # we can't improve on best_max_val with this branch (since elements are increasing).
        if last_val >= best_max_val:
            return False

        # Iterate candidate next elements
        # We can limit the range. If we are at depth d, we have 32-d slots left.
        # The max element of the final set is at least last_val + (32-d) * 1 (min gap 1).
        # But gaps can be larger.
        
        # Let's just iterate.
        # To speed up, we can check if adding x is valid.
        
        # Optimization: 
        # If we have many missing diffs, we might need to jump.
        # But let's stick to simple DFS first.
        
        # We can try to order candidates by how many new diffs they satisfy.
        # This is a good heuristic.
        
        # However, calculating this for every x in a loop is expensive.
        # Let's just try x in increasing order.
        
        # We can set an upper bound for x in this iteration.
        # If x > best_max_val, we can't improve the max element if we haven't found a solution yet?
        # No, if we haven't found a solution, best_max_val is inf.
        # Once we find one, we prune branches where last_val >= best_max_val.
        
        # What if we haven't found a solution yet? We might go deep with large x.
        # To prevent this, we can use a "fail fast" heuristic.
        # But for N=30, the solution is small.
        
        # Let's iterate x from last_val + 1
        # We can stop if x > 10^6 (task constraint)
        
        # To improve speed, we can use a "next_x" hint.
        # If we have missing diffs, the smallest missing diff d_min.
        # We need some pair (a, b) in set such that b-a = d_min.
        # This doesn't directly constrain the next element x unless x is involved.
        
        # Let's just do the standard DFS.
        
        # To handle the "minimize max element" properly, 
        # we can use a branch and bound where we track the current max.
        # The pruning `if last_val >= best_max_val: return False` is valid 
        # because all future elements will be >= last_val.
        
        # We need to be careful: what if the current set is [0, 100] and best_max_val is 50?
        # Then last_val (100) >= 50, so we prune. Correct.
        
        # What if current set is [0, 10] and best_max_val is 50?
        # We continue. We might add 20, 30... up to 49.
        # If we reach 50, last_val becomes 50. Next iteration prunes.
        
        # This works.
        
        # One issue: we might find a solution with max 100, then later find one with max 90.
        # The DFS will explore the 100 branch, find solution, set best=100.
        # Then it backtracks and explores other branches.
        # If it finds a branch leading to max 90, it updates best=90.
        # Then it prunes branches with last_val >= 90.
        
        # This is correct.
        
        # To ensure we find a solution quickly, we should try to find *any* solution first.
        # The DFS with increasing x will find a solution.
        
        # Let's add a small optimization: 
        # If we have no found_diffs, we are at start.
        
        # We'll iterate x.
        # To avoid infinite loop if no solution exists (but one does), 
        # the depth limit 32 ensures termination.
        
        # We can also add a heuristic to skip x that don't add any new diffs 
        # if we are "behind" on diff coverage? 
        # No, sometimes you need "dummy" elements to position for future diffs.
        
        # Let's just run it.
        
        # We can limit the number of x's we try per step to avoid getting stuck?
        # No, DFS is complete.
        
        # Let's try to order x's.
        # Candidates: last_val + 1, last_val + 2, ...
        
        # We can break early if x > best_max_val (handled by last_val check? 
        # No, last_val is the previous element. x is the new one.
        # If x >= best_max_val, and we haven't found a solution in this branch yet,
        # can we prune?
        # If x >= best_max_val, the max element of the final set will be at least x.
        # So if x >= best_max_val, we cannot improve the best_max_val.
        # So we can break the loop if x >= best_max_val.
        
        # Wait, if best_max_val is inf, we don't break.
        
        x = last_val + 1
        while x < 10**6:
            if time.time() > deadline:
                return False
            
            if best_solution is not None and x >= best_max_val:
                # Since x is increasing, all further x will be >= best_max_val.
                # We can't improve the max element.
                break
            
            # Check if x can be added
            valid = True
            new_found_diffs = set(found_diffs) # Copy, or use set operations
            
            # Compute diffs with all existing elements
            for y in current_set:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                if d in target_diffs:
                    new_found_diffs.add(d)
            
            if not valid:
                x += 1
                continue
                
            # If we add x, the new used_diffs will be used_diffs union {x-y for y in current_set}
            new_used_diffs = set(used_diffs)
            for y in current_set:
                new_used_diffs.add(x - y)
                
            # Recurse
            if dfs(current_set + [x], new_used_diffs, new_found_diffs, depth + 1):
                # A solution was found in this subtree.
                # We can return True to propagate up?
                # If we return True, the parent will stop exploring other x's?
                # If we want to find the *optimal* (min max), we should continue 
                # exploring other x's at this level?
                # No, if we found a solution with max M, and we are at a node 
                # where we chose x, any other x' > x will result in a larger 
                # or equal max element (since x' > x). 
                # So if we found a solution with x, we don't need to try x' > x 
                # at this level, because they will yield max >= x > max(solution with x)?
                # Not necessarily. The solution with x might have a large later element.
                # The solution with x' might have a smaller later element?
                # No, the max element is the max of the set.
                # If we choose x, the set is {..., x, ...}. Max >= x.
                # If we choose x' > x, the set is {..., x', ...}. Max >= x'.
                # So max(solution with x') >= x' > x.
                # But max(solution with x) could be much larger than x.
                # So we cannot simply prune x' > x just because x yielded a solution.
                
                # However, if we found a solution, we update best_max_val.
                # The loop condition `if x >= best_max_val: break` will handle pruning 
                # for subsequent x's if best_max_val is updated to something small.
                
                # But should we stop searching this branch entirely?
                # If we return True, the caller might stop.
                # Let's return False to indicate "I found a solution, but keep looking 
                # for a better one at this level"?
                # No, standard DFS returns True to stop.
                # If we want to find the global minimum, we must explore all branches 
                # that could potentially beat the current best.
                
                # If we return True, the parent stops. 
                # This means we accept the first solution found in the DFS order.
                # Is the first solution found by DFS (trying small x first) optimal?
                # Not necessarily. 
                # Example: 
                # Path 1: 0, 1, ... -> Solution max 100.
                # Path 2: 0, 2, ... -> Solution max 50.
                # DFS will explore Path 1 first. It finds max 100. 
                # If it returns True, it stops. It misses Path 2.
                
                # So we should NOT return True immediately.
                # We should update best_solution and continue searching 
                # for a better one, BUT we can prune branches that can't beat best.
                
                # How to implement this?
                # Instead of returning True/False for "found solution", 
                # we just let the DFS run until deadline or full exploration.
                # The pruning `if last_val >= best_max_val: return` and 
                # `if x >= best_max_val: break` will cut off bad branches.
                
                # So we should NOT return True here. We just continue the loop 
                # to try next x, unless the loop condition breaks it.
                
                # But wait, if we don't return True, the function continues 
                # to try x+1, x+2...
                # This is correct for finding the global min.
                
                # However, this might be slow if the search space is large.
                # But with the pruning, it should be okay.
                
                # Let's change the logic: 
                # The function doesn't return "found". It just runs.
                # We remove the `return True` from the base case.
                # We just update best_solution.
                
                # We need to be careful with the `deadline` check.
                
                # Let's restructure:
                # The DFS function doesn't return a bool.
                # It just explores.
                
                # But we need to stop if time runs out.
                
                # Let's modify the code structure.
                
                pass # Placeholder for logic change
            
            x += 1
            
        return None # End of candidates

    # Start DFS
    # Initial set: [0]
    # used_diffs: empty
    # found_diffs: empty
    dfs([0], set(), set(), 0)
    
    if best_solution is None:
        # Fallback to a known valid set if DFS failed (should not happen)
        # We can use the greedy result as a fallback
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
        
    return best_solution
