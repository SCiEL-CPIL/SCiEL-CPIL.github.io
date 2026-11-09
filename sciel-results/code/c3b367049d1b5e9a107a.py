import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    target_diffs = set(range(1, 31))
    best_max = float('inf')
    best_set = None
    
    # Initial state
    S = [0]
    D = set()
    miss = set(target_diffs)
    
    # We can use a recursive function or a stack.
    # Recursion depth is small (max 32 elements), so recursion is fine.
    # But Python recursion limit might be an issue if we go deep, though 32 is small.
    # We'll use a stack to be safe and allow iterative deepening if needed, 
    # but simple DFS with branch and bound should work.
    
    # To improve pruning, we can calculate the maximum number of new differences 
    # we can generate with `k` more elements.
    # If current size is n, adding k elements generates n*k + k*(k-1)/2 differences.
    
    def dfs(S, D, miss, current_max):
        nonlocal best_max, best_set
        
        if time.time() > deadline:
            return
        
        if not miss:
            if current_max < best_max:
                best_max = current_max
                best_set = list(S)
            return
        
        # Pruning 1: Already exceeded best
        if current_max >= best_max:
            return
            
        n = len(S)
        # We can add at most 32 - n elements.
        # But we don't need to add more than necessary.
        # Minimum elements needed:
        # Let k be number of new elements.
        # New diffs = n*k + k*(k-1)/2
        # We need new diffs >= len(miss)
        
        # Determine min k needed
        min_k = 0
        max_k = 32 - n
        found_k = False
        for k in range(1, max_k + 1):
            new_diffs_cap = n * k + k * (k - 1) // 2
            if new_diffs_cap >= len(miss):
                min_k = k
                found_k = True
                break
        if not found_k:
            return # Impossible to cover all missing diffs with max 32 elements
        
        # Pruning 2: If we add min_k elements, the max element will be at least current_max + 1 * min_k?
        # Not strictly, but roughly.
        # A stronger bound: The smallest possible next element is current_max + 1.
        # If we add min_k elements, the max will be at least current_max + min_k.
        # If current_max + min_k >= best_max, prune.
        if current_max + min_k >= best_max:
            return
            
        # Generate candidates for next element x
        # x must be > current_max
        # x must be < best_max (since we want to improve)
        
        # To find good candidates, we can look for x such that x - s covers missing diffs.
        # We iterate x from current_max + 1 to best_max - 1.
        # This range might be large if best_max is inf initially.
        # Initial best_max is inf. We need a heuristic initial bound.
        # A known Golomb ruler for 9 marks has length ~ 40-50?
        # Let's set an initial bound, say 1000, or find a quick greedy solution first.
        
        if best_max == float('inf'):
            # Quick greedy to get a bound
            S_greedy = [0]
            D_greedy = set()
            miss_greedy = set(target_diffs)
            for _ in range(31): # Max 32 elements
                if not miss_greedy: break
                found = False
                for x in range(1, 10000):
                    new_diffs = [x - s for s in S_greedy]
                    if len(set(new_diffs)) != len(new_diffs): continue
                    if any(d in D_greedy for d in new_diffs): continue
                    # Check if x is too big? No, just find any valid x.
                    # To keep it small, we stop at first valid x.
                    S_greedy.append(x)
                    D_greedy.update(new_diffs)
                    miss_greedy -= set(new_diffs)
                    found = True
                    break
                if not found:
                    break
            if miss_greedy:
                # Greedy failed to cover all, try larger x?
                # For now, just set a high bound
                best_max = 10000
            else:
                best_max = max(S_greedy)
                best_set = list(S_greedy)
                
        # Now we have a finite best_max.
        
        candidates = []
        # We only need to consider x up to best_max - 1
        # But we can be smarter.
        # If we need to cover `m` missing diffs, and we have `n` elements,
        # the next element x should be such that x - s hits missing diffs.
        
        # Iterate x from current_max + 1 to best_max - 1
        # This loop could be expensive if best_max is large.
        # We can limit the search depth or use a better heuristic.
        
        # Heuristic: Sort candidates by how many missing diffs they cover.
        # But generating all candidates up to best_max is O(best_max).
        # If best_max is ~50, it's fine. If 10000, it's slow.
        
        # Let's try to bound x more tightly.
        # The smallest missing diff is d_min.
        # If we pick x, x - s = d_min => x = s + d_min.
        # So x is likely close to s + d_min.
        
        # Instead of iterating all x, we can generate candidate x values
        # based on missing differences.
        # For each s in S and d in miss, x = s + d is a candidate.
        # Also x = current_max + 1, etc.
        
        cand_set = set()
        # Add small increments
        for i in range(1, 10):
            cand_set.add(current_max + i)
            
        # Add candidates from missing diffs
        for s in S:
            for d in miss:
                x = s + d
                if x > current_max:
                    cand_set.add(x)
                
        # Filter candidates
        valid_candidates = []
        for x in cand_set:
            if x >= best_max:
                continue
            new_diffs = [x - s for s in S]
            # Check distinctness among new diffs
            if len(set(new_diffs)) != len(new_diffs):
                continue
            # Check against existing D
            if any(d in D for d in new_diffs):
                continue
            # Count how many missing diffs are covered
            covered = len(set(new_diffs) & miss)
            if covered == 0:
                # If it doesn't cover any missing diff, it's likely useless for minimizing max,
                # unless it enables future smaller diffs?
                # But it increases max. So we can prune if covered == 0 and we are not forced to add it?
                # We are not forced to add it. We can skip it.
                continue
            
            valid_candidates.append((x, covered, new_diffs))
        
        # Sort by covered descending, then x ascending
        valid_candidates.sort(key=lambda item: (-item[1], item[0]))
        
        for x, covered, new_diffs in valid_candidates:
            new_S = S + [x]
            new_D = D.union(new_diffs)
            new_miss = miss.difference(new_diffs)
            new_max = x
            
            dfs(new_S, new_D, new_miss, new_max)
            
            # Optional: if we found a solution, we might want to break early if x is small?
            # No, we want to explore all to find min max.
            
    dfs(S, D, miss, 0)
    
    if best_set is None:
        # Fallback to the seed program's result or a simple valid set
        # The seed program returns a valid set.
        # We should ensure we return *a* valid set.
        # If search failed, we can run the greedy again.
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
        
    return best_set
# EVOLVE-BLOCK-END
