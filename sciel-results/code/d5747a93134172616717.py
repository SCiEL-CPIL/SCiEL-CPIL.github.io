import time

def construct():
    target_diffs = set(range(1, 31))
    best_set = None
    best_max = float('inf')
    start_time = time.time()
    timeout = 10.0 # 10 seconds limit for search

    def backtrack(S, diffs_set, last_val):
        nonlocal best_set, best_max
        
        # Time check
        if time.time() - start_time > timeout:
            return

        # Prune if current max is already >= best found
        if last_val >= best_max:
            return

        # If we have enough elements, check coverage
        if len(S) >= 9:
            if target_diffs.issubset(diffs_set):
                m = max(S)
                if m < best_max:
                    best_max = m
                    best_set = list(S)
            return

        # Prune if we can't possibly beat best_max
        # Min possible max element if we add 1 each time:
        if last_val + (9 - len(S)) >= best_max:
            return

        # Determine upper bound for next element
        # We don't need to go beyond best_max
        upper_bound = best_max if best_max < float('inf') else 100
        
        for x in range(last_val + 1, upper_bound + 1):
            valid = True
            new_diffs = []
            for y in S:
                d = x - y
                if d in diffs_set:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue

            # Recurse
            new_diffs_set = diffs_set.union(new_diffs)
            backtrack(S + [x], new_diffs_set, x)
            
            # If we found a solution, we might want to break early?
            # No, because a smaller x might lead to a smaller max later?
            # Actually, if we fix the prefix S, choosing a smaller x 
            # doesn't necessarily lead to a smaller final max if we have to stretch later.
            # But generally, keeping elements small is good.
            # However, we must explore all valid x to ensure we find the global min.
            # But if we find a solution, best_max is updated, which prunes other branches.

    backtrack([0], set(), 0)

    if best_set:
        return best_set
    else:
        # Fallback to greedy
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
