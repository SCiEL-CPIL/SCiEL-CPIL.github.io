import time

def construct():
    deadline = time.time() + 145
    
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]

    # Greedy to find initial solution
    def greedy_solve():
        a = [0]
        for d in range(1, 31):
            if d in get_diffs(sorted(a)):
                continue
            found = False
            for x in range(1, 500):
                if x in a:
                    continue
                trial = sorted(a + [x])
                t_diffs = get_diffs(trial)
                if len(set(t_diffs)) == len(t_diffs) and d in t_diffs:
                    a = trial
                    found = True
                    break
            if not found:
                return None
        return a

    best_solution = greedy_solve()
    if best_solution:
        best_max_val = max(best_solution)
    else:
        best_max_val = float('inf')
        best_solution = None

    def solve(current_set, used_diffs):
        nonlocal best_solution, best_max_val
        
        if time.time() > deadline:
            return
        
        if all(d in used_diffs for d in range(1, 31)):
            if len(current_set) <= 32:
                m = max(current_set)
                if m < best_max_val:
                    best_max_val = m
                    best_solution = current_set[:]
            return
        
        if len(current_set) >= 32:
            return
            
        if max(current_set) >= best_max_val:
            return
            
        missing = [d for d in range(1, 31) if d not in used_diffs]
        
        candidates = set()
        for y in current_set:
            for d in missing:
                candidates.add(y + d)
                
        for x in sorted(candidates):
            if x > best_max_val:
                continue
            if x in current_set:
                continue
                
            valid = True
            new_diffs = set()
            for y in current_set:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                solve(current_set + [x], used_diffs | new_diffs)

    if best_solution:
        solve(sorted(best_solution), set(get_diffs(sorted(best_solution))))
    else:
        solve([0], set())

    return best_solution
