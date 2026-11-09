import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    target_diffs = set(range(1, 31))
    best_solution = None
    best_max_val = float('inf')
    
    # Initial greedy to get a valid upper bound quickly
    def greedy_construct():
        s = [0]
        covered = set()
        all_diffs = set()
        
        for _ in range(32):
            if covered == target_diffs:
                break
            found = False
            candidates = set()
            last = max(s)
            for d in target_diffs - covered:
                for v in s:
                    x = v + d
                    if x > last:
                        candidates.add(x)
            
            # Sort candidates by value
            for x in sorted(candidates):
                new_diffs = {x - v for v in s}
                if new_diffs & all_diffs:
                    continue
                s.append(x)
                s.sort()
                all_diffs |= new_diffs
                covered |= (new_diffs & target_diffs)
                found = True
                break
            if not found:
                x = last + 1
                while True:
                    new_diffs = {x - v for v in s}
                    if not (new_diffs & all_diffs):
                        s.append(x)
                        s.sort()
                        all_diffs |= new_diffs
                        covered |= (new_diffs & target_diffs)
                        break
                    x += 1
        return s if covered == target_diffs else None

    init_sol = greedy_construct()
    if init_sol:
        best_solution = init_sol
        best_max_val = max(init_sol)

    def dfs(s, covered, all_diffs):
        nonlocal best_solution, best_max_val
        
        if time.time() > deadline:
            return False
            
        if covered == target_diffs:
            curr_max = max(s)
            if curr_max < best_max_val:
                best_max_val = curr_max
                best_solution = list(s)
            return True
            
        if len(s) >= 32:
            return False
            
        curr_max = max(s)
        if curr_max >= best_max_val:
            return False
            
        last = curr_max
        missing = target_diffs - covered
        slots_left = 32 - len(s)
        
        # Heuristic Pruning:
        # Max new diffs we can generate with slots_left new elements:
        # Each new element pairs with len(s) existing + (slots_left - 1) other new ones?
        # Actually, if we add j elements, total new pairs involving at least one new element:
        # j * len(s) + j * (j - 1) / 2
        max_new_diffs = slots_left * len(s) + (slots_left * (slots_left - 1)) // 2
        if len(missing) > max_new_diffs:
            return False
            
        candidates = set()
        for d in missing:
            for v in s:
                x = v + d
                if x > last:
                    candidates.add(x)
        
        # Sort candidates
        # We want to try small values first to minimize diameter.
        # However, if a small value covers few diffs, it might be a dead end.
        # Sorting by value is standard.
        for x in sorted(candidates):
            if x >= best_max_val:
                break
                
            new_diffs = {x - v for v in s}
            if new_diffs & all_diffs:
                continue
                
            new_covered = covered | (new_diffs & target_diffs)
            new_all_diffs = all_diffs | new_diffs
            new_s = s + [x]
            
            if dfs(new_s, new_covered, new_all_diffs):
                return True
                
        return False

    dfs([0], set(), set())
    
    if best_solution:
        return best_solution
    else:
        return [0, 1, 4, 10, 12, 17, 20, 23, 29, 34, 37, 40]
