import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Quick greedy to get an initial upper bound
    def greedy_construct():
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            # Try to add x such that x - y = d or y - x = d (x > y)
            # x = y + d
            for y in sorted(a):
                x = y + d
                if x in a: continue
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if found: continue
            # Try other x?
            # Just pick a large x to satisfy d
            # We need x - y = d for some y.
            # Let's just pick x = max(a) + d + 100 (large enough to avoid collisions)
            # But we want small diameter.
            # Let's try all x in range(max(a)+1, max(a)+100)
            for x in range(max(a) + 1, max(a) + 100):
                if x in a: continue
                trial = sorted(a + [x])
                if d in diffs(trial) and len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                # Fallback: add two elements
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    initial_solution = greedy_construct()
    best_diameter = max(initial_solution)
    best_solution = initial_solution
    
    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in target_diffs:
        target_mask |= (1 << d)
        
    def search(current_set, used_diffs, covered_mask, depth):
        nonlocal best_solution, best_diameter
        
        if time.time() > deadline:
            return
            
        current_max = current_set[-1] if current_set else 0
        
        if covered_mask == target_mask:
            if current_max < best_diameter:
                best_diameter = current_max
                best_solution = list(current_set)
            return
            
        if len(current_set) >= 32:
            return
            
        if current_max >= best_diameter:
            return

        missing_diffs = [d for d in target_diffs if not (covered_mask & (1 << d))]
        if not missing_diffs:
            return
            
        candidates = set()
        for d in missing_diffs:
            for y in current_set:
                x = y + d
                if x > current_max:
                    candidates.add(x)
        
        valid_candidates = []
        for x in candidates:
            valid = True
            for y in current_set:
                if x - y in used_diffs:
                    valid = False
                    break
            if valid:
                valid_candidates.append(x)
                
        if not valid_candidates:
             # Fallback candidates
             for x in range(current_max + 1, current_max + 20):
                 valid = True
                 for y in current_set:
                     if x - y in used_diffs:
                         valid = False
                         break
                 if valid:
                     valid_candidates.append(x)

        # Sort by x, then by coverage
        def score(x):
            new_covered = 0
            for d in missing_diffs:
                if x - d in current_set:
                    new_covered += 1
            return (x, -new_covered)
            
        valid_candidates.sort(key=score)
        
        for x in valid_candidates:
            if time.time() > deadline:
                return
                
            new_diffs = set()
            valid = True
            for y in current_set:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                continue
                
            new_used_diffs = used_diffs | new_diffs
            new_covered_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_covered_mask |= (1 << d)
                    
            new_set = current_set + [x]
            search(new_set, new_used_diffs, new_covered_mask, depth + 1)
            
    search([0], set(), 0, 0)
    
    return best_solution

