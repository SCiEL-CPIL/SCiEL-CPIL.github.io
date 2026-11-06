"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 110
    
    best_solution = None
    best_max = float('inf')
    
    def check_sidon(s):
        """Check if sorted list s is Sidon."""
        n = len(s)
        diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                d = s[j] - s[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def get_diffs_set(s):
        n = len(s)
        diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(s[j] - s[i])
        return diffs
    
    # DFS: build Sidon set containing 0, try to cover 1..30
    # State: current sorted list, set of covered differences
    def dfs(current, covered):
        nonlocal best_solution, best_max
        
        if time.time() > deadline:
            return
        
        # Check completion
        if covered >= 30:
            mx = current[-1]
            if mx < best_max:
                best_max = mx
                best_solution = list(current)
            return
        
        # Lower bound pruning: 
        # We need 30 - len(covered) more differences
        # Each new element adds len(current) new differences
        # But some might be > 30 or duplicates
        # Loose bound: need at least ceil((30-len(covered))/len(current)) more elements
        # Not very useful for pruning max element
        
        # Try adding next element
        start = current[-1] + 1 if current else 1
        
        # Upper bound: don't exceed current best
        upper = best_max - 1 if best_max < float('inf') else 50
        
        for x in range(start, upper + 1):
            if time.time() > deadline:
                return
            
            # Check Sidon constraint with x
            valid = True
            new_diffs = []
            for s_i in current:
                d = x - s_i
                if d <= 0:
                    valid = False
                    break
                if d in covered:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # new_diffs are all distinct (since s_i are distinct)
            # and none are in covered (checked above)
            
            new_covered = covered | set(new_diffs)
            
            # Additional pruning: if even with optimal future additions we can't reach 30
            # Skip for now
            
            dfs(current + [x], new_covered)
            
            if best_solution is not None:
                # If we found a solution, we might be able to prune
                if best_max <= x:
                    return
    
    dfs([0], set())
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: greedy
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

# EVOLVE-BLOCK-END
