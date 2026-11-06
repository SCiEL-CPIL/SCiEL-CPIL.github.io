import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 128
    
    # Precompute 4-term progressions
    # A 4-term progression is (a, a+d, a+2d, a+3d) with d > 0
    # Indices are 0-based in our array.
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
    
    # For each position i, store the list of indices of progressions it belongs to.
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[d].append(idx)
    
    def count_violations(colors):
        v = 0
        for (a, b, c, d) in progressions:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                v += 1
        return v
        
    def minimize_colors(k):
        # We want to find a valid coloring using at most k colors.
        # If k=3 works, return it.
        
        # Strategy: Simulated Annealing or WalkSAT style.
        # Start with a random assignment.
        # Iteratively pick a random variable, try all values, pick the one that minimizes local cost.
        # Accept worse moves with probability exp(-delta/T) to escape local minima.
        
        random.seed(42)
        colors = [random.randint(0, k-1) for _ in range(n)]
        
        # Precompute initial violation count
        # To make this fast, we can maintain a count of violations for each progression?
        # Or just recompute for affected progressions.
        
        # Let's maintain a boolean array is_violated for each progression.
        is_violated = [False] * len(progressions)
        total_violations = 0
        for idx, (a, b, c, d) in enumerate(progressions):
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                is_violated[idx] = True
                total_violations += 1
        
        if total_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = total_violations
        
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        
        max_iters = 200000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            i = random.randint(0, n-1)
            old_c = colors[i]
            
            # Find best new color for i based on local violation change
            best_new_c = old_c
            best_delta = 0
            
            # Calculate delta for each possible new color
            # Only progressions containing i are affected
            affected_progs = prog_at[i]
            
            # Current violations for affected progs
            current_violations = sum(1 for idx in affected_progs if is_violated[idx])
            
            for c in range(k):
                if c == old_c:
                    continue
                colors[i] = c
                
                new_violations = 0
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    if len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3:
                        new_violations += 1
                        
                delta = new_violations - current_violations
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = c
                    
            # Revert to old color for now, then decide
            colors[i] = old_c
            
            if best_new_c == old_c:
                # No better color found, maybe accept a random worse one?
                # Or just skip.
                # In WalkSAT, we usually pick a random variable from a violated clause.
                # Let's stick to picking the best local move, even if it's 0 (no change).
                # If delta > 0, we might still accept it.
                
                # Calculate probability of accepting worse move
                if best_delta > 0 and T > min_T:
                    prob = pow(2.718281828, -best_delta / T)
                    if random.random() < prob:
                        # Accept worse move
                        colors[i] = best_new_c
                        # Update state
                        for idx in affected_progs:
                            a, b, c_idx, d = progressions[idx]
                            old_violated = is_violated[idx]
                            new_violated = len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3
                            if old_violated != new_violated:
                                if new_violated:
                                    total_violations += 1
                                else:
                                    total_violations -= 1
                                is_violated[idx] = new_violated
                        if total_violations < best_violations:
                            best_violations = total_violations
                            best_colors = colors[:]
                            if best_violations == 0:
                                return best_colors
                        T *= cooling
                # Else do nothing
            else:
                # Accept better or equal move
                colors[i] = best_new_c
                # Update state
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    old_violated = is_violated[idx]
                    new_violated = len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3
                    if old_violated != new_violated:
                        if new_violated:
                            total_violations += 1
                        else:
                            total_violations -= 1
                        is_violated[idx] = new_violated
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
                T *= cooling
        
        return best_colors if best_violations == 0 else None

    # Try k=3
    result = minimize_colors(3)
    if result:
        return result
        
    # Try k=4
    result = minimize_colors(4)
    if result:
        return result
        
    # Fallback: Greedy construction with lookahead or just random restarts?
    # The seed program's fallback is a greedy that might fail.
    # Let's try a simple greedy that picks the smallest color that doesn't immediately violate.
    # This is the same as the seed's fallback which scored 1.0 (meaning it returned 128 colors or failed to minimize?).
    # Wait, the seed program returns `colors` which are in 0..127.
    # If the greedy fails to assign a color, it might crash or return something invalid.
    # The task says "Minimize the number of labels actually used".
    # If we return a list of length 128 with values 0..127, the number of distinct labels is the score.
    # The seed program uses up to 128 colors, so its score is 128?
    # No, the score is "seed/value". If seed is 128 and value is 128, score is 1.0.
    # If we find a solution with 3 colors, value is 3, score is 128/3 = 42.6.
    # So we want to minimize the number of distinct colors.
    
    # If local search fails, we should return the best valid coloring we found, even if it uses more colors.
    # But minimize_colors returns None if it doesn't find a valid one with k colors.
    # We should keep track of the best valid coloring found during the search, even if it has violations?
    # No, the result MUST be valid.
    # If we can't find a valid one with k=3 or k=4, we should try k=5, etc.
    # Or return a valid coloring with more colors.
    
    # Let's try increasing k until we find a solution.
    for k in range(5, 129):
        if time.time() > deadline:
            break
        result = minimize_colors(k)
        if result:
            return result
            
    # Absolute fallback: Use 128 colors, one per position. This is always valid.
    # Positions 0..127 get colors 0..127.
    # Any 4-AP has 4 distinct positions, thus 4 distinct colors.
    return list(range(128))

# To make the backtracking faster, we need better pruning.
# The current implementation might be too slow for n=128 with k=3.
# Let's add a time check inside the loop.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        progressions_ending_at[d].append((a, b, c))
        
    def solve(k):
        colors = [-1] * n
        colors[0] = 0
        
        def backtrack(i):
            if time.time() > deadline:
                return False, None
            
            if i == n:
                return True, colors[:]
            
            for c in range(k):
                valid = True
                for (a, b, c_idx) in progressions_ending_at[i]:
                    cols = {colors[a], colors[b], colors[c_idx], c}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[i] = c
                ok, res = backtrack(i + 1)
                if ok:
                    return True, res
                colors[i] = -1
                
            return False, None
        
        # Symmetry breaking
        colors[1] = 0
        ok, res = backtrack(2)
        if ok:
            return res
        
        if k >= 2:
            colors[1] = 1
            ok, res = backtrack(2)
            if ok:
                return res
                
        return None

    result = solve(3)
    if result:
        return result
        
    result = solve(4)
    if result:
        return result
        
    # Fallback
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# This simple backtracking is likely too slow.
# We need to use the "fail-first" heuristic or constraint propagation more aggressively.
# Or, we can use a SAT solver approach if available? No, only stdlib/np/scipy.
# Let's try to improve the pruning by checking forward constraints.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    # For each index i, store progressions where i is a, b, or c.
    # prog_involving[i] = list of (pos, a, b, c, d) where pos is 0,1,2 for a,b,c
    prog_involving = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        prog_involving[a].append((0, a, b, c, d))
        prog_involving[b].append((1, a, b, c, d))
        prog_involving[c].append((2, a, b, c, d))
        
    def solve(k):
        colors = [-1] * n
        colors[0] = 0
        
        def backtrack(i):
            if time.time() > deadline:
                return False, None
            
            if i == n:
                return True, colors[:]
            
            # Heuristic: Try colors that are used less frequently? Or just 0..k-1
            for c in range(k):
                # Check backward constraints (progressions ending at i)
                valid = True
                for (a, b, c_idx) in progressions_ending_at[i]:
                    cols = {colors[a], colors[b], colors[c_idx], c}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check forward constraints (progressions where i is a, b, or c)
                # For each such progression, if the other elements are assigned,
                # check if a valid color exists for the unassigned element.
                
                # Simplified: Just check if the assignment creates an immediate conflict
                # with any already assigned element in the same progression.
                # This is already covered by backward constraints if we process in order?
                # No, backward constraints only check progressions where i is the LAST element.
                # If i is the FIRST element, the progression is not complete.
                
                # Let's do a light forward check:
                # For each progression (a, b, c, d) where i is a, b, or c:
                # If d is assigned, check validity.
                # If d is not assigned, check if the current partial assignment 
                # restricts d to an empty set.
                
                # To make it fast, we only check if the set of assigned colors in {a,b,c}
                # has size 3. If it does, d can be anything.
                # If size 2, d must be a 3rd color.
                # If size 1, d must be a 2nd color.
                
                # This check is necessary but not sufficient.
                
                # Given the time limit, let's rely on the fact that k=3 is very constraining
                # and the search space might be small enough with symmetry breaking.
                
                colors[i] = c
                ok, res = backtrack(i + 1)
                if ok:
                    return True, res
                colors[i] = -1
                
            return False, None
        
        # Symmetry breaking
        colors[1] = 0
        ok, res = backtrack(2)
        if ok:
            return res
        
        if k >= 2:
            colors[1] = 1
            ok, res = backtrack(2)
            if ok:
                return res
                
        return None

    # Precompute progressions_ending_at
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        progressions_ending_at[d].append((a, b, c))
        
    result = solve(3)
    if result:
        return result
        
    result = solve(4)
    if result:
        return result
        
    # Fallback
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# The above code is still likely too slow.
# Let's try a different approach: Randomized Local Search (Simulated Annealing or WalkSAT style).
# Objective: Minimize number of distinct colors.
# Constraint: Every 4-AP has >= 3 colors.
# We can start with a random coloring using k colors and try to reduce k or fix violations.
# Or start with k=3 and try to find a valid coloring. If not, increase k.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    def count_violations(colors):
        v = 0
        for (a, b, c, d) in progressions:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                v += 1
        return v
        
    def minimize_colors(k, max_iters=100000):
        # Random initialization
        import random
        random.seed(42)
        colors = [random.randint(0, k-1) for _ in range(n)]
        
        best_colors = colors[:]
        best_violations = count_violations(colors)
        
        if best_violations == 0:
            return colors
            
        # Local search: pick a random position and change its color
        # to the one that reduces violations the most.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            i = random.randint(0, n-1)
            old_c = colors[i]
            
            best_new_c = old_c
            best_delta = 0
            
            # Calculate delta for each possible new color
            # To be fast, we can just pick a random new color and accept if it improves.
            # Or try all k colors.
            
            # Let's try all k colors
            min_violations = best_violations
            for c in range(k):
                if c == old_c:
                    continue
                colors[i] = c
                v = count_violations(colors)
                if v < min_violations:
                    min_violations = v
                    best_new_c = c
            colors[i] = best_new_c
            
            if min_violations < best_violations:
                best_violations = min_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return best_colors
            else:
                # Revert if not improved (greedy local search)
                # Or accept with some probability (simulated annealing)
                # Let's stick to greedy for simplicity
                colors[i] = old_c
                
        return best_colors if best_violations == 0 else None

    # Try k=3
    result = minimize_colors(3)
    if result:
        return result
        
    # Try k=4
    result = minimize_colors(4)
    if result:
        return result
        
    # Fallback to greedy
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# The local search might be slow if count_violations is O(n^2).
# n=128, progressions ~ 128*64 = 8192.
# 100000 iters * 8192 checks = 8e8 operations. Too slow in Python.
# We need to update violations incrementally.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    # For each position i, store the list of progressions it belongs to.
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[d].append(idx)
        
    def count_violations(colors):
        v = 0
        for (a, b, c, d) in progressions:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                v += 1
        return v
        
    def minimize_colors(k):
        import random
        random.seed(42)
        colors = [random.randint(0, k-1) for _ in range(n)]
        violations = count_violations(colors)
        
        if violations == 0:
            return colors
            
        # Incremental update
        # When we change color at i, only progressions containing i are affected.
        
        def get_violation_count_for_prog(a, b, c, d):
            return 0 if len({colors[a], colors[b], colors[c], colors[d]}) >= 3 else 1
            
        best_colors = colors[:]
        best_violations = violations
        
        max_iters = 50000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            i = random.randint(0, n-1)
            old_c = colors[i]
            
            # Calculate current violations for progressions involving i
            # This is expensive to do from scratch.
            # Let's just do a simple hill climbing with full recalculation for simplicity 
            # but limit iterations.
            
            # Actually, let's try a different heuristic:
            # Pick a random progression that is violated, and change one of its colors.
            
            # Find a violated progression
            violated_progs = []
            for idx, (a, b, c, d) in enumerate(progressions):
                if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                    violated_progs.append(idx)
                    
            if not violated_progs:
                return colors
                
            # Pick one randomly
            p_idx = random.choice(violated_progs)
            a, b, c, d = progressions[p_idx]
            
            # Try changing one of the 4 positions to a different color
            best_improvement = 0
            best_change = None
            
            for pos in [a, b, c, d]:
                old_color = colors[pos]
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    colors[pos] = new_color
                    # Recalculate violations for this progression and others?
                    # Just check this progression
                    if len({colors[a], colors[b], colors[c], colors[d]}) >= 3:
                        # Check if this change creates new violations elsewhere
                        # This is the hard part.
                        # Let's just accept if it fixes this one and hope it doesn't break others too much.
                        # Or do a full check.
                        pass
                    colors[pos] = old_color
                    
            # This is getting complicated. Let's stick to the simple backtracking 
            # with a very tight time limit and see if it finds a solution for k=3.
            # If not, fallback.
            
        return best_colors if best_violations == 0 else None

    # Try k=3 with backtracking
    # (Reusing the backtracking code from before, but with a stricter time limit)
    
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        progressions_ending_at[d].append((a, b, c))
        
    def solve(k):
        colors = [-1] * n
        colors[0] = 0
        
        def backtrack(i):
            if time.time() > deadline:
                return False, None
            
            if i == n:
                return True, colors[:]
            
            for c in range(k):
                valid = True
                for (a, b, c_idx) in progressions_ending_at[i]:
                    cols = {colors[a], colors[b], colors[c_idx], c}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[i] = c
                ok, res = backtrack(i + 1)
                if ok:
                    return True, res
                colors[i] = -1
                
            return False, None
        
        colors[1] = 0
        ok, res = backtrack(2)
        if ok:
            return res
        
        if k >= 2:
            colors[1] = 1
            ok, res = backtrack(2)
            if ok:
                return res
                
        return None

    result = solve(3)
    if result:
        return result
        
    result = solve(4)
    if result:
        return result
        
    # Fallback
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# This is a mess. Let's clean it up and focus on the most promising strategy:
# Backtracking with constraint propagation.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        progressions_ending_at[d].append((a, b, c))
        
    def solve(k):
        colors = [-1] * n
        colors[0] = 0
        
        def backtrack(i):
            if time.time() > deadline:
                return False, None
            
            if i == n:
                return True, colors[:]
            
            for c in range(k):
                valid = True
                for (a, b, c_idx) in progressions_ending_at[i]:
                    cols = {colors[a], colors[b], colors[c_idx], c}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[i] = c
                ok, res = backtrack(i + 1)
                if ok:
                    return True, res
                colors[i] = -1
                
            return False, None
        
        # Symmetry breaking
        colors[1] = 0
        ok, res = backtrack(2)
        if ok:
            return res
        
        if k >= 2:
            colors[1] = 1
            ok, res = backtrack(2)
            if ok:
                return res
                
        return None

    result = solve(3)
    if result:
        return result
        
    result = solve(4)
    if result:
        return result
        
    # Fallback
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# The backtracking is likely too slow.
# Let's try the local search with incremental violation counting.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[d].append(idx)
        
    def count_violations(colors):
        v = 0
        for (a, b, c, d) in progressions:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                v += 1
        return v
        
    def minimize_colors(k):
        import random
        random.seed(42)
        colors = [random.randint(0, k-1) for _ in range(n)]
        violations = count_violations(colors)
        
        if violations == 0:
            return colors
            
        best_colors = colors[:]
        best_violations = violations
        
        # Precompute violation status for each progression
        is_violated = [False] * len(progressions)
        for idx, (a, b, c, d) in enumerate(progressions):
            is_violated[idx] = len({colors[a], colors[b], colors[c], colors[d]}) < 3
            
        for _ in range(100000):
            if time.time() > deadline:
                break
                
            # Pick a random position
            i = random.randint(0, n-1)
            old_c = colors[i]
            
            # Find best new color for i
            best_new_c = old_c
            best_delta = 0
            
            # Calculate delta for each new color
            # Only progressions containing i are affected
            affected_progs = prog_at[i]
            
            # Current violations for affected progs
            current_violations = sum(1 for idx in affected_progs if is_violated[idx])
            
            for c in range(k):
                if c == old_c:
                    continue
                colors[i] = c
                
                new_violations = 0
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    if len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3:
                        new_violations += 1
                        
                delta = new_violations - current_violations
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = c
                    
            colors[i] = best_new_c
            
            # Update is_violated and violations count
            if best_new_c != old_c:
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    old_violated = is_violated[idx]
                    new_violated = len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3
                    if old_violated != new_violated:
                        if new_violated:
                            violations += 1
                        else:
                            violations -= 1
                        is_violated[idx] = new_violated
                        
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
                        
        return best_colors if best_violations == 0 else None

    result = minimize_colors(3)
    if result:
        return result
        
    result = minimize_colors(4)
    if result:
        return result
        
    # Fallback
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

# This local search is more likely to find a solution within the time limit.
# The incremental update makes it fast enough.

# The code above is a mess of multiple definitions. The last one wins.
# Let's clean up and provide a single, robust implementation.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[d].append(idx)
        
    def minimize_colors(k):
        import random
        random.seed(42)
        colors = [random.randint(0, k-1) for _ in range(n)]
        
        is_violated = [False] * len(progressions)
        total_violations = 0
        for idx, (a, b, c, d) in enumerate(progressions):
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                is_violated[idx] = True
                total_violations += 1
        
        if total_violations == 0:
            return colors
            
        best_colors = colors[:]
        best_violations = total_violations
        
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        
        max_iters = 200000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            i = random.randint(0, n-1)
            old_c = colors[i]
            
            best_new_c = old_c
            best_delta = 0
            
            affected_progs = prog_at[i]
            current_violations = sum(1 for idx in affected_progs if is_violated[idx])
            
            for c in range(k):
                if c == old_c:
                    continue
                colors[i] = c
                
                new_violations = 0
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    if len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3:
                        new_violations += 1
                        
                delta = new_violations - current_violations
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = c
                    
            colors[i] = old_c
            
            if best_new_c == old_c:
                if best_delta > 0 and T > min_T:
                    prob = pow(2.718281828, -best_delta / T)
                    if random.random() < prob:
                        colors[i] = best_new_c
                        for idx in affected_progs:
                            a, b, c_idx, d = progressions[idx]
                            old_violated = is_violated[idx]
                            new_violated = len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3
                            if old_violated != new_violated:
                                if new_violated:
                                    total_violations += 1
                                else:
                                    total_violations -= 1
                                is_violated[idx] = new_violated
                        if total_violations < best_violations:
                            best_violations = total_violations
                            best_colors = colors[:]
                            if best_violations == 0:
                                return best_colors
                        T *= cooling
            else:
                colors[i] = best_new_c
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    old_violated = is_violated[idx]
                    new_violated = len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3
                    if old_violated != new_violated:
                        if new_violated:
                            total_violations += 1
                        else:
                            total_violations -= 1
                        is_violated[idx] = new_violated
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
                T *= cooling
        
        return best_colors if best_violations == 0 else None

    for k in range(3, 129):
        if time.time() > deadline:
            break
        result = minimize_colors(k)
        if result:
            return result
            
    return list(range(128))
