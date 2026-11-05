"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 84  # safety margin before 90s kill
    
    n_max = 4096
    best = []
    best_len = 0
    
    # Precompute for speed: use a set for O(1) lookups of sums
    # We'll maintain sums_set as a Python set of ints
    
    def try_add(selected, sums_set, x):
        """Try adding x to selected. Returns True if successful, updates sums_set."""
        # Check all triples involving x
        # (x,x,x)
        s = 3 * x
        if s in sums_set:
            return False
        
        # (x,x,a) for each a in selected
        for a in selected:
            s = 2 * x + a
            if s in sums_set:
                return False
        
        # (x,a,b) for each a<=b in selected
        n = len(selected)
        for i in range(n):
            ai = selected[i]
            for j in range(i, n):
                s = x + ai + selected[j]
                if s in sums_set:
                    return False
        
        # All clear, add the sums
        sums_set.add(3 * x)
        for a in selected:
            sums_set.add(2 * x + a)
        for i in range(n):
            ai = selected[i]
            for j in range(i, n):
                sums_set.add(x + ai + selected[j])
        return True
    
    def remove_element_sums(selected, sums_set, idx):
        """Remove all triple sums involving selected[idx]. Returns set of removed sums."""
        x = selected[idx]
        n = len(selected)
        removed = set()
        removed.add(3 * x)
        for i in range(n):
            if i == idx:
                continue
            a = selected[i]
            s = 2 * x + a
            removed.add(s)
        for i in range(n):
            if i == idx:
                continue
            ai = selected[i]
            for j in range(i, n):
                if j == idx:
                    continue
                s = x + ai + selected[j]
                removed.add(s)
        for s in removed:
            sums_set.discard(s)
        return removed
    
    def greedy_build(start_val, step_fn=None, max_iter=50000):
        """Greedy build with optional step function."""
        selected = []
        sums_set = set()
        count = 0
        if step_fn is None:
            # Linear scan
            x = start_val
            while x <= n_max:
                if time.time() > deadline:
                    break
                if try_add(selected, sums_set, x):
                    selected.append(x)
                x += 1
        else:
            x = start_val
            while x <= n_max and count < max_iter:
                if time.time() > deadline:
                    break
                if try_add(selected, sums_set, x):
                    selected.append(x)
                x = step_fn(x)
                count += 1
        return selected, sums_set
    
    def backtrack_search(selected, sums_set, start_idx, max_depth, budget):
        """Backtracking: try to extend by choosing from a window of candidates.
        At each level, try several candidates and recurse."""
        nonlocal best, best_len
        if time.time() > deadline:
            return
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(best)
        
        if len(selected) >= max_depth:
            return
        
        # Try a window of candidate values starting from start_idx
        # Use the last element + 1 as minimum to keep sorted
        min_val = (selected[-1] + 1) if selected else 1
        # Try a small window of candidates
        window_size = 8 if len(selected) < 15 else 4
        candidates = list(range(min_val, min(min_val + window_size, n_max + 1)))
        
        for x in candidates:
            if time.time() > deadline:
                return
            if try_add(selected, sums_set, x):
                selected.append(x)
                backtrack_search(selected, sums_set, start_idx + 1, max_depth, budget)
                selected.pop()
                # Remove sums of x
                n = len(selected)
                sums_set.discard(3 * x)
                for a in selected:
                    sums_set.discard(2 * x + a)
                for i in range(n):
                    ai = selected[i]
                    for j in range(i, n):
                        sums_set.discard(x + ai + selected[j])
            budget[0] -= 1
            if budget[0] <= 0:
                return
    
    def local_search(selected, sums_set, max_iters=200):
        """Try to improve by swapping elements: remove one, try to add a larger one."""
        nonlocal best, best_len
        if time.time() > deadline:
            return selected, sums_set
        
        improved = True
        iteration = 0
        while improved and iteration < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1
            n = len(selected)
            # Try removing each element and replacing with a larger one
            for i in range(n - 1, -1, -1):  # try removing from the end first
                if time.time() > deadline:
                    break
                x = selected[i]
                # Remove x
                remove_element_sums(selected, sums_set, i)
                selected.pop(i)
                
                # Try to add a larger value
                found = False
                for new_x in range(x + 1, min(x + 200, n_max + 1)):
                    if try_add(selected, sums_set, new_x):
                        selected.append(new_x)
                        selected.sort()
                        found = True
                        improved = True
                        break
                
                if not found:
                    # Put x back
                    selected.insert(i, x)
                    # Re-add sums
                    n2 = len(selected)
                    sums_set.add(3 * x)
                    for j in range(n2):
                        if j == i:
                            continue
                        a = selected[j]
                        sums_set.add(2 * x + a)
                    for j in range(n2):
                        if j == i:
                            continue
                        aj = selected[j]
                        for k in range(j, n2):
                            if k == i:
                                continue
                            sums_set.add(x + aj + selected[k])
        
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(best)
        
        return selected, sums_set
    
    # === Strategy 1: Linear greedy from 1 ===
    selected, sums_set = greedy_build(1)
    if len(selected) > best_len:
        best = list(selected)
        best_len = len(best)
    
    # === Strategy 2: Greedy from various offsets ===
    random.seed(42)
    for restart in range(80):
        if time.time() > deadline:
            break
        offset = random.randint(1, 300)
        selected, sums_set = greedy_build(offset)
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(best)
    
    # === Strategy 3: Backtracking search ===
    # Start from a greedy solution and try to extend via backtracking
    for restart in range(20):
        if time.time() > deadline:
            break
        offset = random.randint(1, 100)
        selected, sums_set = greedy_build(offset)
        # Now try backtracking to extend
        budget = [500]
        backtrack_search(selected, sums_set, 0, len(selected) + 5, budget)
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(best)
    
    # === Strategy 4: Exponential spacing ===
    for restart in range(25):
        if time.time() > deadline:
            break
        base = random.uniform(1.3, 3.0)
        def step_fn(x, base=base):
            return int(x * base) + 1
        selected, sums_set = greedy_build(1, step_fn)
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(best)
    
    # === Strategy 5: Local search polish on best so far ===
    if best:
        # Rebuild sums_set for best
        sums_set = set()
        n = len(best)
        for i in range(n):
            ai = best[i]
            for j in range(i, n):
                aj = best[j]
                for k in range(j, n):
                    ak = best[k]
                    sums_set.add(ai + aj + ak)
        local_search(list(best), sums_set, max_iters=300)
    
    # === Strategy 6: More diverse greedy with random skips ===
    for restart in range(40):
        if time.time() > deadline:
            break
        selected = []
        sums_set = set()
        current = random.randint(1, 50)
        while current <= n_max:
            if time.time() > deadline:
                break
            if try_add(selected, sums_set, current):
                selected.append(current)
            skip = random.randint(1, 15)
            current += skip
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(best)
    
    return best
# EVOLVE-BLOCK-END
