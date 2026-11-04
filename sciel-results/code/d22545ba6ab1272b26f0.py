import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 150s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task is specific
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback: Direct sum or random
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists as numpy arrays of int32 for fast operations
    neighbors_list = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))

    # ---------- Greedy Set-Cover Construction ----------
    # Start with an empty code and greedily add codewords that cover the most
    # currently uncovered words. This avoids the potentially suboptimal structure
    # of the direct-sum construction and often yields a smaller initial code.
    
    coverage = np.zeros(N, dtype=np.int32)
    uncovered_mask = np.ones(N, dtype=bool)
    code = []
    
    # Precompute neighbor sets as Python sets for faster intersection counting
    # in the greedy phase (only needed during construction)
    neighbor_sets = [set(nb.tolist()) for nb in neighbors_list]
    
    while uncovered_mask.any() and time.time() < deadline:
        # Find the codeword that covers the most uncovered words
        best_c = -1
        best_count = 0
        
        # We can optimize by only checking codewords that have at least one
        # uncovered neighbor, but for N=2187 a full scan is fast enough
        # and numpy can help with counting.
        
        # Vectorized approach: For each candidate c, count how many of its neighbors
        # are currently uncovered.
        # Since neighbors_list is a list of arrays, we can do this in a loop
        # but use numpy sum for speed.
        
        for c in range(N):
            if best_count == N:  # Can't do better
                break
            # Count uncovered neighbors
            # neighbors_list[c] is an array of indices
            # uncovered_mask is a boolean array
            cnt = np.sum(uncovered_mask[neighbors_list[c]])
            if cnt > best_count:
                best_count = cnt
                best_c = c
                if best_count == 0:
                    break  # No more uncovered words can be covered by any codeword?
                           # This shouldn't happen if uncovered_mask.any() is True,
                           # unless there's a bug. But if best_count is 0, we're stuck.
        
        if best_c == -1 or best_count == 0:
            # Should not happen if uncovered_mask.any() is True
            # But just in case, break to avoid infinite loop
            break
        
        # Add best_c to code
        code.append(best_c)
        # Update coverage and uncovered_mask
        coverage[neighbors_list[best_c]] += 1
        uncovered_mask[neighbors_list[best_c]] = False
    
    # ---------- Local Search: Removal and Kick ----------
    
    def is_redundant(c):
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set):
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            # Build candidate pool from neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbor_sets[w])
            
            # Find best candidate
            for c in candidate_pool:
                if c in code:
                    continue
                cnt = 0
                for x in neighbors_list[c]:
                    if x in temp_uncovered:
                        cnt += 1
                        if cnt > best_count:
                            best_count = cnt
                            best_c = c
                        if best_count == len(temp_uncovered):
                            break
                if best_count == len(temp_uncovered):
                    break
            
            if best_c == -1:
                break
            
            added.append(best_c)
            for w in neighbors_list[best_c]:
                temp_uncovered.discard(w)
        
        return added

    def try_remove():
        nonlocal code, coverage
        removed = False
        for c in list(code):
            if is_redundant(c):
                code.remove(c)
                coverage[neighbors_list[c]] -= 1
                removed = True
        return removed

    def kick():
        nonlocal code, coverage
        if not code:
            return False
            
        old_size = len(code)
        
        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]
        
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)
        
        new_size = old_size - 1 + len(added)
        
        if new_size < old_size:
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            for c in added:
                code.append(c)
                coverage[neighbors_list[c]] += 1
            return True
        else:
            return False

    # Main Loop
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = try_remove()
        
        # 2. Kick
        if time.time() > deadline:
            break
        
        kicks_tried = 0
        success = False
        max_kicks = 20
        while kicks_tried < max_kicks:
            if kick():
                success = True
                kicks_tried = 0
            else:
                kicks_tried += 1
            if time.time() > deadline:
                break
        
        if not success and len(code) > 2:
            # 2-removal kick
            i1 = random.randint(0, len(code) - 1)
            i2 = random.randint(0, len(code) - 1)
            if i1 == i2:
                i2 = (i2 + 1) % len(code)
            
            c1 = code[i1]
            c2 = code[i2]
            
            temp_coverage = coverage.copy()
            temp_coverage[neighbors_list[c1]] -= 1
            temp_coverage[neighbors_list[c2]] -= 1
            
            uncovered = np.where(temp_coverage == 0)[0]
            
            if len(uncovered) > 0:
                uncovered_set = set(uncovered.tolist())
                added = greedy_repair(uncovered_set)
                
                new_size = len(code) - 2 + len(added)
                
                if new_size < len(code):
                    code.remove(c1)
                    code.remove(c2)
                    coverage[neighbors_list[c1]] -= 1
                    coverage[neighbors_list[c2]] -= 1
                    for c in added:
                        if c not in code:
                            code.append(c)
                            coverage[neighbors_list[c]] += 1

    result = [idx_to_tuple(c) for c in code]
    return result
