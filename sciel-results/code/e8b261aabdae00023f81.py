import math
import random
import time

def _conflict(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g == 0

def _is_valid(pairs):
    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            r1, m1 = pairs[i]
            r2, m2 = pairs[j]
            if _conflict(r1, m1, r2, m2):
                return False
    return True

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s
    
    # All candidate (residue, modulus) pairs
    candidates = []
    for m in range(2, 129):
        for r in range(m):
            candidates.append((r, m))
    
    # Precompute conflict bitsets for speed
    n = len(candidates)
    # conflict[i] = set of j that conflict with i
    # Use bitsets via integers for speed
    conflict = [0] * n
    for i in range(n):
        r1, m1 = candidates[i]
        for j in range(i+1, n):
            r2, m2 = candidates[j]
            g = math.gcd(m1, m2)
            if (r1 - r2) % g == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    best = [[1, 2]]  # at least one valid pair
    best_len = 1
    
    # Greedy with multiple orderings
    def greedy_order(order):
        selected = []
        selected_indices = set()
        for idx in order:
            r, m = candidates[idx]
            # Check against all selected
            ok = True
            for sidx in selected_indices:
                if (conflict[idx] >> sidx) & 1:
                    ok = False
                    break
            if ok:
                selected.append((r, m))
                selected_indices.add(idx)
        return selected
    
    # Try several random orderings
    indices = list(range(n))
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        random.shuffle(indices)
        result = greedy_order(indices)
        if len(result) > best_len:
            best = result
            best_len = len(result)
            # Try local improvement
            improved = True
            while improved:
                if time.time() > deadline - 5:
                    break
                improved = False
                # Try adding any unselected candidate
                sel_set = set()
                for (r, m) in best:
                    for idx in range(n):
                        if candidates[idx] == (r, m):
                            sel_set.add(idx)
                            break
                for idx in range(n):
                    if idx in sel_set:
                        continue
                    ok = True
                    for sidx in sel_set:
                        if (conflict[idx] >> sidx) & 1:
                            ok = False
                            break
                    if ok:
                        best.append(candidates[idx])
                        best_len += 1
                        improved = True
                        break
                if not improved:
                    # Try swap: remove one, add two
                    for i in range(len(best)):
                        if time.time() > deadline - 5:
                            break
                        removed = best[i]
                        best_no_i = best[:i] + best[i+1:]
                        sel_set2 = set()
                        for (r, m) in best_no_i:
                            for idx in range(n):
                                if candidates[idx] == (r, m):
                                    sel_set2.add(idx)
                                    break
                        # Try adding two
                        added = 0
                        for idx in range(n):
                            if idx in sel_set2:
                                continue
                            ok = True
                            for sidx in sel_set2:
                                if (conflict[idx] >> sidx) & 1:
                                    ok = False
                                    break
                            if ok:
                                best_no_i.append(candidates[idx])
                                sel_set2.add(idx)
                                added += 1
                                if added == 2:
                                    break
                        if added >= 2:
                            best = best_no_i
                            best_len = len(best)
                            improved = True
                            break
    
    # Return in the required format
    return [[r, m] for (r, m) in best]
