import time
import random
from itertools import combinations

def _largest_sidon(s):
    """Exact largest subset of s with all unordered pair sums (incl. 2x) unique."""
    n = len(s)
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        # Two elements a,b: sums are 2a, a+b, 2b. Always Sidon if a!=b
        return 2
    
    s = sorted(s)
    best = 0
    
    # Precompute pair sums for quick lookup
    pair_sum = [[s[i] + s[j] for j in range(n)] for i in range(n)]
    
    def bt(idx, chosen, sumset):
        nonlocal best
        # Bound: even adding all remaining can't beat best
        rem = n - idx
        if len(chosen) + rem <= best:
            return
        if idx == n:
            if len(chosen) > best:
                best = len(chosen)
            return
        
        # Try to take idx
        v = s[idx]
        ok = True
        new_sums = []
        if (v + v) in sumset:
            ok = False
        else:
            new_sums.append(v + v)
            for j in chosen:
                ss = v + s[j]
                if ss in sumset:
                    ok = False
                    break
                new_sums.append(ss)
        
        if ok:
            for ns in new_sums:
                sumset.add(ns)
            chosen.append(idx)
            bt(idx + 1, chosen, sumset)
            chosen.pop()
            for ns in new_sums:
                sumset.remove(ns)
        
        # Skip idx
        bt(idx + 1, chosen, sumset)

    bt(0, [], set())
    return best

def _count_sidon_k(A, k):
    """Count k-element Sidon subsets of A."""
    count = 0
    for subset in combinations(A, k):
        sums = set()
        is_sidon = True
        for i in range(len(subset)):
            ai = subset[i]
            for j in range(i, len(subset)):
                s = ai + subset[j]
                if s in sums:
                    is_sidon = False
                    break
                sums.add(s)
            if not is_sidon:
                break
        if is_sidon:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin under 90s limit
    
    # Structured initial candidates - dense sets likely have small max Sidon
    structured_candidates = [
        list(range(12)),           # 0-11 AP
        [i * 2 for i in range(12)],  # 0,2,4,...,22
        [i * 3 for i in range(12)],  # 0,3,6,...,33
        [i * 5 for i in range(12)],  # 0,5,10,...,55
        list(range(12)) + [12,13][:0],  # just 0-11
        [0,1,2,3,4,5,6,7,8,9,10,11],
        [1,2,3,4,5,6,7,8,9,10,11,12],
        [0,1,3,4,6,7,9,10,12,13,15,16],  # clustered
        [0,1,2,4,5,6,8,9,10,12,13,14],   # block pattern
        [0,1,2,3,5,6,7,8,10,11,12,13],   # block pattern 2
    ]
    
    best_set = None
    best_val = 10**9
    
    def try_set(vals):
        nonlocal best_set, best_val
        vals = sorted(set(vals))
        if len(vals) != 12:
            return
        v = _largest_sidon(vals)
        if v < best_val:
            best_val = v
            best_set = list(vals)
            if best_val <= 3:
                return True  # Can't do much better
        return False
    
    # Evaluate structured candidates
    for sc in structured_candidates:
        if try_set(sc):
            return best_set
    
    # If we found 3, we're done
    if best_val == 3:
        return best_set
    
    # Local search from best structured candidate
    current = list(best_set) if best_set else list(range(12))
    current_val = best_val
    
    rng = random.Random(42)
    
    def get_neighbor(A):
        A = list(A)
        idx = rng.randint(0, len(A) - 1)
        old_val = A[idx]
        
        # Strategy: mostly local moves to stay dense
        if rng.random() < 0.8:
            # Small perturbation
            delta = rng.choice([-2,-1,1,2])
            new_val = old_val + delta
            if new_val < 0:
                new_val = old_val - delta
            if new_val > 128:
                new_val = old_val - delta
        else:
            # Medium move
            new_val = rng.randint(max(0, old_val - 15), min(128, old_val + 15))
        
        if new_val in A and new_val != old_val:
            # Try to find a valid replacement
            for _ in range(30):
                new_val = rng.randint(0, 128)
                if new_val not in A:
                    break
            else:
                return None
        
        A[idx] = new_val
        return A
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        neighbor = get_neighbor(current)
        if neighbor is None:
            no_improve += 1
            if no_improve > max_no_improve:
                # Random restart from a structured candidate
                current = list(rng.choice(structured_candidates))
                current_val = _largest_sidon(current)
                no_improve = 0
            continue
        
        val = _largest_sidon(neighbor)
        
        if val < current_val:
            current = neighbor
            current_val = val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_set = list(current)
                if best_val <= 3:
                    break
        elif val == current_val:
            # Accept equal with small probability to explore
            if rng.random() < 0.05:
                current = neighbor
            no_improve += 1
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Restart
            if rng.random() < 0.5:
                current = list(rng.choice(structured_candidates))
                current_val = _largest_sidon(current)
            else:
                # Perturb best found
                current = list(best_set)
                for _ in range(rng.randint(2, 5)):
                    idx = rng.randint(0, 11)
                    new_val = rng.randint(0, 128)
                    if new_val not in current:
                        current[idx] = new_val
                current_val = _largest_sidon(current)
            no_improve = 0
    
    return best_set if best_set else current
