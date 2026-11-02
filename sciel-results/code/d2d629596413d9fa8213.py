import time
import math
import random
import sys

_T0 = time.time()
DEADLINE = _T0 + 140.0

def _time_left():
    return DEADLINE - time.time()

def _score_set(U):
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    if n < 2:
        return 1.0

    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _is_golomb(U):
    U = sorted(U)
    diffs = set()
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _greedy_golomb(n, seed):
    rng = random.Random(seed)
    U = [0]
    diffs = set()
    for _ in range(1, n):
        found = False
        # Try a range of candidates
        # Start from last + 1, but randomize order slightly or just scan
        # Scanning is fast for small n
        lo = U[-1] + 1
        # Upper bound heuristic
        hi = lo + n * 10
        candidates = list(range(lo, hi))
        rng.shuffle(candidates)
        for x in candidates:
            ok = True
            for a in U:
                d = x - a
                if d in diffs:
                    ok = False
                    break
            if ok:
                U.append(x)
                for a in U[:-1]:
                    diffs.add(x - a)
                found = True
                break
        if not found:
            # Expand search
            for x in range(lo, lo + 1000):
                ok = True
                for a in U:
                    d = x - a
                    if d in diffs:
                        ok = False
                        break
                if ok:
                    U.append(x)
                    for a in U[:-1]:
                        diffs.add(x - a)
                    found = True
                    break
        if not found:
            break
    return U

def _local_search_golomb(U, max_iters=1000, seed=0):
    rng = random.Random(seed)
    U = U[:]
    n = len(U)
    if n <= 1:
        return U
    best_max = U[-1]
    
    # Precompute diffs for current U
    diffs = set()
    for i in range(n):
        for j in range(i+1, n):
            diffs.add(U[j] - U[i])
            
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        i = rng.randint(1, n - 1)
        # Try to move U[i]
        # Direction: mostly try to decrease if i == n-1, else random
        if i == n - 1:
            deltas = [-1, -2, -3, -5, -10, -20, 1, 2]
        else:
            deltas = [-1, 1, -2, 2, -5, 5]
        rng.shuffle(deltas)
        
        moved = False
        for d in deltas:
            new_val = U[i] + d
            # Bounds check
            if i == 0:
                continue
            if new_val < U[i-1] or (i < n-1 and new_val > U[i+1]):
                continue
            if new_val == U[i]:
                continue
            
            # Check validity
            # Remove old diffs involving i
            old_diffs = set()
            for j in range(n):
                if j != i:
                    if U[i] > U[j]:
                        old_diffs.add(U[i] - U[j])
                    else:
                        old_diffs.add(U[j] - U[i])
            
            new_diffs = set()
            for j in range(n):
                if j != i:
                    if new_val > U[j]:
                        new_diffs.add(new_val - U[j])
                    else:
                        new_diffs.add(U[j] - new_val)
            
            # Check conflicts
            # New diffs must not be in (old_diffs - old_diffs_involving_i)
            # Actually, simpler:
            # 1. new_diffs must be disjoint from (diffs - old_diffs)
            # 2. new_diffs must be disjoint from each other (which they are, as they are differences from a single point to others, and U is sorted distinct)
            # Wait, differences from new_val to U[j] are distinct for distinct j? Yes, because U[j] are distinct.
            # So we just need to check if any d in new_diffs is in (diffs - old_diffs).
            
            remaining_diffs = diffs - old_diffs
            valid = True
            for d in new_diffs:
                if d in remaining_diffs:
                    valid = False
                    break
            
            if valid:
                # Accept move
                # Update U
                U[i] = new_val
                # Update diffs
                diffs = (diffs - old_diffs) | new_diffs
                # U is no longer sorted if we moved i?
                # We maintained U[i-1] < new_val < U[i+1], so it's still sorted.
                moved = True
                if i == n - 1 and new_val < best_max:
                    best_max = new_val
                break
                
        if not moved and rng.random() < 0.1:
             # Restart or jump?
             # Try a bigger jump
             pass

    return U

def construct():
    start = time.time()
    deadline = DEADLINE
    
    best_score = 0.0
    best_U = [0]
    
    # Try different sizes n
    for n in range(3, 31):
        if time.time() > deadline - 10:
            break
            
        # Try multiple seeds for greedy
        best_U_for_n = None
        best_max_for_n = float('inf')
        
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            U = _greedy_golomb(n, seed)
            if len(U) < n:
                continue
            # Local search to minimize max
            U_ls = _local_search_golomb(U, max_iters=200, seed=seed)
            if U_ls[-1] < best_max_for_n:
                best_max_for_n = U_ls[-1]
                best_U_for_n = U_ls
                
        if best_U_for_n:
            s = _score_set(best_U_for_n)
            if s > best_score:
                best_score = s
                best_U = best_U_for_n
                
    # Final polish on best_U
    if best_U:
        best_U = _local_search_golomb(best_U, max_iters=1000, seed=999)
        s = _score_set(best_U)
        # It's possible local search broke it? No, it maintains Golomb.
        # But score might decrease if max U increases? No, we minimize max U.
        # However, minimizing max U doesn't always maximize score if |U+U| changes significantly.
        # But for Golomb sets, |U+U| is relatively stable.
        # Let's just return the best found.
        
    return sorted(set(best_U))

if __name__ == "__main__":
    U = construct()
    print("size:", len(U), "max:", max(U) if U else 0, "score:", _score_set(U))
