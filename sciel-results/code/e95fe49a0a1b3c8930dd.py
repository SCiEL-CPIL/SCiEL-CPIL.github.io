import time
import random
import math
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85  # safety margin for 90s kill
    
    n = 128
    lo, hi = 128, 256
    targets = list(range(lo, hi + 1))
    T = np.array(targets)
    
    # Precompute all unordered pairs (a, b) with a <= b, a+b in [lo, hi]
    pair_list = []
    for a in range(n + 1):
        for b in range(a, n + 1):
            s = a + b
            if lo <= s <= hi:
                pair_list.append((a, b, s))
    
    pair_idx = {}
    for i, (a, b, s) in enumerate(pair_list):
        pair_idx[(a, b)] = i
        if a != b:
            pair_idx[(b, a)] = i
    
    num_pairs = len(pair_list)
    pair_sums = np.array([p[2] for p in pair_list], dtype=np.int32)
    pair_a = np.array([p[0] for p in pair_list], dtype=np.int32)
    pair_b = np.array([p[1] for p in pair_list], dtype=np.int32)
    
    # For each element x, which pairs involve x?
    pairs_for_elem = {x: [] for x in range(n + 1)}
    for i, (a, b, s) in enumerate(pair_list):
        pairs_for_elem[a].append(i)
        if a != b:
            pairs_for_elem[b].append(i)
    
    # Mapping from sum to pairs
    pairs_for_sum = {s: [] for s in range(lo, hi + 1)}
    for i, (a, b, s) in enumerate(pair_list):
        pairs_for_sum[s].append(i)
    
    # ---------- Phase 1: Greedy construction ----------
    # Start empty, greedily add elements to cover targets
    A = set()
    coverage = np.zeros(hi + 1, dtype=np.int32)  # coverage[s] = number of pairs in A summing to s
    
    def add_element(x):
        A.add(x)
        for y in sorted(A):
            if y > x:
                s = x + y
                if lo <= s <= hi:
                    coverage[s] += 1
        s2 = x + x
        if lo <= s2 <= hi:
            coverage[s2] += 1
    
    def remove_element(x):
        for y in sorted(A):
            if y > x:
                s = x + y
                if lo <= s <= hi:
                    coverage[s] -= 1
        s2 = x + x
        if lo <= s2 <= hi:
            coverage[s2] -= 1
        A.discard(x)
    
    # Greedy: repeatedly add the element that covers the most uncovered targets
    remaining = set(targets)
    while remaining and time.time() < deadline - 5:
        best_x = None
        best_gain = 0
        for x in range(n + 1):
            if x in A:
                continue
            gain = 0
            for y in sorted(A):
                if y > x:
                    s = x + y
                    if s in remaining:
                        gain += 1
            s2 = x + x
            if s2 in remaining:
                gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None or best_gain == 0:
            break
        add_element(best_x)
        # Remove newly covered from remaining
        for y in sorted(A):
            if y > best_x:
                s = best_x + y
                remaining.discard(s)
        s2 = best_x + best_x
        remaining.discard(s2)
    
    # If still not covered, add random elements
    while remaining and time.time() < deadline - 4:
        x = random.randint(0, n)
        if x not in A:
            add_element(x)
            for y in sorted(A):
                if y > x:
                    remaining.discard(x + y)
            remaining.discard(x + x)
    
    # ---------- Phase 2: Local search with peak-first objective ----------
    # Compute current mults from A
    def compute_mults(A_set):
        mults = np.zeros(hi + 1, dtype=np.int32)
        A_sorted = sorted(A_set)
        for i in range(len(A_sorted)):
            for j in range(i, len(A_sorted)):
                s = A_sorted[i] + A_sorted[j]
                if lo <= s <= hi:
                    mults[s] += 1
        return mults
    
    mults = compute_mults(A)
    best_A = set(A)
    best_peak = int(np.max(mults[lo:hi+1]))
    best_lse = None
    
    def smooth_max(mults, beta=1.5):
        v = mults[lo:hi+1].astype(np.float64)
        mx = v.max()
        if mx == 0:
            return 0.0
        return mx + np.log(np.sum(np.exp(v - mx)) / len(v)) / beta
    
    best_lse = smooth_max(mults)
    
    # Local search: try swaps, adds, removes
    T = 3.0
    cooling = 0.9998
    A_list = list(A)
    
    def try_move(new_A, new_mults):
        nonlocal A, A_list, mults, best_A, best_peak, best_lse, T
        if not np.all(new_mults[lo:hi+1] > 0):
            return False
        new_peak = int(np.max(new_mults[lo:hi+1]))
        new_lse = smooth_max(new_mults)
        
        # Acceptance: peak first, then LSE
        if new_peak < best_peak or (new_peak == best_peak and new_lse < best_lse):
            A = set(new_A)
            A_list = list(A)
            mults = new_mults
            best_A = set(A)
            best_peak = new_peak
            best_lse = new_lse
            return True
        elif new_peak <= best_peak:
            delta = new_lse - best_lse
            if delta <= 0 or random.random() < math.exp(-delta / T):
                A = set(new_A)
                A_list = list(A)
                mults = new_mults
                if new_peak < best_peak or (new_peak == best_peak and new_lse < best_lse):
                    best_A = set(A)
                    best_peak = new_peak
                    best_lse = new_lse
                return True
        return False
    
    iter_count = 0
    while time.time() < deadline - 2 and T > 0.01:
        iter_count += 1
        T *= cooling
        move_type = random.random()
        
        if move_type < 0.35:
            # Swap: remove one, add one
            if len(A) < 30:
                continue
            x = random.choice(A_list)
            candidates = [y for y in range(n+1) if y not in A]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            new_A = set(A)
            new_A.discard(x)
            new_A.add(y)
            new_mults = compute_mults(new_A)
            if try_move(new_A, new_mults):
                pass
            # recompute A_list
            A_list = list(A)
        
        elif move_type < 0.60:
            # Remove
            if len(A) < 30:
                continue
            x = random.choice(A_list)
            new_A = set(A)
            new_A.discard(x)
            new_mults = compute_mults(new_A)
            if try_move(new_A, new_mults):
                pass
            A_list = list(A)
        
        else:
            # Add
            candidates = [y for y in range(n+1) if y not in A]
            if not candidates:
                continue
            y = random.choice(candidates)
            new_A = set(A)
            new_A.add(y)
            new_mults = compute_mults(new_A)
            if try_move(new_A, new_mults):
                pass
            A_list = list(A)
        
        # Periodic restart if stuck
        if iter_count % 500 == 0 and best_peak > 4:
            # Try a fresh greedy with different random seed
            pass
    
    # Final: try to reduce A size while maintaining coverage
    for x in sorted(best_A):
        test_A = best_A - {x}
        if not test_A:
            continue
        test_mults = compute_mults(test_A)
        if np.all(test_mults[lo:hi+1] > 0):
            best_A = test_A
    
    return list(best_A)
