import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 128

    # Pre-compute 4-term APs
    fours = []
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            fours.append((i, i + d, i + 2 * d, i + 3 * d))
    fours = np.array(fours, dtype=np.int32)
    n_fours = len(fours)
    
    # For each position, precompute which APs it participates in
    pos_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(fours):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    # Build AP index arrays for each position for fast lookup
    pos_ap_indices = np.zeros((N, 20), dtype=np.int32)  # max ~20 APs per position
    pos_ap_counts = np.zeros(N, dtype=np.int32)
    for pos in range(N):
        pos_ap_indices[pos, :len(pos_aps[pos])] = np.array(pos_aps[pos], dtype=np.int32)
        pos_ap_counts[pos] = len(pos_aps[pos])

    # Vectorized violation check: returns number of 4-APs with < 3 distinct colors
    def count_violations(coloring):
        a = coloring[fours[:, 0]]
        b = coloring[fours[:, 1]]
        c = coloring[fours[:, 2]]
        d = coloring[fours[:, 3]]
        # A 4-AP has < 3 distinct colors iff it has exactly 1 or 2 distinct colors
        # 1 distinct: a==b==c==d
        # 2 distinct: exactly two values appear
        # Use: number of distinct < 3
        # Distinct count = 4 - (number of equal pairs that are "redundant")
        # Simpler: check if all in {x,y} for some x,y
        # A 4-tuple has < 3 distinct iff:
        #   (a==b & a==c) | (a==b & a==d) | (a==c & a==d) | (b==c & b==d) | (c==d & a==c) | (b==c & a==b) | ...
        # Actually, < 3 distinct means the set size is 1 or 2.
        # Size 1: a==b==c==d
        # Size 2: not size 1, but all values are in {x, y}
        # Equivalent: NOT (at least 3 distinct)
        # At least 3 distinct iff there exist 3 positions with different colors
        abc = (a != b) & (a != c) & (b != c)
        abd = (a != b) & (a != d) & (b != d)
        acd = (a != c) & (a != d) & (c != d)
        bcd = (b != c) & (b != d) & (c != d)
        has_three = abc | abd | acd | bcd
        return np.sum(~has_three)

    # Objective: minimize violations * 1000 + number of distinct colors
    def objective(coloring):
        v = count_violations(coloring)
        if v > 0:
            return v * 1000 + len(set(coloring.tolist()))
        return len(set(coloring.tolist()))

    # Random initialization with k colors
    def random_init(k):
        coloring = np.random.randint(0, k, size=N, dtype=np.int32)
        return coloring

    # Try to find a valid coloring with k colors via local search
    def local_search_init(k, max_iters=50000):
        coloring = random_init(k)
        current_obj = objective(coloring)
        best_coloring = coloring.copy()
        best_obj = current_obj
        temp = 10.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            if current_obj == 0:
                break
            
            # Pick a random position
            pos = np.random.randint(N)
            old_color = coloring[pos]
            new_color = np.random.randint(0, k)
            if new_color == old_color:
                continue
            
            coloring[pos] = new_color
            new_obj = objective(coloring)
            
            delta = new_obj - current_obj
            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 0.01)):
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_coloring = coloring.copy()
            else:
                coloring[pos] = old_color
            
            temp *= cooling
        
        return best_coloring, best_obj

    # Color merging: try to merge color 'src' into color 'dst'
    def try_merge(coloring, src, dst):
        positions = np.where(coloring == src)[0]
        if len(positions) == 0:
            return None
        
        new_coloring = coloring.copy()
        for pos in positions:
            new_coloring[pos] = dst
            # Check if this creates a violation
            v = count_violations(new_coloring)
            if v > 0:
                new_coloring[pos] = src
                return None
        return new_coloring

    # Main algorithm
    best_result = None
    best_obj = float('inf')
    
    # Phase 1: Find valid colorings with increasing k
    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > DEADLINE - 15:
            break
        
        # Multiple restarts
        best_k_obj = float('inf')
        best_k_coloring = None
        
        n_restarts = 20 if k <= 4 else 10
        for r in range(n_restarts):
            if time.time() > DEADLINE - 15:
                break
            
            coloring, obj = local_search_init(k, max_iters=30000)
            
            if obj < best_k_obj:
                best_k_obj = obj
                best_k_coloring = coloring
            
            # If we found a valid coloring, try to merge colors
            if obj == 0:
                # Try to merge colors to reduce count
                unique = sorted(set(coloring.tolist()))
                changed = True
                while changed and time.time() < DEADLINE - 10:
                    changed = False
                    unique = sorted(set(coloring.tolist()))
                    if len(unique) <= k - 1:
                        # Try to merge the least frequent color
                        counts = {c: np.sum(coloring == c) for c in unique}
                        src = min(unique, key=lambda c: counts[c])
                        for dst in unique:
                            if dst == src:
                                continue
                            merged = try_merge(coloring, src, dst)
                            if merged is not None:
                                coloring = merged
                                changed = True
                                break
                    else:
                        # We have more colors than k, which shouldn't happen
                        break
                
                obj = objective(coloring)
                if obj < best_k_obj:
                    best_k_obj = obj
                    best_k_coloring = coloring
        
        if best_k_obj < best_obj:
            best_obj = best_k_obj
            best_result = best_k_coloring
        
        # If we found a valid coloring with k colors, stop
        if best_obj == 0:
            break
    
    # Phase 2: If we have a valid coloring, try to further reduce colors
    if best_result is not None and best_obj == 0:
        coloring = best_result
        changed = True
        while changed and time.time() < DEADLINE - 5:
            changed = False
            unique = sorted(set(coloring.tolist()))
            if len(unique) <= 2:
                break
            
            # Try merging each pair
            for src in unique:
                if time.time() > DEADLINE - 5:
                    break
                counts = {c: np.sum(coloring == c) for c in unique}
                if counts[src] != min(counts.values()):
                    continue  # Only try least frequent
                
                for dst in unique:
                    if dst == src:
                        continue
                    merged = try_merge(coloring, src, dst)
                    if merged is not None:
                        coloring = merged
                        changed = True
                        break
                if changed:
                    break
        
        obj = objective(coloring)
        if obj < best_obj:
            best_obj = obj
            best_result = coloring
    
    # Phase 3: If no valid coloring found, do a more aggressive search
    if best_result is None or best_obj > 0:
        # Try with more colors and longer search
        for k in [8, 10, 12]:
            if time.time() > DEADLINE - 5:
                break
            
            coloring, obj = local_search_init(k, max_iters=80000)
            if obj < best_obj:
                best_obj = obj
                best_result = coloring
            
            if obj == 0:
                # Try to reduce
                unique = sorted(set(coloring.tolist()))
                changed = True
                while changed and time.time() < DEADLINE - 3:
                    changed = False
                    unique = sorted(set(coloring.tolist()))
                    if len(unique) <= 3:
                        break
                    counts = {c: np.sum(coloring == c) for c in unique}
                    src = min(unique, key=lambda c: counts[c])
                    for dst in unique:
                        if dst == src:
                            continue
                        merged = try_merge(coloring, src, dst)
                        if merged is not None:
                            coloring = merged
                            changed = True
                            break
                
                obj = objective(coloring)
                if obj < best_obj:
                    best_obj = obj
                    best_result = coloring
                break
    
    # Ensure we have a result
    if best_result is None:
        # Fallback: use a simple valid coloring with many colors
        coloring = np.arange(N, dtype=np.int32)
        best_result = coloring
    
    # Remap colors to be in 0..127 range and minimize label spread
    unique = sorted(set(best_result.tolist()))
    color_map = {c: i for i, c in enumerate(unique)}
    final = np.array([color_map[c] for c in best_result], dtype=np.int32)
    
    return [int(x) for x in final]

if __name__ == "__main__":
    res = construct()
    print(f"Distinct colors: {len(set(res))}")
