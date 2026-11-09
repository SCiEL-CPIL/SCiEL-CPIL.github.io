import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
        
    def is_sidon(S):
        if not S:
            return True
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2

    def covers_all(S):
        diffs = get_diffs(S)
        return target_diffs.issubset(diffs)

    best_max = float('inf')
    best_set = None
    
    # Beam Search Parameters
    beam_width = 5000
    max_candidates_per_state = 20

    # State: (max_val, num_missing, S, used_diffs, covered_mask)
    S_init = (0,)
    used_diffs_init = set()
    covered_mask_init = 0
    max_val_init = 0
    
    beam = [(max_val_init, 30, S_init, used_diffs_init, covered_mask_init)]
    
    rng = random.Random(42)

    def get_candidates(S, used_diffs, covered_mask):
        candidates = set()
        missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        
        # Shuffle missing diffs to add randomness and avoid bias
        rng.shuffle(missing_diffs)
        
        # Generate candidates for all missing diffs (they're needed)
        for d in missing_diffs:
            for s in S:
                x = s + d
                if x <= max(S) if S else x <= 0:
                    continue
                if x > 10**6:
                    continue
                candidates.add(x)
        
        # Add some random candidates to escape local minima
        if S:
            last = max(S)
            for _ in range(10):
                rand_x = last + rng.randint(1, 30)
                if rand_x <= 10**6:
                    candidates.add(rand_x)
        
        valid_candidates = []
        S_list = list(S)
        last = S_list[-1] if S_list else -1
        
        for x in candidates:
            if x <= last:
                continue
            if x > 10**6:
                continue
            
            new_diffs = []
            valid = True
            for s in S_list:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            new_mask = covered_mask
            new_coverage = 0
            for d in new_diffs:
                if 1 <= d <= 30:
                    if not (covered_mask & (1 << (d - 1))):
                        new_coverage += 1
                    new_mask |= (1 << (d - 1))
            
            new_missing = 30 - bin(new_mask).count('1')
            # Score: prefer smaller x (to keep max low), then more coverage
            valid_candidates.append((x, new_mask, new_diffs, new_missing, new_coverage))
            
        # Sort and limit
        # Primary key: x (smaller is better for minimizing max)
        # Secondary key: new_coverage (higher is better)
        valid_candidates.sort(key=lambda item: (item[0], -item[4]))
        return valid_candidates[:max_candidates_per_state]

    for step in range(32):
        if time.time() > deadline:
            break
            
        new_beam = []
        for max_val, num_missing, S, used_diffs, covered_mask in beam:
            if time.time() > deadline:
                break
                
            if covered_mask == full_mask:
                if max_val < best_max:
                    best_max = max_val
                    best_set = list(S)
                continue
            
            candidates = get_candidates(S, used_diffs, covered_mask)
            for x, new_mask, new_diffs, new_missing, _ in candidates:
                new_S = S + (x,)
                new_used_diffs = used_diffs.union(new_diffs)
                new_max = x
                
                new_beam.append((new_max, new_missing, new_S, new_used_diffs, new_mask))
                
                if time.time() > deadline:
                    break
                    
        if not new_beam:
            break
            
        # Sort by max_val primarily, then num_missing
        new_beam.sort(key=lambda item: (item[0], item[1]))
        
        seen = set()
        unique_beam = []
        for state in new_beam:
            s_tuple = state[2]
            if s_tuple not in seen:
                seen.add(s_tuple)
                unique_beam.append(state)
            if len(unique_beam) >= beam_width:
                break
                
        beam = unique_beam
        
        if best_max < float('inf'):
            beam = [s for s in beam if s[0] < best_max]
            if not beam:
                break

    if best_set:
        # Local Search Polish
        S = sorted(best_set[:])
        
        def try_reduce_element(S, i):
            """Try to reduce element at index i to any valid smaller value.
            Returns (new_S, new_val) or (None, None)."""
            s_i = S[i]
            S_rest = S[:i] + S[i+1:]
            lower_bound = S[i-1] + 1 if i > 0 else 1
            
            # Try all values from s_i-1 down to lower_bound
            for new_val in range(s_i - 1, lower_bound - 1, -1):
                S_new = S_rest + [new_val]
                S_new.sort()
                if is_sidon(S_new) and covers_all(S_new):
                    return S_new, new_val
            return None, None
        
        # Phase 1: Exhaustive single-element reduction, iterating until no improvement
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            S.sort()
            
            # Try each element from largest to smallest
            for i in range(len(S) - 1, 0, -1):
                if (time.time() - start_time) > deadline:
                    break
                    
                new_S, new_val = try_reduce_element(S, i)
                if new_S is not None:
                    S = new_S
                    improved = True
                    break  # Restart scan with updated S
        
        # Phase 2: DFS Polish with tight bounds
        # Try to find a solution with max < current max
        current_max = max(S)
        # Lower bound estimate: k=9, D >= 72. 
        # We can start from max(72, current_max - 50) to save time
        start_D = max(72, current_max - 100)
        
        for D in range(start_D, current_max):
            if time.time() > deadline:
                break
                
            def dfs_polish(current_set, used_diffs, missing_diffs, last_elem, max_elem):
                if not missing_diffs:
                    return current_set[:]
                
                if time.time() > deadline:
                    return None
                    
                # Pruning:
                # Max elements we can add: 32 - len(current_set)
                # Max new diffs: sum_{k=1}^{rem} (len(current_set) + k - 1)
                rem = 32 - len(current_set)
                if rem <= 0:
                    return None
                max_new_diffs = 0
                for k in range(1, rem + 1):
                    max_new_diffs += len(current_set) + k - 1
                if max_new_diffs < len(missing_diffs):
                    return None

                # Candidates: last_elem + 1 to max_elem
                # Prioritize candidates that cover missing diffs
                candidates = range(last_elem + 1, max_elem + 1)
                
                # Heuristic: sort by number of missing diffs covered
                scored_candidates = []
                for x in candidates:
                    new_diffs = []
                    valid = True
                    for y in current_set:
                        d = abs(x - y)
                        if d in used_diffs:
                            valid = False
                            break
                        if d <= 30:
                            new_diffs.append(d)
                    if not valid:
                        continue
                    cov = len([d for d in new_diffs if d in missing_diffs])
                    scored_candidates.append((cov, x))
                
                # Sort by coverage (desc), then x (asc)
                scored_candidates.sort(key=lambda item: (-item[0], item[1]))
                
                for cov, x in scored_candidates:
                    new_diffs = [abs(x - y) for y in current_set]
                    new_missing = missing_diffs - set(new_diffs)
                    new_used = used_diffs | set(new_diffs)
                    
                    res = dfs_polish(current_set + [x], new_used, new_missing, x, max_elem)
                    if res is not None:
                        return res
                return None

            # Initialize DFS
            initial_missing = target_diffs.copy()
            # We can pre-fill some elements to speed up? No, start from 0.
            res = dfs_polish([0], set(), initial_missing, 0, D)
            if res is not None:
                # Verify
                if is_sidon(res) and covers_all(res) and max(res) <= D:
                    S = res
                    current_max = D
                    # Restart loop with new lower bound
                    start_D = max(72, current_max - 100)
                    # Update the loop range effectively by breaking and letting the outer loop handle it?
                    # The outer loop iterates D upwards. If we find a solution for D, we want to try D-1 next?
                    # No, the loop is `for D in range(start_D, current_max)`.
                    # If we find a solution for D, we update current_max to D.
                    # The range is already fixed. We should break and restart the DFS search with a new range.
                    break

        return S
    else:
        # Fallback
        S = [0]
        covered = set()
        while not covered.issuperset(target_diffs) and len(S) <= 32:
            missing = target_diffs - covered
            found = False
            for d in missing:
                for s in S:
                    c = s + d
                    if c in S or c < 0:
                        continue
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        found = True
                        break
                if found:
                    break
            if not found:
                c = 2 * max(S) + 3
                while True:
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        break
                    c += 1
        return S
