import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    TARGET_START = 128
    TARGET_END = 256
    TARGETS = list(range(TARGET_START, TARGET_END + 1))
    NUM_TARGETS = TARGET_END - TARGET_START + 1  # 129 targets
    
    # Precompute: for each pair (a,b) with a<=b, the target index or -1
    pair_index = np.full((129, 129), -1, dtype=np.int32)
    for a in range(129):
        for b in range(a, 129):
            s = a + b
            if TARGET_START <= s <= TARGET_END:
                pair_index[a][b] = s - TARGET_START
    
    # Precompute: for each element x, which targets it pairs with each other element
    # pair_targets[x] = list of (other_element, target_idx) for all other in [0,128]
    # This helps with fast evaluation
    
    def build_counts(A_set):
        """Build unordered pair counts for targets [128,256]."""
        counts = np.zeros(NUM_TARGETS, dtype=np.int32)
        A_list = sorted(A_set)
        for i, a in enumerate(A_list):
            for b in A_list[i:]:
                idx = pair_index[a][b]
                if idx >= 0:
                    counts[idx] += 1
        return counts
    
    def evaluate(A_set):
        """Return (max_multiplicity, covered)."""
        counts = build_counts(A_set)
        max_mult = int(np.max(counts))
        covered = bool(np.all(counts > 0))
        return max_mult, covered
    
    def greedy_construct(K, seed=None):
        """
        Greedily construct a subset A of [0,128] such that:
        - Every target in [128,256] has at least one representation
        - No target has more than K representations
        Uses a priority-based greedy with backtracking on conflicts.
        Returns (A, max_mult, covered) or None if infeasible.
        """
        if seed is not None:
            random.seed(seed)
        
        # Strategy: start with a covering set, then prune to reduce max multiplicity
        # Phase 1: Find a minimal covering set using greedy set cover
        # Phase 2: Add elements to balance counts, staying under K
        
        # Initial: use all of [64, 128] as a base (guarantees coverage of [128,256])
        # Actually [64,128] gives sums [128,256] with each sum having multiple reps
        # Let's use a smarter initial: try to find a small covering set
        
        # Greedy set cover: repeatedly add the element that covers the most uncovered targets
        counts = np.zeros(NUM_TARGETS, dtype=np.int32)
        A = set()
        
        # Precompute: for each x, the list of target indices that would be affected
        # when x is added given current A
        # For speed, just compute on the fly
        
        # Phase 1: Build a covering set
        # Start with a reasonable seed
        # Try multiple random orderings
        best_covering = None
        best_covering_size = 999
        
        for attempt in range(5):
            A_try = set()
            counts_try = np.zeros(NUM_TARGETS, dtype=np.int32)
            
            # Random priority ordering
            elements = list(range(129))
            random.shuffle(elements)
            
            # First, ensure basic coverage by adding key elements
            # Elements that are "central" for coverage: around 64
            # Add elements in shuffled order, keeping only those that help coverage
            # and don't violate K constraint
            
            for x in elements:
                # Check if adding x violates K
                new_counts = counts_try.copy()
                for a in A_try:
                    if a != x:
                        idx = pair_index[min(a,x)][max(a,x)]
                        if idx >= 0:
                            new_counts[idx] += 1
                idx = pair_index[x][x]
                if idx >= 0:
                    new_counts[idx] += 1
                
                # Check if this helps coverage
                uncovered_before = np.sum(counts_try == 0)
                uncovered_after = np.sum(new_counts == 0)
                max_after = int(np.max(new_counts))
                
                # Add if it helps coverage or if we're not yet covered and it doesn't hurt too much
                if uncovered_after < uncovered_before:
                    if max_after <= K:
                        A_try.add(x)
                        counts_try = new_counts
                elif not np.all(counts_try > 0) and max_after <= K:
                    # Still need coverage, add if feasible
                    A_try.add(x)
                    counts_try = new_counts
            
            if np.all(counts_try > 0) and len(A_try) < best_covering_size:
                best_covering = A_try.copy()
                best_covering_size = len(A_try)
        
        if best_covering is None:
            # Fallback: use [64, 128]
            best_covering = set(range(64, 129))
        
        # Phase 2: Try to reduce max multiplicity by swapping elements
        A = best_covering
        counts = build_counts(A)
        current_max = int(np.max(counts))
        
        # Iterative improvement: try removing elements and re-adding different ones
        improved = True
        max_iters = 200
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if not improved:
                break
            improved = False
            
            current_max = int(np.max(counts))
            if current_max <= 1:
                break
            
            # Find the target with max count
            max_targets = np.where(counts == current_max)[0]
            
            # For each element in A, try removing it
            for x in list(A):
                if len(A) <= 10:
                    break
                
                # Remove x
                test_counts = counts.copy()
                for a in A:
                    if a != x:
                        idx = pair_index[min(a,x)][max(a,x)]
                        if idx >= 0:
                            test_counts[idx] -= 1
                idx = pair_index[x][x]
                if idx >= 0:
                    test_counts[idx] -= 1
                
                test_max = int(np.max(test_counts))
                
                if test_max < current_max and np.all(test_counts > 0):
                    # Direct improvement: just remove x
                    A = A - {x}
                    counts = test_counts
                    improved = True
                    break
                elif not np.all(test_counts > 0):
                    # Coverage broken. Try to fix with one addition
                    uncovered = np.where(test_counts == 0)[0]
                    if len(uncovered) > 3:
                        continue
                    
                    # Try adding various y to fix coverage
                    best_y = None
                    best_y_max = current_max + 1
                    best_y_counts = None
                    
                    # Try y values that might cover uncovered targets
                    candidates = []
                    for u in uncovered:
                        t = u + TARGET_START
                        for a in A:
                            if a != x:
                                b = t - a
                                if 0 <= b <= 128 and b != x:
                                    candidates.append(b)
                    # Also try midpoints
                    for u in uncovered:
                        t = u + TARGET_START
                        mid = t // 2
                        if 0 <= mid <= 128:
                            candidates.append(mid)
                        if t % 2 == 1 and mid + 1 <= 128:
                            candidates.append(mid + 1)
                    
                    # Sample up to 20 candidates
                    random.shuffle(candidates)
                    for y in candidates[:20]:
                        if y in A:
                            continue
                        
                        cand_counts = test_counts.copy()
                        for a in A:
                            if a != x:
                                idx = pair_index[min(a,y)][max(a,y)]
                                if idx >= 0:
                                    cand_counts[idx] += 1
                        idx = pair_index[y][y]
                        if idx >= 0:
                            cand_counts[idx] += 1
                        
                        cand_max = int(np.max(cand_counts))
                        if np.all(cand_counts > 0) and cand_max < best_y_max:
                            best_y_max = cand_max
                            best_y = y
                            best_y_counts = cand_counts
                    
                    if best_y is not None and best_y_max < current_max:
                        A = (A - {x}) | {best_y}
                        counts = best_y_counts
                        improved = True
                        break
        
        final_max = int(np.max(counts))
        covered = bool(np.all(counts > 0))
        return A, final_max, covered
    
    def branch_and_bound(K, time_limit):
        """
        Branch and bound to find if there exists a subset A with max multiplicity <= K
        that covers all targets. Uses greedy lower bounds.
        Returns (A, max_mult, covered) if found, else None.
        """
        end_time = time.time() + time_limit
        best_result = None
        
        # Lower bound: for each target t, the minimum number of pairs needed
        # is 1 (for coverage). The maximum over targets of the minimum pairs
        # gives a lower bound on max multiplicity.
        # Actually, we need at least 1 per target, so lower bound is 1.
        # Better: count how many pairs hit each target in the full set [0,128]
        
        # Full set counts
        full_counts = np.zeros(NUM_TARGETS, dtype=np.int32)
        for a in range(129):
            for b in range(a, 129):
                idx = pair_index[a][b]
                if idx >= 0:
                    full_counts[idx] += 1
        
        # The maximum in full_counts tells us the "natural" peak
        # We want to reduce this by removing elements
        
        # Greedy removal: repeatedly remove the element whose removal
        # reduces the max count the most, while maintaining coverage
        A = set(range(129))
        counts = full_counts.copy()
        
        while time.time() < end_time:
            current_max = int(np.max(counts))
            if current_max <= K:
                covered = bool(np.all(counts > 0))
                if covered:
                    return A, current_max, covered
                break
            
            # Find element whose removal most reduces max
            best_x = None
            best_new_max = current_max + 1
            best_new_counts = None
            
            # Sample elements to try (for speed)
            A_list = sorted(A)
            random.shuffle(A_list)
            for x in A_list[:50]:  # Try up to 50 elements
                test_counts = counts.copy()
                for a in A:
                    if a != x:
                        idx = pair_index[min(a,x)][max(a,x)]
                        if idx >= 0:
                            test_counts[idx] -= 1
                idx = pair_index[x][x]
                if idx >= 0:
                    test_counts[idx] -= 1
                
                test_max = int(np.max(test_counts))
                if test_max < best_new_max:
                    best_new_max = test_max
                    best_x = x
                    best_new_counts = test_counts
            
            if best_x is None or best_new_max >= current_max:
                break
            
            # Check coverage
            if not np.all(best_new_counts > 0):
                # Try to fix coverage
                uncovered = np.where(best_new_counts == 0)[0]
                fixed = False
                for y in range(129):
                    if y in A and y != best_x:
                        continue
                    if y == best_x:
                        continue
                    cand = best_new_counts.copy()
                    for a in A:
                        if a != best_x:
                            idx = pair_index[min(a,y)][max(a,y)]
                            if idx >= 0:
                                cand[idx] += 1
                    idx = pair_index[y][y]
                    if idx >= 0:
                        cand[idx] += 1
                    if np.all(cand > 0):
                        A = (A - {best_x}) | {y}
                        counts = cand
                        fixed = True
                        break
                if not fixed:
                    # Revert
                    break
            else:
                A = A - {best_x}
                counts = best_new_counts
        
        final_max = int(np.max(counts))
        covered = bool(np.all(counts > 0))
        if covered:
            return A, final_max, covered
        return None
    
    # Main search strategy:
    # 1. Try K=1, 2, 3, ... until we find a feasible solution
    # 2. Use greedy construction + local search for each K
    
    best_A = None
    best_score = float('inf')
    
    # Phase 1: Try increasing K values with greedy + local search
    for K in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds for greedy construction
        found = False
        for seed in [42, 123, 456, 789, 101]:
            if time.time() > deadline - 10:
                break
            result = greedy_construct(K, seed=seed)
            if result is not None:
                A, max_mult, covered = result
                if covered:
                    if max_mult < best_score:
                        best_score = max_mult
                        best_A = A
                    if max_mult <= K:
                        found = True
                        break
        
        if found:
            # Try to improve further with local search
            if best_A is not None and best_score > 1:
                # Local search to try to reduce below K
                A_ls = set(best_A)
                counts_ls = build_counts(A_ls)
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    current_max = int(np.max(counts_ls))
                    for x in list(A_ls):
                        if len(A_ls) <= 10:
                            break
                        test_counts = counts_ls.copy()
                        for a in A_ls:
                            if a != x:
                                idx = pair_index[min(a,x)][max(a,x)]
                                if idx >= 0:
                                    test_counts[idx] -= 1
                        idx = pair_index[x][x]
                        if idx >= 0:
                            test_counts[idx] -= 1
                        test_max = int(np.max(test_counts))
                        if test_max < current_max and np.all(test_counts > 0):
                            A_ls = A_ls - {x}
                            counts_ls = test_counts
                            improved = True
                            if int(np.max(counts_ls)) < best_score:
                                best_score = int(np.max(counts_ls))
                                best_A = A_ls
                            break
    
    # Phase 2: Branch and bound for small K
    if time.time() < deadline - 15:
        for K in [2, 3, 4]:
            if time.time() > deadline - 15:
                break
            result = branch_and_bound(K, time_limit=min(10, deadline - 15 - time.time()))
            if result is not None:
                A, max_mult, covered = result
                if covered and max_mult < best_score:
                    best_score = max_mult
                    best_A = A
    
    # Phase 3: Random restarts with local search
    if time.time() < deadline - 10 and best_A is not None:
        for j in range(8):
            if time.time() >= deadline - 5:
                break
            random.seed(5000 + j)
            # Start from a random subset and greedily fix coverage
            size = random.randint(30, 70)
            init = set(random.sample(range(0, 129), size))
            
            # Greedily add elements to fix coverage
            counts_init = build_counts(init)
            for _ in range(200):
                if np.all(counts_init > 0):
                    break
                uncovered = np.where(counts_init == 0)[0]
                if len(uncovered) == 0:
                    break
                # Find best element to add
                best_y = None
                best_gain = 0
                for y in range(129):
                    if y in init:
                        continue
                    # Count how many uncovered targets this would cover
                    gain = 0
                    for u in uncovered:
                        t = u + TARGET_START
                        # Check if y helps cover t
                        a = t - y
                        if 0 <= a <= 128 and a in init:
                            gain += 1
                        # Also check (y,y)
                        if 2*y == t and y not in init:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
                if best_y is None or best_gain == 0:
                    break
                init.add(best_y)
                counts_init = build_counts(init)
            
            if not np.all(counts_init > 0):
                continue
            
            init_max = int(np.max(counts_init))
            if init_max >= best_score:
                continue
            
            # Local search on this initial
            A_ls = set(init)
            counts_ls = counts_init.copy()
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                current_max = int(np.max(counts_ls))
                # Try removing elements
                for x in sorted(A_ls):
                    if len(A_ls) <= 10:
                        break
                    test_counts = counts_ls.copy()
                    for a in A_ls:
                        if a != x:
                            idx = pair_index[min(a,x)][max(a,x)]
                            if idx >= 0:
                                test_counts[idx] -= 1
                    idx = pair_index[x][x]
                    if idx >= 0:
                        test_counts[idx] -= 1
                    test_max = int(np.max(test_counts))
                    if test_max < current_max and np.all(test_counts > 0):
                        A_ls = A_ls - {x}
                        counts_ls = test_counts
                        improved = True
                        if int(np.max(counts_ls)) < best_score:
                            best_score = int(np.max(counts_ls))
                            best_A = A_ls
                        break
                # Try swaps
                if not improved:
                    for x in sorted(A_ls):
                        if len(A_ls) <= 10:
                            break
                        # Remove x
                        temp_counts = counts_ls.copy()
                        for a in A_ls:
                            if a != x:
                                idx = pair_index[min(a,x)][max(a,x)]
                                if idx >= 0:
                                    temp_counts[idx] -= 1
                        idx = pair_index[x][x]
                        if idx >= 0:
                            temp_counts[idx] -= 1
                        
                        if not np.all(temp_counts > 0):
                            # Try to fix with one addition
                            uncovered = np.where(temp_counts == 0)[0]
                            for y in range(129):
                                if y in A_ls or y == x:
                                    continue
                                cand = temp_counts.copy()
                                for a in A_ls:
                                    if a != x:
                                        idx = pair_index[min(a,y)][max(a,y)]
                                        if idx >= 0:
                                            cand[idx] += 1
                                idx = pair_index[y][y]
                                if idx >= 0:
                                    cand[idx] += 1
                                if np.all(cand > 0):
                                    cand_max = int(np.max(cand))
                                    if cand_max < current_max:
                                        A_ls = (A_ls - {x}) | {y}
                                        counts_ls = cand
                                        improved = True
                                        if cand_max < best_score:
                                            best_score = cand_max
                                            best_A = A_ls
                                        break
                            if improved:
                                break
    
    # Fallback
    if best_A is None or best_score == float('inf'):
        best_A = set(range(0, 129))
        final_score, final_covered = evaluate(best_A)
        if not final_covered:
            best_A = set(range(64, 129))
    
    # Final verification
    final_score, final_covered = evaluate(best_A)
    if not final_covered:
        best_A = set(range(0, 129))
        final_score, final_covered = evaluate(best_A)
    
    return sorted(best_A)
