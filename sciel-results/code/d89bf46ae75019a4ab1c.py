import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    
    n = 128
    # A is a subset of [0, 128]
    # Target: every t in [128,256] must be expressible as a+b with a<=b, a,b in A
    # Minimize the largest representation multiplicity (unordered pairs, a<=b)
    
    # Precompute the target range
    TARGET_START = 128
    TARGET_END = 256
    TARGETS = list(range(TARGET_START, TARGET_END + 1))
    NUM_TARGETS = TARGET_END - TARGET_START + 1  # 129 targets
    
    def build_unordered_counts(A_set):
        """Build the unordered pair count for each target t in [128,256] using convolution."""
        # Create indicator vector for A
        ind = np.zeros(n + 1, dtype=np.float64)
        for x in A_set:
            ind[x] = 1.0
        
        # Convolution gives ordered pairs count
        conv = np.convolve(ind, ind, mode='full')
        
        # For unordered pairs with a<=b:
        # count_unordered[t] = (count_ordered[t] + count_diagonal[t]) / 2
        # where count_diagonal[t] = 1 if t is even and t/2 in A, else 0
        counts = np.zeros(NUM_TARGETS, dtype=np.int32)
        for i, t in enumerate(range(TARGET_START, TARGET_END + 1)):
            ordered_count = int(conv[t]) if t < len(conv) else 0
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            counts[i] = (ordered_count + diag) // 2
        return counts
    
    def evaluate(A_set):
        """Return (max_multiplicity, covered) for the given set A."""
        counts = build_unordered_counts(A_set)
        max_mult = int(np.max(counts))
        covered = bool(np.all(counts > 0))
        return max_mult, covered
    
    def get_unordered_counts(A_set):
        """Return the unordered pair count array for targets [128,256]."""
        return build_unordered_counts(A_set)
    
    def compute_cost(counts, A_set):
        """Compute a smooth cost function for the SA search.
        Uses soft-max surrogate for the max multiplicity to provide gradient information.
        """
        max_mult = int(np.max(counts))
        uncovered = int(np.sum(counts == 0))
        
        # Soft-max surrogate: approximates max but is smooth
        # Using log-sum-exp with temperature
        temp = 2.0
        shifted = counts.astype(np.float64) / temp
        max_shifted = np.max(shifted)
        log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        soft_max = log_sum_exp * temp
        
        # Combine: prioritize coverage, then minimize soft-max (which approximates max)
        # Add small penalty for variance to encourage flattening
        var = float(np.var(counts))
        return soft_max + 100 * uncovered + 0.1 * var
    
    def local_search(A_set, time_budget, seed=None):
        """Local search with incremental evaluation and swap moves. Returns (best_A, best_score)."""
        if seed is not None:
            random.seed(seed)
        
        current_A = set(A_set)
        current_counts = get_unordered_counts(current_A)
        
        # Best tracking
        best_A = set(current_A)
        best_counts = current_counts.copy()
        best_max = int(np.max(best_counts))
        best_covered = bool(np.all(best_counts > 0))
        best_score = best_max if best_covered else float('inf')
        
        T = 50.0
        T_min = 1.0
        cooling = 0.9995
        
        iter_count = 0
        while time.time() < time_budget and T > T_min:
            iter_count += 1
            current_cost = compute_cost(current_counts, current_A)
            
            move_type = random.random()
            accepted = False
            new_A = None
            new_counts = None
            new_cost = None
            
            if move_type < 0.4:
                # Add move
                x = random.randint(0, 128)
                if x not in current_A:
                    new_A = current_A | {x}
                    new_counts = get_unordered_counts(new_A)
                    new_cost = compute_cost(new_counts, new_A)
            elif move_type < 0.7:
                # Remove move
                if len(current_A) > 10:
                    x = random.choice(list(current_A))
                    new_A = current_A - {x}
                    new_counts = get_unordered_counts(new_A)
                    new_cost = compute_cost(new_counts, new_A)
            else:
                # Swap move
                if len(current_A) > 10:
                    x = random.choice(list(current_A))
                    y = random.randint(0, 128)
                    if y not in current_A:
                        new_A = (current_A - {x}) | {y}
                        new_counts = get_unordered_counts(new_A)
                        new_cost = compute_cost(new_counts, new_A)
            
            if new_A is not None:
                delta = new_cost - current_cost
                if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                    current_A = new_A
                    current_counts = new_counts
                    accepted = True
                    
                    # Update best
                    new_max = int(np.max(new_counts))
                    new_covered = bool(np.all(new_counts > 0))
                    new_score = new_max if new_covered else float('inf')
                    if new_score < best_score:
                        best_score = new_score
                        best_A = set(new_A)
                        best_counts = new_counts.copy()
            
            T *= cooling
        
        # Final Polish: Systematic 1-out-1-in swap search to reduce max multiplicity
        if best_score < float('inf'):
            improved = True
            while improved and time.time() < time_budget:
                improved = False
                best_max = int(np.max(best_counts))
                best_A_list = sorted(best_A)
                
                for x in best_A_list[:]:
                    if len(best_A) <= 10:
                        break
                    
                    # Try removing x
                    test_A = best_A - {x}
                    test_counts = get_unordered_counts(test_A)
                    test_max = int(np.max(test_counts))
                    test_covered = bool(np.all(test_counts > 0))
                    
                    if test_covered and test_max < best_max:
                        # Direct improvement
                        best_A = test_A
                        best_counts = test_counts
                        best_max = test_max
                        best_score = best_max
                        improved = True
                        break
                    elif not test_covered:
                        # Coverage broken. Try to fix by adding one element y
                        # Find the best y that restores coverage and minimizes max
                        best_y = None
                        best_y_score = float('inf')
                        
                        # Try all possible y values (0 to 128)
                        for y in range(0, 129):
                            if y in best_A:
                                continue
                            cand_A = test_A | {y}
                            cand_counts = get_unordered_counts(cand_A)
                            cand_max = int(np.max(cand_counts))
                            cand_covered = bool(np.all(cand_counts > 0))
                            
                            if cand_covered and cand_max < best_y_score:
                                best_y_score = cand_max
                                best_y = y
                                best_y_counts = cand_counts
                        
                        if best_y is not None and best_y_score < best_max:
                            # Perform swap
                            best_A = (best_A - {x}) | {best_y}
                            best_counts = best_y_counts
                            best_max = best_y_score
                            best_score = best_max
                            improved = True
                            break
        
        return best_A, best_score

    # Multi-start: try each initial solution with local search
    best_A = None
    best_score = float('inf')
    
    def make_initials():
        initials = []
        # Full set
        initials.append(set(range(0, 129)))
        # Half: [64, 128]
        initials.append(set(range(64, 129)))
        # Every other in [0,128]: even numbers
        initials.append(set(range(0, 129, 2)))
        # Odd numbers
        initials.append(set(range(1, 129, 2)))
        # Random sets of various sizes
        random.seed(123)
        for size in [40, 50, 60, 70, 80, 90, 100]:
            s = set(random.sample(range(0, 129), size))
            initials.append(s)
        # Structured: multiples of k
        for k in [2, 3, 4]:
            s = set(range(0, 129, k))
            initials.append(s)
        # Complementary pairs
        s = set()
        for i in range(0, 129, 2):
            s.add(i)
            s.add(128 - i)
        initials.append(s)
        return initials

    def fix_coverage(A_set):
        """Greedily add elements to ensure all targets are covered."""
        A = set(A_set)
        for t in TARGETS:
            # Check if t is covered
            found = False
            for a in A:
                b = t - a
                if b >= a and b in A:
                    found = True
                    break
            if not found:
                mid = t // 2
                if 0 <= mid <= 128:
                    A.add(mid)
                if t % 2 == 1 and mid + 1 <= 128:
                    A.add(mid + 1)
        return A

    initials = make_initials()
    
    # Shuffle for diversity
    random.seed(42)
    random.shuffle(initials)
    
    # Phase 1: Quick evaluation of all initials
    scored_initials = []
    for init in initials:
        if time.time() > start_time + 10:
            break
        s, c = evaluate(init)
        cost = s if c else s + 10000
        scored_initials.append((cost, init))
    
    # Sort by cost (best first)
    scored_initials.sort(key=lambda x: x[0])
    
    # Phase 2: Local search from best initials
    for i, (cost, init) in enumerate(scored_initials):
        if time.time() >= deadline - 5:
            break
        remaining_starts = len(scored_initials) - i
        remaining_time = deadline - 5 - time.time()
        if remaining_time <= 0:
            break
        time_alloc = max(2, min(remaining_time / remaining_starts, 25))
        
        seed = 1000 + i
        ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=seed)
        
        if ls_score < best_score:
            best_score = ls_score
            best_A = ls_result
    
    # Phase 3: Additional random restarts if time remains
    if time.time() < deadline - 10 and best_A is not None:
        num_extra = 3
        for j in range(num_extra):
            if time.time() >= deadline - 5:
                break
            random.seed(2000 + j)
            size = random.randint(40, 90)
            init = set(random.sample(range(0, 129), size))
            init = fix_coverage(init)
            
            time_alloc = max(2, (deadline - 5 - time.time()) / (num_extra - j))
            ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=3000 + j)
            
            if ls_score < best_score:
                best_score = ls_score
                best_A = ls_result
    
    # Fallback: ensure we have a valid solution
    if best_A is None or best_score == float('inf'):
        best_A = set(range(0, 129))
        best_score, best_covered = evaluate(best_A)
        if not best_covered:
            best_A = set(range(64, 129))
    
    # Final verification
    final_score, final_covered = evaluate(best_A)
    if not final_covered:
        best_A = set(range(0, 129))
        final_score, final_covered = evaluate(best_A)
    
    return sorted(best_A)
