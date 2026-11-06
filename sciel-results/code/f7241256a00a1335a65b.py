import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    # A is a subset of [0, 128]
    # Target: every t in [128,256] must be expressible as a+b with a<=b, a,b in A
    # Minimize the largest representation multiplicity (unordered pairs, a<=b)
    
    # Precompute the target range
    TARGET_START = 128
    TARGET_END = 256
    TARGETS = list(range(TARGET_START, TARGET_END + 1))
    
    # Precompute target indices for fast evaluation
    TARGET_INDICES = np.array(range(TARGET_START, TARGET_END + 1))
    
    def fast_evaluate(A_set, v_cache=None):
        """Return (max_multiplicity, covered) for the given set A using cached vector if provided."""
        if v_cache is None:
            v = np.zeros(129, dtype=np.int32)
            for x in A_set:
                v[x] = 1
        else:
            v = v_cache
        
        # Convolution: conv[t] = number of ordered pairs (a,b) with a+b=t
        conv = np.convolve(v, v, mode='full')
        
        # Get ordered counts for all targets at once
        ordered_counts = conv[TARGET_INDICES]
        
        # Compute diagonal corrections
        diag_mask = np.zeros(len(TARGET_INDICES), dtype=np.int32)
        for i, t in enumerate(TARGET_INDICES):
            if t % 2 == 0:
                mid = t // 2
                if mid in A_set:
                    diag_mask[i] = 1
        
        unordered_counts = (ordered_counts + diag_mask) // 2
        
        max_mult = int(np.max(unordered_counts)) if len(unordered_counts) > 0 else 0
        covered = np.all(unordered_counts > 0)
        
        return max_mult, covered
    
    # Strategy: Multi-start local search with diverse initial solutions
    # 1. Try several structured initializations
    # 2. Local search (simulated annealing) from each
    # 3. Keep the best valid solution found
    
    # Candidate initial solutions
    def make_initials():
        initials = []
        # Full set
        initials.append(set(range(0, 129)))
        # Half: [64, 128]
        initials.append(set(range(64, 129)))
        # Every other in [0,128]: even numbers
        initials.append(set(range(0, 129, 2)))
        # Random sets of various sizes
        random.seed(123)
        for size in [40, 50, 60, 70, 80, 90, 100]:
            s = set(random.sample(range(0, 129), size))
            # Ensure coverage potential: add elements to cover gaps
            # Simple heuristic: if t not covered, add t//2 if in range
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
        # New: structured patterns with better coverage properties
        # Symmetric around 64
        s = set()
        for i in range(0, 65):
            if random.random() < 0.5:
                s.add(i)
                s.add(128 - i)
        initials.append(s)
        # Dense in middle, sparse at edges
        s = set(range(32, 97))
        for i in range(0, 32):
            if random.random() < 0.3:
                s.add(i)
        for i in range(97, 129):
            if random.random() < 0.3:
                s.add(i)
        initials.append(s)
        return initials
    
    def local_search(A_set, time_budget, seed=None):
        """Simulated annealing local search with improved move generation and cooling."""
        if seed is not None:
            random.seed(seed)
        
        current_A = set(A_set)
        current_score, current_covered = fast_evaluate(current_A)
        
        if not current_covered:
            # Try to fix coverage greedily
            for t in TARGETS:
                # Check if t is covered
                found = False
                for a in current_A:
                    b = t - a
                    if b >= a and b in current_A:
                        found = True
                        break
                if not found:
                    # Add a pair that covers t
                    mid = t // 2
                    if 0 <= mid <= 128:
                        current_A.add(mid)
                        if t % 2 == 0:
                            pass  # (mid, mid) covers t
                        else:
                            if mid + 1 <= 128:
                                current_A.add(mid + 1)
            current_score, current_covered = fast_evaluate(current_A)
        
        best_A = set(current_A)
        best_score = current_score if current_covered else float('inf')
        
        # Improved cooling schedule
        T = 50.0
        T_min = 0.1
        cooling = 0.999
        
        # Maintain a list of elements for faster random selection
        current_list = sorted(current_A)
        
        iter_count = 0
        while time.time() < time_budget and T > T_min:
            iter_count += 1
            
            # Generate a move: add, remove, or swap
            move_type = random.random()
            
            if len(current_A) > 10 and move_type < 0.4:
                # Remove
                idx = random.randint(0, len(current_list) - 1)
                x = current_list[idx]
                new_A = current_A - {x}
                new_list = current_list[:idx] + current_list[idx+1:]
            elif move_type < 0.7:
                # Add
                x = random.randint(0, 128)
                if x in current_A:
                    continue
                new_A = current_A | {x}
                # Insert in sorted position
                new_list = current_list[:]
                pos = 0
                while pos < len(new_list) and new_list[pos] < x:
                    pos += 1
                new_list.insert(pos, x)
            else:
                # Swap: remove one, add another
                if len(current_A) > 5:
                    idx = random.randint(0, len(current_list) - 1)
                    x_remove = current_list[idx]
                    x_add = random.randint(0, 128)
                    if x_add in current_A and x_add != x_remove:
                        continue
                    new_A = (current_A - {x_remove}) | {x_add}
                    new_list = current_list[:idx] + current_list[idx+1:]
                    pos = 0
                    while pos < len(new_list) and new_list[pos] < x_add:
                        pos += 1
                    new_list.insert(pos, x_add)
                else:
                    # Just add if too small
                    x = random.randint(0, 128)
                    if x in current_A:
                        continue
                    new_A = current_A | {x}
                    new_list = current_list[:]
                    pos = 0
                    while pos < len(new_list) and new_list[pos] < x:
                        pos += 1
                    new_list.insert(pos, x)
            
            new_score, new_covered = fast_evaluate(new_A)
            
            # Score function: penalize lack of coverage heavily
            if new_covered:
                new_cost = new_score
            else:
                new_cost = new_score + 10000
            
            if current_covered:
                current_cost = current_score
            else:
                current_cost = current_score + 10000
            
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                current_A = new_A
                current_list = new_list
                current_score = new_score
                current_covered = new_covered
                
                if current_covered and current_score < best_score:
                    best_A = set(current_A)
                    best_score = current_score
            
            T *= cooling
        
        return best_A, best_score
    
    # Multi-start: try each initial solution with local search
    best_A = None
    best_score = float('inf')
    
    initials = make_initials()
    
    # Shuffle for diversity
    random.seed(42)
    random.shuffle(initials)
    
    # Allocate time budget across starts
    num_starts = len(initials)
    # Use first ~10s for quick evaluations to pick good starting points
    quick_time = start_time + 10
    # Remaining time for local search
    ls_budget = (deadline - start_time - 5) / num_starts
    
    # Phase 1: Quick evaluation of all initials
    scored_initials = []
    for init in initials:
        if time.time() > quick_time:
            break
        s, c = fast_evaluate(init)
        cost = s if c else s + 10000
        scored_initials.append((cost, init))
    
    # Sort by cost (best first)
    scored_initials.sort(key=lambda x: x[0])
    
    # Phase 2: Local search from best initials
    # Prioritize lower-cost starts
    for i, (cost, init) in enumerate(scored_initials):
        if time.time() >= deadline - 5:
            break
        # Allocate time based on rank: better starts get more time
        remaining_starts = len(scored_initials) - i
        remaining_time = deadline - 5 - time.time()
        if remaining_time <= 0:
            break
        # Give more time to better starts
        time_alloc = max(3, min(remaining_time / remaining_starts, 25))
        
        # Run local search with a different seed for diversity
        seed = 1000 + i
        ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=seed)
        
        if ls_score < best_score:
            best_score = ls_score
            best_A = ls_result
    
    # Phase 3: Additional random restarts if time remains
    if time.time() < deadline - 15 and best_A is not None:
        # Try a few more random restarts
        num_extra = 5
        for j in range(num_extra):
            if time.time() >= deadline - 5:
                break
            random.seed(2000 + j)
            size = random.randint(40, 90)
            init = set(random.sample(range(0, 129), size))
            # Fix coverage
            for t in TARGETS:
                found = False
                for a in sorted(init):
                    b = t - a
                    if b >= a and b in init:
                        found = True
                        break
                if not found:
                    mid = t // 2
                    if 0 <= mid <= 128:
                        init.add(mid)
                    if t % 2 == 1 and mid + 1 <= 128:
                        init.add(mid + 1)
            
            time_alloc = max(3, (deadline - 5 - time.time()) / (num_extra - j))
            ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=3000 + j)
            
            if ls_score < best_score:
                best_score = ls_score
                best_A = ls_result
    
    # Phase 4: Final polish - try to reduce peak multiplicity further
    if best_A is not None and best_score > 1 and time.time() < deadline - 10:
        # Try targeted removals to reduce peak
        polish_A = set(best_A)
        polish_score, polish_covered = fast_evaluate(polish_A)
        
        # Find elements that contribute most to high-multiplicity targets
        for _ in range(50):  # Try up to 50 removal attempts
            if time.time() >= deadline - 5:
                break
            if not polish_covered or len(polish_A) < 10:
                break
            
            # Find the target with highest multiplicity
            v = np.zeros(129, dtype=np.int32)
            for x in polish_A:
                v[x] = 1
            conv = np.convolve(v, v, mode='full')
            ordered_counts = conv[TARGET_INDICES]
            diag_mask = np.zeros(len(TARGET_INDICES), dtype=np.int32)
            for i, t in enumerate(TARGET_INDICES):
                if t % 2 == 0:
                    mid = t // 2
                    if mid in polish_A:
                        diag_mask[i] = 1
            unordered_counts = (ordered_counts + diag_mask) // 2
            
            max_idx = int(np.argmax(unordered_counts))
            max_mult = int(unordered_counts[max_idx])
            
            if max_mult <= 1:
                break
            
            target_t = TARGET_INDICES[max_idx]
            
            # Try removing elements that contribute to this target
            candidates = []
            for a in polish_A:
                b = target_t - a
                if a <= b and b in polish_A:
                    candidates.append(a)
            
            if candidates:
                # Try removing the most frequent contributor
                random.shuffle(candidates)
                for cand in candidates[:3]:
                    test_A = polish_A - {cand}
                    test_score, test_covered = fast_evaluate(test_A)
                    if test_covered and test_score < polish_score:
                        polish_A = test_A
                        polish_score = test_score
                        break
        
        if polish_covered and polish_score < best_score:
            best_score = polish_score
            best_A = polish_A
    
    # Fallback: ensure we have a valid solution
    if best_A is None or best_score == float('inf'):
        # Use a known valid solution: full set
        best_A = set(range(0, 129))
        best_score, best_covered = fast_evaluate(best_A)
        if not best_covered:
            # This shouldn't happen
            best_A = set(range(64, 129))
    
    # Final verification
    final_score, final_covered = fast_evaluate(best_A)
    if not final_covered:
        # Fallback to guaranteed valid
        best_A = set(range(0, 129))
        final_score, final_covered = fast_evaluate(best_A)
    
    return sorted(best_A)
