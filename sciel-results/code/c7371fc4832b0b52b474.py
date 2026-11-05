"""Football pool problem: ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random

_DEADLINE = 142.0  # seconds, safety margin before 150s kill

def _word_to_int(word, q, n):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx

def _int_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word

def _build_balls(q, n, radius):
    """Precompute for each word the list of words within Hamming distance radius."""
    total = q ** n
    balls = []
    for w in range(total):
        word = _int_to_word(w, q, n)
        neighbors = [w]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for v in range(q):
                    if v == orig:
                        continue
                    word[pos] = v
                    neighbors.append(_word_to_int(word, q, n))
                    word[pos] = orig
        balls.append(neighbors)
    return balls

def construct(q, n, radius):
    start_time = time.time()
    total = q ** n
    rng = random.Random(12345)
    
    # Precompute balls as numpy arrays for fast coverage updates
    balls = _build_balls(q, n, radius)
    balls_np = [np.array(b, dtype=np.int32) for b in balls]
    
    # Build initial code: tetracode direct sum (81 words)
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    best_code_set = set(_word_to_int(c, q, n) for c in initial_code)
    best_size = len(best_code_set)
    
    def compute_coverage(code_set):
        cov = np.zeros(total, dtype=np.int32)
        for ci in code_set:
            cov[balls_np[ci]] += 1
        return cov

    def greedy_remove(code_set, coverage, max_removals=200):
        """Remove redundant codewords greedily."""
        removed = 0
        for _ in range(max_removals):
            if time.time() - start_time > _DEADLINE - 3:
                break
            best_ci = -1
            best_slack = -1
            found = False
            candidates = list(code_set)
            rng.shuffle(candidates)
            for ci in candidates:
                ball = balls_np[ci]
                min_cov = np.min(coverage[ball])
                if min_cov >= 2:
                    found = True
                    slack = int(np.sum(coverage[ball] - 1))
                    if slack > best_slack:
                        best_slack = slack
                        best_ci = ci
            if not found:
                break
            coverage[balls_np[best_ci]] -= 1
            code_set.discard(best_ci)
            removed += 1
        return removed

    def local_search_swap(code_set, coverage, time_limit, max_iterations=5000):
        """Local search: iteratively swap codewords to reduce count while maintaining coverage.
        Uses a smooth surrogate: prioritize swaps that maximize the minimum coverage improvement."""
        current_size = len(code_set)
        best_local_set = set(code_set)
        best_local_size = current_size
        iterations = 0
        
        while time.time() - start_time < time_limit and iterations < max_iterations:
            iterations += 1
            if time.time() - start_time > time_limit:
                break
            
            # Find all removable codewords (min coverage in their ball >= 2)
            removable = []
            for ci in code_set:
                if np.min(coverage[balls_np[ci]]) >= 2:
                    removable.append(ci)
            
            if not removable:
                # No single removal possible; try swaps
                break
            
            # Strategy: pick the removal that maximizes "slack" (total over-coverage removed)
            # This is a greedy surrogate for the smooth min-coverage objective
            best_ci = None
            best_score = -1
            sample_size = min(len(removable), 50)
            sample = rng.sample(removable, sample_size) if len(removable) > sample_size else removable
            
            for ci in sample:
                ball = balls_np[ci]
                # Score: how much over-coverage we eliminate (higher = better removal)
                score = int(np.sum(coverage[ball] - 1))
                # Penalize if removal creates new bottlenecks
                min_after = np.min(coverage[ball]) - 1
                if min_after < 0:
                    continue
                # Smooth surrogate: use log-sum-exp style soft-min
                # Approximate: score = sum(coverage-1) - 10*max(0, 2-min_after)
                penalty = 10.0 * max(0, 2 - min_after)
                adj_score = score - penalty
                if adj_score > best_score:
                    best_score = adj_score
                    best_ci = ci
            
            if best_ci is None:
                break
            
            # Perform the removal
            coverage[balls_np[best_ci]] -= 1
            code_set.discard(best_ci)
            new_size = len(code_set)
            
            if new_size < best_local_size:
                best_local_size = new_size
                best_local_set = set(code_set)
            
            # After removal, try to see if we can add back a different word
            # that covers more scarce areas, enabling further removals
            if new_size < current_size - 1:
                # We've made progress; continue
                current_size = new_size
        
        return best_local_set

    def simulated_annealing(initial_set, time_limit):
        """SA search minimizing code size while maintaining full coverage."""
        code_set = set(initial_set)
        coverage = compute_coverage(code_set)
        current_size = len(code_set)
        best_local_set = set(code_set)
        best_local_size = current_size
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.1
        cooling = 0.995
        
        T = T_start
        no_improve_count = 0
        
        while time.time() - start_time < time_limit and T > T_end:
            T *= cooling
            
            # Generate a move
            move_type = rng.random()
            if move_type < 0.4:
                # Try to remove a codeword (if redundant)
                redundant = []
                for ci in code_set:
                    if np.min(coverage[balls_np[ci]]) >= 2:
                        redundant.append(ci)
                if redundant:
                    ci = rng.choice(redundant)
                    old_cov = coverage.copy()
                    coverage[balls_np[ci]] -= 1
                    code_set.discard(ci)
                    new_size = len(code_set)
                    # Check if coverage is still valid (all positions covered)
                    if np.min(coverage) >= 1:
                        delta = new_size - current_size  # negative = improvement
                        if delta <= 0 or rng.random() < np.exp(-delta / T):
                            current_size = new_size
                            if new_size < best_local_size:
                                best_local_size = new_size
                                best_local_set = set(code_set)
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                        else:
                            coverage = old_cov
                            code_set.add(ci)
                    else:
                        # Invalid: revert
                        coverage = old_cov
                        code_set.add(ci)
                else:
                    no_improve_count += 1
                    
            elif move_type < 0.8:
                # Swap: remove one, add one
                if len(code_set) == 0:
                    continue
                ci = rng.choice(list(code_set))
                # Pick a replacement word
                # Prefer words that cover currently scarce areas
                ball_ci = balls_np[ci]
                # Try random candidates
                for _attempt in range(5):
                    new_ci = rng.randint(0, total - 1)
                    if new_ci in code_set:
                        continue
                    old_cov = coverage.copy()
                    coverage[ball_ci] -= 1
                    coverage[balls_np[new_ci]] += 1
                    code_set.discard(ci)
                    code_set.add(new_ci)
                    new_size = len(code_set)
                    if np.min(coverage) >= 1:
                        delta = new_size - current_size
                        if delta <= 0 or rng.random() < np.exp(-delta / T):
                            current_size = new_size
                            if new_size < best_local_size:
                                best_local_size = new_size
                                best_local_set = set(code_set)
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                        else:
                            coverage = old_cov
                            code_set.discard(new_ci)
                            code_set.add(ci)
                        break
                    else:
                        coverage = old_cov
                        code_set.discard(new_ci)
                        code_set.add(ci)
                else:
                    no_improve_count += 1
            else:
                # Add a word (increases size, but may enable future removals)
                new_ci = rng.randint(0, total - 1)
                if new_ci not in code_set:
                    old_cov = coverage.copy()
                    coverage[balls_np[new_ci]] += 1
                    code_set.add(new_ci)
                    new_size = len(code_set)
                    delta = new_size - current_size  # positive = worse
                    if delta <= 0 or rng.random() < np.exp(-delta / T):
                        current_size = new_size
                        # Try greedy removal to see if we can get back
                        greedy_remove(code_set, coverage, max_removals=5)
                        new_size_after = len(code_set)
                        if new_size_after < best_local_size:
                            best_local_size = new_size_after
                            best_local_set = set(code_set)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        coverage = old_cov
                        code_set.discard(new_ci)
                
            # Periodic greedy cleanup
            if no_improve_count > 50:
                greedy_remove(code_set, coverage, max_removals=20)
                new_size_after = len(code_set)
                if new_size_after < best_local_size:
                    best_local_size = new_size_after
                    best_local_set = set(code_set)
                    no_improve_count = 0
                no_improve_count = 0
        
        return best_local_set

    # Phase 1: Initial greedy pruning from the 81-word direct sum
    cov = compute_coverage(best_code_set)
    greedy_remove(best_code_set, cov, max_removals=200)
    best_size = len(best_code_set)
    
    # Phase 2: Local search swap phase to further reduce
    if time.time() - start_time < _DEADLINE - 30:
        ls_result = local_search_swap(best_code_set, cov, _DEADLINE - 30, max_iterations=3000)
        if len(ls_result) < best_size:
            best_size = len(ls_result)
            best_code_set = ls_result
            cov = compute_coverage(best_code_set)
    
    # Phase 3: Simulated annealing with multiple restarts
    time_limit = _DEADLINE - 5
    num_restarts = 8
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if r == 0:
            start_set = set(best_code_set)
        else:
            # Perturb the best solution
            start_set = set(best_code_set)
            # Remove some random words and re-add different ones
            to_remove = min(len(start_set) // 10, 5)
            for _ in range(to_remove):
                if start_set:
                    ci = rng.choice(list(start_set))
                    start_set.discard(ci)
                    # Add a random replacement
                    new_ci = rng.randint(0, total - 1)
                    if new_ci not in start_set:
                        start_set.add(new_ci)
        
        # First do greedy removal on the perturbed set
        cov_r = compute_coverage(start_set)
        greedy_remove(start_set, cov_r, max_removals=100)
        
        # Run SA
        sa_result = simulated_annealing(start_set, time_limit)
        
        if len(sa_result) < best_size:
            best_size = len(sa_result)
            best_code_set = sa_result
            if best_size <= 73:
                break
    
    # Final greedy cleanup
    cov = compute_coverage(best_code_set)
    greedy_remove(best_code_set, cov, max_removals=100)
    
    # Final local search polish
    if time.time() - start_time < _DEADLINE - 3:
        ls_final = local_search_swap(best_code_set, cov, _DEADLINE - 3, max_iterations=2000)
        if len(ls_final) < best_size:
            best_size = len(ls_final)
            best_code_set = ls_final
    
    # Verify coverage
    cov = compute_coverage(best_code_set)
    if np.min(cov) < 1:
        # Fallback to initial if somehow invalid
        best_code_set = set(_word_to_int(c, q, n) for c in initial_code)
    
    result = [_int_to_word(ci, q, n) for ci in sorted(best_code_set)]
    return result
# EVOLVE-BLOCK-END
