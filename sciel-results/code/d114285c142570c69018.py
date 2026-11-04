import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total = q ** n
    if total > 5_000_000:
        if q == 3 and n >= 4 and radius == 1:
            _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in _TETRA
                    for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    if q == 3 and n == 8 and radius == 1:
        return _solve_3_8_1(deadline)

    return _generic_greedy(q, n, radius, deadline)


def _solve_3_8_1(deadline):
    q = 3
    n = 8
    radius = 1
    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute neighbors using vectorized approach
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    # Precompute powers of 3
    powers = np.array([q**i for i in range(n)], dtype=np.int64)
    
    for idx in range(total_points):
        word = np.zeros(n, dtype=np.int32)
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q
        
        neighbors[idx, 0] = idx
        k = 1
        for pos in range(n):
            orig = word[pos]
            for delta in [1, 2]:
                new_digit = (orig + delta) % q
                # Compute new index efficiently
                diff = new_digit - orig
                new_idx = idx + diff * powers[pos]
                neighbors[idx, k] = new_idx
                k += 1

    def prune(selected):
        """Remove redundant codewords."""
        if not selected:
            return []
        selected = list(selected)
        coverage = np.zeros(total_points, dtype=np.int32)
        for idx in selected:
            coverage[neighbors[idx]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                idx = selected[i]
                if np.min(coverage[neighbors[idx]]) > 1:
                    selected.pop(i)
                    coverage[neighbors[idx]] -= 1
                    changed = True
        return selected

    def greedy_init(seed):
        """Vectorized greedy set cover with random tie-breaking."""
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            if time.time() > deadline - 10:
                break
                
            uncovered_counts = np.sum(~is_covered[neighbors], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
                
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
                
            selected.append(int(best_idx))
            is_covered[neighbors[best_idx]] = True
            
        return selected

    def local_search(initial_code, seed, deadline_ls):
        """Local search with neutral swaps and kicks to escape local minima."""
        rng = np.random.default_rng(seed)
        code = prune(list(initial_code))
        best_code = code[:]
        best_len = len(code)
        
        coverage = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            coverage[neighbors[idx]] += 1
        
        code_set = set(code)
        
        # Phase 1: Standard improving moves (prune + valid swaps)
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline_ls - 5:
            improved = False
            
            # Try to prune
            pruned = prune(code)
            if len(pruned) < len(code):
                code = pruned
                code_set = set(code)
                coverage = np.zeros(total_points, dtype=np.int32)
                for idx in code:
                    coverage[neighbors[idx]] += 1
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
                improved = True
                no_improve_count = 0
                continue
            
            if not code:
                break
            
            # Try swaps: remove c, add c_prime
            c_idx = rng.integers(len(code))
            c = code[c_idx]
            
            # Choose target: neighbor of c (80%) or random (20%)
            if rng.random() < 0.8:
                n_idx = rng.integers(1, ball_size)
                c_prime = int(neighbors[c, n_idx])
            else:
                c_prime = int(rng.integers(total_points))
            
            if c_prime in code_set:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                continue
            
            # Check if swap preserves coverage
            new_coverage = coverage.copy()
            new_coverage[neighbors[c]] -= 1
            new_coverage[neighbors[c_prime]] += 1
            
            if np.min(new_coverage) >= 0:
                # Valid swap (preserves coverage)
                code[c_idx] = c_prime
                code_set.remove(c)
                code_set.add(c_prime)
                coverage = new_coverage
                improved = True
                no_improve_count = 0
                
                # Try to prune after swap
                pruned = prune(code)
                if len(pruned) < len(code):
                    code = pruned
                    code_set = set(code)
                    coverage = np.zeros(total_points, dtype=np.int32)
                    for idx in code:
                        coverage[neighbors[idx]] += 1
                    if len(code) < best_len:
                        best_len = len(code)
                        best_code = code[:]
            else:
                # Invalid swap - try a kick: temporarily accept violation, then repair
                # This is the key improvement: allow temporary coverage holes
                if rng.random() < 0.3:  # 30% chance to attempt kick
                    # Apply the swap anyway (creates coverage holes)
                    code[c_idx] = c_prime
                    code_set.remove(c)
                    code_set.add(c_prime)
                    coverage = new_coverage
                    
                    # Repair: add words to cover uncovered points
                    uncovered_mask = coverage == 0
                    if np.any(uncovered_mask):
                        # Greedily add words to cover uncovered points
                        repair_rounds = 0
                        while np.any(uncovered_mask) and repair_rounds < 20:
                            if time.time() > deadline_ls - 3:
                                break
                            uncovered_counts = np.sum(uncovered_mask[neighbors], axis=1)
                            max_count = np.max(uncovered_counts)
                            if max_count == 0:
                                break
                            candidates = np.where(uncovered_counts == max_count)[0]
                            # Prefer candidates not already in code
                            new_candidates = [c for c in candidates if c not in code_set]
                            if new_candidates:
                                best_add = int(rng.choice(new_candidates))
                            else:
                                best_add = int(rng.choice(candidates))
                            
                            code.append(best_add)
                            code_set.add(best_add)
                            coverage[neighbors[best_add]] += 1
                            uncovered_mask = coverage == 0
                            repair_rounds += 1
                        
                        # Prune after repair
                        code = prune(code)
                        code_set = set(code)
                        coverage = np.zeros(total_points, dtype=np.int32)
                        for idx in code:
                            coverage[neighbors[idx]] += 1
                        
                        if len(code) < best_len:
                            best_len = len(code)
                            best_code = code[:]
                        improved = True
                        no_improve_count = 0
                    else:
                        # No uncovered points, just prune
                        code = prune(code)
                        code_set = set(code)
                        coverage = np.zeros(total_points, dtype=np.int32)
                        for idx in code:
                            coverage[neighbors[idx]] += 1
                        if len(code) < best_len:
                            best_len = len(code)
                            best_code = code[:]
                        improved = True
                        no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        # Phase 2: Final polish - aggressive pruning
        best_code = prune(best_code)
        return best_code

    # Main logic
    best_code = None
    best_len = float('inf')
    
    # Run multiple greedy instances
    num_greedy = 8
    greedy_codes = []
    
    for i in range(num_greedy):
        if time.time() > deadline - 30:
            break
        code = greedy_init(i)
        pruned = prune(code)
        greedy_codes.append(pruned)
        if len(pruned) < best_len:
            best_len = len(pruned)
            best_code = pruned[:]

    # Local search on the best greedy results
    greedy_codes.sort(key=len)
    
    # Take top 3 for local search
    for code in greedy_codes[:3]:
        if time.time() > deadline - 10:
            break
        ls_code = local_search(code, seed=42 + len(code), deadline_ls=deadline)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code[:]
    
    # Final prune
    if best_code is not None:
        best_code = prune(best_code)
    
    # Convert indices back to words
    if best_code is None:
        _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in _TETRA
                for tail in itertools.product(range(q), repeat=n - 4)]

    result = []
    for idx in best_code:
        word = [0] * n
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q
        result.append(word)
        
    return result


def _generic_greedy(q, n, radius, deadline):
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    balls = []
    for w in all_words:
        w_idx = word_to_idx[w]
        neighbors = [w_idx]
        for i in range(n):
            for d in range(q):
                if d != w[i]:
                    new_w = list(w)
                    new_w[i] = d
                    neighbors.append(word_to_idx[tuple(new_w)])
        balls.append(neighbors)
        
    uncovered = np.ones(total, dtype=bool)
    covered_count = total
    chosen = []
    
    while covered_count > 0 and time.time() < deadline:
        best_idx = -1
        best_gain = 0
        for i in range(total):
            gain = 0
            for idx in balls[i]:
                if uncovered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        chosen.append(best_idx)
        for idx in balls[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                covered_count -= 1
                
    return [list(all_words[i]) for i in chosen]
