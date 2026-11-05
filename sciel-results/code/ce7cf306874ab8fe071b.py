import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Handle the specific case K_3(8,1)
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Generic fallback for other cases
    return _generic_greedy(q, n, radius, deadline)

def _solve_ternary_8_1(deadline):
    q = 3
    n = 8
    total = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Precompute neighbors (balls)
    # neighbors[idx] is an array of 17 indices
    neighbors = np.empty((total, ball_size), dtype=np.int32)
    
    # Precompute base-3 digits for all indices to speed up neighbor calculation
    # Or just calculate on the fly, 6561 is small.
    # Let's use a fast conversion method.
    # idx -> word (list of 8 ints)
    # word -> idx
    
    # Precompute powers of 3
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    for idx in range(total):
        # Convert idx to base-3 digits
        word = [0] * n
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
                # Calculate new index
                # new_idx = idx - orig * powers[pos] + new_digit * powers[pos]
                new_idx = idx - orig * powers[pos] + new_digit * powers[pos]
                neighbors[idx, k] = new_idx
                k += 1

    def prune(selected, coverage):
        """Remove redundant codewords. Modifies selected and coverage in place? 
        To be safe, returns new list and new coverage array."""
        if not selected:
            return [], coverage
        
        selected = list(selected)
        # We need a local copy of coverage to test removals? 
        # No, we can just test on the current coverage.
        # If we remove idx, coverage[neighbors[idx]] -= 1.
        # We check if min(coverage[neighbors[idx]]) > 1 BEFORE removing.
        
        changed = True
        while changed:
            changed = False
            # Iterate in reverse
            for i in range(len(selected) - 1, -1, -1):
                idx = selected[i]
                # Check if any point in this ball is covered only by this codeword
                # If min(coverage[neighbors[idx]]) > 1, it is redundant
                if np.min(coverage[neighbors[idx]]) > 1:
                    selected.pop(i)
                    coverage[neighbors[idx]] -= 1
                    changed = True
        return selected, coverage

    def greedy_init(seed):
        """Vectorized greedy set cover with immediate pruning."""
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total, dtype=bool)
        selected = []
        coverage = np.zeros(total, dtype=np.int32)
        
        while not np.all(is_covered):
            if time.time() > deadline - 5:
                break
                
            # Calculate gains: number of uncovered points in each ball
            uncovered_counts = np.sum(~is_covered[neighbors], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
                
            # Find candidates with max count
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
                
            # Add best_idx
            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True
            coverage[neighbors[best_idx]] += 1
            
            # Immediate Prune
            selected, coverage = prune(selected, coverage)
            
        return selected, coverage

    def local_search(initial_code, initial_coverage, deadline):
        """Kick and Prune local search with fast updates."""
        rng = np.random.default_rng()
        
        code = list(initial_code)
        coverage = initial_coverage.copy()
        
        best_code = code[:]
        best_coverage = coverage.copy()
        best_len = len(code)
        
        # Set of current codewords for O(1) lookup
        code_set = set(code)
        
        time_limit = deadline - 2
        
        while time.time() < time_limit:
            if not code:
                break
                
            # Pick a random codeword to move
            c_idx = rng.integers(len(code))
            c = code[c_idx]
            
            # Pick a random target
            if rng.random() < 0.8:
                n_idx = rng.integers(1, ball_size)
                c_prime = neighbors[c, n_idx]
            else:
                c_prime = rng.integers(total)
            
            if c_prime in code_set:
                continue
            
            # Check validity of swap c -> c_prime
            # Condition: For all p in B(c), if coverage[p] == 1, then p must be in B(c_prime).
            pts_c = neighbors[c]
            pts_cp = neighbors[c_prime]
            set_cp = set(pts_cp.tolist())
            
            valid = True
            for p in pts_c:
                if coverage[p] == 1:
                    # p is uniquely covered by c.
                    # It must be covered by c_prime.
                    if p not in set_cp:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Valid swap. Update state.
            coverage[pts_c] -= 1
            coverage[pts_cp] += 1
            code_set.remove(c)
            code_set.add(c_prime)
            code[c_idx] = c_prime
            
            # Try to prune
            pruned_code, pruned_coverage = prune(code, coverage)
            
            if len(pruned_code) < len(code):
                # Improvement
                code = pruned_code
                coverage = pruned_coverage
                code_set = set(code)
                
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
                    best_coverage = coverage.copy()
            else:
                # No improvement. Revert.
                # Revert coverage
                coverage[pts_c] += 1
                coverage[pts_cp] -= 1
                # Revert code
                code_set.remove(c_prime)
                code_set.add(c)
                code[c_idx] = c
                
        return best_code, best_coverage

    # Run multiple rounds
    best_code = None
    best_len = float('inf')
    
    # Initial greedy
    init_code, init_cov = greedy_init(0)
    best_code = init_code
    best_len = len(init_code)
    
    # Local search
    improved_code, improved_cov = local_search(best_code, init_cov, deadline)
    if len(improved_code) < best_len:
        best_len = len(improved_code)
        best_code = improved_code
        
    # If time permits, try more random restarts
    while time.time() < deadline - 10:
        seed = random.randint(0, 1000000)
        init_code, init_cov = greedy_init(seed)
        
        # Local search
        improved_code, improved_cov = local_search(init_code, init_cov, deadline)
        
        if len(improved_code) < best_len:
            best_len = len(improved_code)
            best_code = improved_code
            
    # Convert indices back to words
    if best_code is None:
        return []
        
    # Re-convert indices to lists of digits
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
    if total > 100000:
        # Fallback
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    balls = []
    for w in all_words:
        neighbors = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            original = w_list[i]
            for d in range(q):
                if d != original:
                    w_list[i] = d
                    neighbors.append(word_to_idx[tuple(w_list)])
            w_list[i] = original
        balls.append(np.array(neighbors, dtype=np.int32))
        
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    remaining = total
    
    while remaining > 0 and time.time() < deadline:
        gains = np.array([uncovered[b].sum() for b in balls], dtype=np.int32)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        chosen.append(best_idx)
        uncovered[balls[best_idx]] = False
        remaining -= gains[best_idx]
        
    return [all_words[i] for i in chosen]
