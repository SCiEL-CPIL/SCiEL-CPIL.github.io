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

    def prune(selected):
        """Remove redundant codewords."""
        if not selected:
            return []
        selected = list(selected)
        # Count coverage
        coverage = np.zeros(total, dtype=np.int32)
        for idx in selected:
            coverage[neighbors[idx]] += 1
        
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to handle removals easily
            for i in range(len(selected) - 1, -1, -1):
                idx = selected[i]
                # Check if any point in this ball is covered only by this codeword
                # If min(coverage[neighbors[idx]]) > 1, it is redundant
                if np.min(coverage[neighbors[idx]]) > 1:
                    selected.pop(i)
                    coverage[neighbors[idx]] -= 1
                    changed = True
        return selected

    def greedy_init(seed):
        """Vectorized greedy set cover."""
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            if time.time() > deadline - 5:
                break
                
            # Calculate gains: number of uncovered points in each ball
            # ~is_covered is boolean array of size total
            # neighbors is (total, ball_size)
            # ~is_covered[neighbors] is (total, ball_size) boolean
            # Sum along axis 1
            uncovered_counts = np.sum(~is_covered[neighbors], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
                
            # Find candidates with max count
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                # Random tie-breaking
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
                
            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True
            
        return selected

    def local_search(initial_code, deadline):
        """Kick and Prune local search."""
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in local search steps? Or random. Let's use random.
        rng = np.random.default_rng()
        
        code = prune(list(initial_code))
        best_code = code[:]
        best_len = len(code)
        
        # Coverage counts for current code
        coverage = np.zeros(total, dtype=np.int32)
        for idx in code:
            coverage[neighbors[idx]] += 1
            
        # Set of current codewords for O(1) lookup
        code_set = set(code)
        
        # Temperature for simulated annealing? 
        # We only accept moves that keep coverage valid.
        # We want to minimize len(code).
        # A move is: remove c, add c'. 
        # If len(code) stays same, we might accept it to escape local minima?
        # But we immediately prune. If pruning reduces size, we definitely accept.
        # If not, we revert.
        
        time_limit = deadline - 2
        attempts = 0
        
        while time.time() < time_limit:
            attempts += 1
            if not code:
                break
                
            # Pick a random codeword to move
            c_idx = rng.integers(len(code))
            c = code[c_idx]
            
            # Pick a random target
            # 80% chance: neighbor of c
            # 20% chance: random word
            if rng.random() < 0.8:
                n_idx = rng.integers(1, ball_size)
                c_prime = neighbors[c, n_idx]
            else:
                c_prime = rng.integers(total)
            
            if c_prime in code_set:
                continue
            
            # Tentative swap: remove c, add c_prime
            # Check validity: 
            # 1. Decrement coverage of c
            # 2. Increment coverage of c_prime
            # 3. Check if min(coverage) >= 1
            
            # Optimization: 
            # Only points in B(c) are decremented. 
            # Only points in B(c_prime) are incremented.
            # Points outside B(c) and B(c_prime) are unaffected.
            # Points in B(c) \ B(c_prime) are decremented. If any had count 1, they become 0 -> Invalid.
            # Points in B(c_prime) \ B(c) are incremented.
            # Points in B(c) \cap B(c_prime) are net 0 change.
            
            # So we only need to check min(coverage[B(c) \ B(c_prime)]) > 0?
            # Actually, if we decrement B(c), any point in B(c) with count 1 becomes 0.
            # If that point is also in B(c_prime), it gets incremented back to 1.
            # So the condition is: 
            # For all p in B(c): if coverage[p] == 1, then p must be in B(c_prime).
            
            # Let's just do the full array update check, it's fast enough (6561 ints).
            # But we can be smarter.
            
            # Unique points in B(c)
            # unique_mask = coverage[neighbors[c]] == 1
            # if np.any(unique_mask):
            #    unique_points = neighbors[c][unique_mask]
            #    # Check if all unique_points are in B(c_prime)
            #    # This is O(|unique| * log(|B|)) or similar.
            
            # Let's stick to the simple array check first, it's robust.
            new_coverage = coverage.copy()
            new_coverage[neighbors[c]] -= 1
            new_coverage[neighbors[c_prime]] += 1
            
            if np.min(new_coverage) < 1:
                # Invalid swap
                continue
            
            # Valid swap
            # Update state
            coverage = new_coverage
            code_set.remove(c)
            code_set.add(c_prime)
            code[c_idx] = c_prime
            
            # Try to prune
            pruned_code = prune(code)
            
            if len(pruned_code) < len(code):
                # Improvement!
                code = pruned_code
                code_set = set(code)
                # Recalculate coverage
                coverage = np.zeros(total, dtype=np.int32)
                for idx in code:
                    coverage[neighbors[idx]] += 1
                
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
            else:
                # No improvement in size, revert to previous state
                # Revert code
                code[c_idx] = c
                code_set.remove(c_prime)
                code_set.add(c)
                
                # Revert coverage
                coverage = np.zeros(total, dtype=np.int32)
                for idx in code:
                    coverage[neighbors[idx]] += 1
                
        return best_code

    # Run multiple rounds of greedy + local search with different seeds
    best_code = None
    best_len = float('inf')
    
    # Initial greedy
    init_code = greedy_init(0)
    init_code = prune(init_code)
    best_code = init_code
    best_len = len(init_code)
    
    # Local search on the initial code
    improved_code = local_search(best_code, deadline)
    if len(improved_code) < best_len:
        best_len = len(improved_code)
        best_code = improved_code
    
    # If time permits, try more random restarts
    # Each restart: Greedy (fast) + Local Search (expensive)
    # Greedy takes ~0.5s. Local search takes remaining time.
    # We can do a few quick restarts if time allows.
    
    while time.time() < deadline - 10:
        seed = random.randint(0, 1000000)
        init_code = greedy_init(seed)
        init_code = prune(init_code)
        
        # Quick local search with limited time?
        # The local_search function uses the global deadline. 
        # We should pass a sub-deadline or just let it run until the main deadline.
        # Since local_search breaks on deadline, it's safe.
        
        improved_code = local_search(init_code, deadline)
        
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
