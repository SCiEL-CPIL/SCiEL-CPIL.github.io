import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Fallback for non-standard cases
    if q != 3 or n != 8 or radius != 1:
        total_words = q ** n
        powers = [q ** i for i in range(n)]
        neighbors_list = []
        for w in range(total_words):
            nbs = [w]
            for k in range(n):
                d_k = (w // powers[k]) % q
                for v in range(q):
                    if v != d_k:
                        w_new = w + (v - d_k) * powers[k]
                        nbs.append(w_new)
            neighbors_list.append(nbs)
        neighbors = np.array(neighbors_list, dtype=np.int32)
        covered = np.zeros(total_words, dtype=bool)
        code = []
        uncovered_count = total_words
        while uncovered_count > 0:
            scores = np.sum(~covered[neighbors], axis=1)
            max_score = np.max(scores)
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[0]
            code.append(int(best_idx))
            nbs = neighbors[best_idx]
            newly = np.sum(~covered[nbs])
            covered[nbs] = True
            uncovered_count -= newly
        result = []
        for w in code:
            digits = [0] * n
            x = w
            for i in range(n):
                digits[i] = x % q
                x //= q
            result.append(digits)
        return result
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    # Precompute neighbors
    neighbors_list = []
    for w in range(total_words):
        nbs = [w]
        for k in range(n):
            d_k = (w // powers[k]) % q
            for v in range(q):
                if v != d_k:
                    w_new = w + (v - d_k) * powers[k]
                    nbs.append(w_new)
        neighbors_list.append(nbs)
        
    neighbors = np.array(neighbors_list, dtype=np.int32)
    
    def greedy_with_tiebreaking(run_seed):
        """Run greedy with specific tie-breaking strategy."""
        covered = np.zeros(total_words, dtype=bool)
        code = []
        uncovered_count = total_words
        rng = np.random.RandomState(run_seed)
        
        while uncovered_count > 0:
            scores = np.sum(~covered[neighbors], axis=1)
            max_score = np.max(scores)
            candidates = np.where(scores == max_score)[0]
            
            if len(candidates) == 1:
                best_idx = candidates[0]
            else:
                # Weighted random choice among top candidates
                # Prefer candidates that cover more unique uncovered words
                # Use a simple heuristic: pick randomly from top candidates
                best_idx = rng.choice(candidates)
                
            code.append(int(best_idx))
            nbs = neighbors[best_idx]
            newly = np.sum(~covered[nbs])
            covered[nbs] = True
            uncovered_count -= newly
            
        return code
    
    def local_search(code, neighbors, total_words, time_limit):
        """Local search with swap operations to reduce code size."""
        code = list(code)
        covered = np.zeros(total_words, dtype=bool)
        
        # Build coverage map: for each word, which codeword covers it
        # First, build the coverage
        for cw in code:
            covered[neighbors[cw]] = True
        
        # Try to remove codewords if possible
        improved = True
        while improved:
            improved = False
            for i in range(len(code) - 1, -1, -1):
                if time.time() > time_limit:
                    return code
                
                # Check if removing code[i] still covers all words
                nbs = neighbors[code[i]]
                # Count how many words would become uncovered
                # We need to check if any word in nbs is ONLY covered by code[i]
                # For efficiency, count how many codewords cover each word in nbs
                # This is expensive, so use a simpler check:
                # Remove code[i], see if all words in nbs are still covered by others
                
                # Temporarily remove
                old_code = code[i]
                code.pop(i)
                
                # Rebuild coverage without this codeword
                test_covered = np.zeros(total_words, dtype=bool)
                for cw in code:
                    test_covered[neighbors[cw]] = True
                
                # Check if all words are covered
                if np.all(test_covered):
                    # Keep it removed
                    improved = True
                    break
                else:
                    # Restore
                    code.insert(i, old_code)
        
        return code
    
    best_code = None
    best_size = float('inf')
    
    # Run multiple rounds of greedy with different tie-breaking
    num_runs = 8
    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        if run == 0:
            # Deterministic (lowest index)
            covered = np.zeros(total_words, dtype=bool)
            code = []
            uncovered_count = total_words
            while uncovered_count > 0:
                scores = np.sum(~covered[neighbors], axis=1)
                max_score = np.max(scores)
                candidates = np.where(scores == max_score)[0]
                best_idx = candidates[0]
                code.append(int(best_idx))
                nbs = neighbors[best_idx]
                newly = np.sum(~covered[nbs])
                covered[nbs] = True
                uncovered_count -= newly
        else:
            # Random tie-breaking
            code = greedy_with_tiebreaking(run * 12345)
        
        # Apply local search
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            code = local_search(code, neighbors, total_words, deadline - 5)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
    if best_code is None:
        # Fallback to seed logic if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert best_code (list of ints) to list of lists of digits
    result = []
    for w in best_code:
        digits = [0] * n
        x = w
        for i in range(n):
            digits[i] = x % q
            x //= q
        result.append(digits)
        
    return result
