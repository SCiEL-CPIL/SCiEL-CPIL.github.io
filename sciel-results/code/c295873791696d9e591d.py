import numpy as np
import time
import random

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        deadline = time.time() + 118.0
        N = q ** n  # 6561
        powers = [q ** i for i in range(n)]
        
        # Precompute neighbors: neighbors[w] is a list of words within distance 1
        neigh_size = 1 + n * (q - 1)
        neighbors = np.zeros((N, neigh_size), dtype=np.int32)
        
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    new_w = w + diff * powers[i]
                    neighbors[w, idx] = new_w
                    idx += 1
        
        # Precompute inverse neighbors for faster candidate generation
        # inv_neighbors[w] = list of codewords c such that w is in N(c)
        inv_neighbors = np.zeros((N, neigh_size), dtype=np.int32)
        for c in range(N):
            for j in range(neigh_size):
                w = neighbors[c, j]
                # Find slot in inv_neighbors[w]
                for k in range(neigh_size):
                    if inv_neighbors[w, k] == -1 or inv_neighbors[w, k] == c:
                        if inv_neighbors[w, k] != c:
                            inv_neighbors[w, k] = c
                        break
        
        # Initialize inv_neighbors properly
        inv_neighbors = np.full((N, neigh_size), -1, dtype=np.int32)
        for c in range(N):
            for j in range(neigh_size):
                w = neighbors[c, j]
                for k in range(neigh_size):
                    if inv_neighbors[w, k] == -1:
                        inv_neighbors[w, k] = c
                        break
        
        def compute_coverage(code_set):
            cov = np.zeros(N, dtype=np.int32)
            for c in code_set:
                cov[neighbors[c, :]] += 1
            return cov

        def greedy_init(rng):
            uncovered = np.ones(N, dtype=np.bool_)
            code = set()
            while np.any(uncovered) and time.time() < deadline - 5:
                gains = uncovered[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                w = rng.choice(candidates)
                code.add(w)
                uncovered[neighbors[w, :]] = False
            return code

        def try_reduce(code_set, cov, rng):
            """Remove redundant codewords. Returns number removed."""
            removed = 0
            to_remove = []
            for c in code_set:
                if np.min(cov[neighbors[c, :]]) > 1:
                    to_remove.append(c)
            
            if to_remove:
                for c in to_remove:
                    code_set.remove(c)
                    removed += 1
                # Update coverage
                for c in to_remove:
                    cov[neighbors[c, :]] -= 1
            return removed

        def try_swap(code_set, cov, rng, deadline):
            """
            Try to swap a codeword c with d such that:
            1. d covers all words that only c covers (essential words)
            2. After swap, we can remove additional codewords
            Returns (swapped, new_code_set, new_cov) or (False, code_set, cov)
            """
            c_list = list(code_set)
            rng.shuffle(c_list)
            
            # Sort by number of essential words (most constrained first)
            def essential_count(c):
                return np.sum(cov[neighbors[c, :]] == 1)
            
            c_list.sort(key=essential_count, reverse=True)
            
            for c in c_list:
                if time.time() > deadline - 2:
                    return False, code_set, cov
                
                c_neighbors = neighbors[c, :]
                essential_mask = (cov[c_neighbors] == 1)
                essential_words = c_neighbors[essential_mask]
                
                if len(essential_words) == 0:
                    continue
                
                # Find candidates d that cover all essential words
                if len(essential_words) == 1:
                    candidates = neighbors[essential_words[0], :]
                else:
                    cand_set = set(neighbors[essential_words[0], :].tolist())
                    for ew in essential_words[1:]:
                        cand_set.intersection_update(neighbors[ew, :].tolist())
                    candidates = np.array(list(cand_set), dtype=np.int32)
                
                if len(candidates) == 0:
                    continue
                
                valid_candidates = [d for d in candidates if d not in code_set]
                if not valid_candidates:
                    continue
                
                # Try each candidate and see if it enables deletions
                best_d = None
                best_deletions = -1
                best_new_cov = None
                
                for d in valid_candidates[:10]:  # Limit candidates for speed
                    # Simulate swap
                    new_cov = cov.copy()
                    new_cov[neighbors[c, :]] -= 1
                    new_cov[neighbors[d, :]] += 1
                    
                    # Count how many codewords become removable
                    deletions = 0
                    for c2 in code_set:
                        if c2 == c:
                            continue
                        if np.min(new_cov[neighbors[c2, :]]) > 1:
                            deletions += 1
                    
                    # Also check if d itself would be essential (shouldn't be if it covers essentials)
                    if deletions > best_deletions:
                        best_deletions = deletions
                        best_d = d
                        best_new_cov = new_cov
                
                if best_d is not None and best_deletions > 0:
                    # Perform the swap
                    code_set.remove(c)
                    code_set.add(best_d)
                    cov = best_new_cov
                    return True, code_set, cov
            
            return False, code_set, cov

        def local_search(code_set, rng, deadline):
            cov = compute_coverage(code_set)
            no_improvement_count = 0
            
            while time.time() < deadline - 2:
                # Deletion pass
                removed = try_reduce(code_set, cov, rng)
                if removed > 0:
                    no_improvement_count = 0
                    continue
                
                # Swap pass
                swapped, code_set, cov = try_swap(code_set, cov, rng, deadline)
                if swapped:
                    no_improvement_count = 0
                    continue
                
                # No improvement - try perturbation
                no_improvement_count += 1
                if no_improvement_count > 3:
                    break
                
                # Random perturbation: remove a random codeword and re-cover
                if len(code_set) > 0 and rng.random() < 0.3:
                    c = rng.choice(list(code_set))
                    code_set.remove(c)
                    cov[neighbors[c, :]] -= 1
                    
                    # Re-cover uncovered words
                    uncovered = (cov == 0)
                    while np.any(uncovered):
                        gains = uncovered[neighbors].sum(axis=1)
                        max_gain = np.max(gains)
                        if max_gain == 0:
                            break
                        candidates = np.where(gains == max_gain)[0]
                        w = rng.choice(candidates)
                        code_set.add(w)
                        cov[neighbors[w, :]] += 1
                        uncovered[neighbors[w, :]] = False
                    
                    # Reduce again
                    try_reduce(code_set, cov, rng)
                
                no_improvement_count = 0
            
            return code_set

        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts with different strategies
        for i in range(20):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.default_rng(42 + i)
            
            # Greedy Init
            code = greedy_init(rng)
            
            # Local Search
            code = local_search(code, rng, deadline)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                
            if best_size <= 486:
                break
        
        if best_code is None:
            # Fallback to a known structure
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            import itertools
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)

        final_code = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        
        return final_code
    
    else:
        N = q ** n
        deadline = time.time() + 110.0
        powers = [q ** i for i in range(n)]
        neigh_size = 1 + n * (q - 1)
        neighbors = np.zeros((N, neigh_size), dtype=np.int32)
        
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    new_w = w + diff * powers[i]
                    neighbors[w, idx] = new_w
                    idx += 1
        
        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False
        
        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
