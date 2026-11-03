import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
        # Parameters for this specific task
        # q=3, n=8, radius=1
        if q != 3 or n != 8 or radius != 1:
            # Fallback to seed for unexpected inputs
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

        N = q ** n
        powers = [q**i for i in range(n)]
        
        # Precompute neighbors
        neighbors_list = []
        for i in range(N):
            nbrs = [i]
            for k in range(n):
                current_digit = (i // powers[k]) % q
                for v in range(q):
                    if v == current_digit: continue
                    new_i = i - current_digit * powers[k] + v * powers[k]
                    nbrs.append(new_i)
            neighbors_list.append(nbrs)
        neighbors = np.array(neighbors_list, dtype=np.int32)
        
        def index_to_digits(idx):
            digits = []
            temp = idx
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            return digits

        def run_greedy(random_seed=None, use_noise=False):
            if random_seed is not None:
                np.random.seed(random_seed)
                random.seed(random_seed)
                
            covered = np.zeros(N, dtype=bool)
            code = []
            
            # Pre-allocate noise array if needed
            noise = np.zeros(N) if use_noise else None
            
            while True:
                if use_noise:
                    # Add small random noise to break ties
                    noise = np.random.uniform(-0.1, 0.1, N)
                
                mask = ~covered[neighbors]
                counts = mask.sum(axis=1)
                
                if use_noise:
                    counts = counts + noise
                
                best_idx = counts.argmax()
                best_count = counts[best_idx]
                
                # If best_count is 0 (or close to 0 if noise), we are done
                # Check actual count without noise for termination
                actual_count = mask[best_idx].sum()
                if actual_count == 0:
                    break
                
                code.append(best_idx)
                covered[neighbors[best_idx]] = True
                
            return code

        def local_search_swap(code_list, covered_arr, neighbors_arr, time_limit):
            """Try to improve by swapping codewords or removing redundant ones."""
            code_set = set(code_list)
            covered_count = covered_arr.sum()
            start = time.time()
            
            # Try to find and remove redundant codewords
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for i in range(len(code_list)):
                    if time.time() > time_limit:
                        break
                    # Temporarily remove code_list[i]
                    removed = code_list[i]
                    code_set.discard(removed)
                    
                    # Check if all its neighbors are still covered by other codewords
                    nbrs = neighbors_arr[removed]
                    all_covered = True
                    for nbr in nbrs:
                        # Check if nbr is covered by any other codeword
                        # A word is covered if it's in the neighbor set of some other codeword
                        # We need to check if nbr is in neighbors[c] for some c in code_set
                        found = False
                        for c in code_set:
                            if nbr in neighbors_arr[c]:
                                found = True
                                break
                        if not found:
                            all_covered = False
                            break
                    
                    if all_covered:
                        # Safe to remove
                        code_list.pop(i)
                        improved = True
                        # Update covered array
                        covered_arr = np.zeros(len(covered_arr), dtype=bool)
                        for c in code_list:
                            covered_arr[neighbors_arr[c]] = True
                    else:
                        code_set.add(removed)
            
            # Try swap moves: replace one codeword with another to cover more
            for _ in range(10):  # Limited iterations
                if time.time() > time_limit:
                    break
                
                # Find a codeword that covers the fewest unique words
                min_coverage = float('inf')
                min_idx = -1
                for i, c in enumerate(code_list):
                    unique = 0
                    nbrs_c = set(neighbors_arr[c])
                    for other in code_list:
                        if other == c:
                            continue
                        nbrs_c -= set(neighbors_arr[other])
                    unique = len(nbrs_c)
                    if unique < min_coverage:
                        min_coverage = unique
                        min_idx = i
                
                if min_idx < 0 or min_coverage == 0:
                    break
                
                # Try to find a better replacement
                candidate = code_list[min_idx]
                # Try random replacements
                for _ in range(50):
                    if time.time() > time_limit:
                        break
                    new_candidate = random.randint(0, len(code_list) - 1)
                    if new_candidate in code_set:
                        continue
                    
                    # Check if new_candidate covers at least as many uncovered words
                    old_unique = set(neighbors_arr[candidate]) - set().union(*[set(neighbors_arr[c]) for c in code_list if c != candidate])
                    new_unique = set(neighbors_arr[new_candidate]) - set().union(*[set(neighbors_arr[c]) for c in code_list if c != candidate])
                    
                    if len(new_unique) > len(old_unique):
                        code_list[min_idx] = new_candidate
                        code_set.discard(candidate)
                        code_set.add(new_candidate)
                        break
            
            return code_list

        start_time = time.time()
        deadline = 110  # 10s margin
        
        best_code = None
        best_len = float('inf')
        
        # Run 1: Deterministic
        code = run_greedy(random_seed=42, use_noise=False)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
        # Run multiple times with noise
        run_id = 0
        while time.time() < deadline:
            run_id += 1
            code = run_greedy(random_seed=run_id, use_noise=True)
            if len(code) < best_len:
                best_len = len(code)
                best_code = code.copy()
                
        # Local search polish on best code
        if best_code is not None and time.time() < deadline:
            covered_arr = np.zeros(N, dtype=bool)
            for c in best_code:
                covered_arr[neighbors[c]] = True
            
            polished = local_search_swap(best_code.copy(), covered_arr, neighbors, deadline)
            if len(polished) < best_len:
                best_len = len(polished)
                best_code = polished
                
        # Convert best_code to required format
        result = [index_to_digits(idx) for idx in best_code]
        return result
