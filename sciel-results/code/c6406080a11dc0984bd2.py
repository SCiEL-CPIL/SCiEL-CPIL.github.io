import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for 120s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task is specific
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback: Direct sum or random
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists
    neighbors_list = []
    neighbors_set = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbors_set.append(set(nb))

    def greedy_construct_random(top_k=5, seed=None):
        """
        Greedy set cover with random tie-breaking among top_k candidates.
        """
        if seed is not None:
            random.seed(seed)
        
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        # First pick: 0 (symmetry)
        code.append(0)
        code_set.add(0)
        coverage[neighbors_list[0]] += 1
        uncovered_mask[neighbors_list[0]] = False
        
        while uncovered_mask.any():
            # Calculate gains for all candidates
            # This is the bottleneck. 
            # We can optimize by only checking candidates that cover at least one uncovered word.
            # But for N=2187, a full scan is ~30k ops.
            
            best_candidates = []
            max_gain = 0
            
            # Optimization: If we know the max possible gain is the number of uncovered words, 
            # we can stop early if we find a perfect cover? 
            # Unlikely in later steps.
            
            for c in range(N):
                if c in code_set:
                    continue
                cnt = np.sum(uncovered_mask[neighbors_list[c]])
                if cnt > max_gain:
                    max_gain = cnt
                    best_candidates = [c]
                elif cnt == max_gain and max_gain > 0:
                    best_candidates.append(c)
            
            if max_gain == 0:
                break # Should not happen if uncovered_mask.any()
            
            # Pick randomly from top candidates
            if len(best_candidates) > top_k:
                # If too many, pick top_k randomly? Or just pick one randomly from all?
                # Picking from all ties is better for diversity.
                # But if there are 100 ties, picking from 100 is fine.
                chosen = random.choice(best_candidates)
            else:
                chosen = random.choice(best_candidates)
                
            code.append(chosen)
            code_set.add(chosen)
            coverage[neighbors_list[chosen]] += 1
            uncovered_mask[neighbors_list[chosen]] = False
            
        return code, code_set, coverage, uncovered_mask

    # Run multiple greedy constructions to find a good start
    best_initial_code = None
    best_initial_size = float('inf')
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline - 30: # Keep time for local search
            break
        c, cs, cov, um = greedy_construct_random(top_k=5, seed=i)
        if len(c) < best_initial_size:
            best_initial_size = len(c)
            best_initial_code = c
            best_initial_set = cs
            best_initial_cov = cov
            best_initial_um = um
            
    if best_initial_code is None:
        # Fallback to deterministic if random failed or time ran out
        c, cs, cov, um = greedy_construct_random(top_k=1, seed=0)
        best_initial_code = c
        best_initial_set = cs
        best_initial_cov = cov
        best_initial_um = um

    code = best_initial_code
    code_set = best_initial_set
    coverage = best_initial_cov
    uncovered_mask = best_initial_um

    # ---------- Local Search ----------
    
    def is_redundant(c):
        # Check if removing c leaves all its neighbors covered by others
        # i.e., coverage[neighbors] > 1 for all neighbors
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set):
        """
        Given a set of uncovered word indices, find a minimal set of codewords
        to add to cover them.
        Returns list of added codewords.
        """
        if not uncovered_set:
            return []
        
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            # Candidates are neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])
            
            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = len(neighbors_set[c] & temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(temp_uncovered):
                        break
            
            if best_c == -1:
                break
            
            added.append(best_c)
            temp_uncovered.difference_update(neighbors_set[best_c])
            
        return added

    def apply_code_change(removed_list, added_list):
        """
        Apply the change to global state.
        """
        nonlocal code, code_set, coverage
        for c in removed_list:
            code.remove(c)
            code_set.remove(c)
            coverage[neighbors_list[c]] -= 1
        for c in added_list:
            if c not in code_set:
                code.append(c)
                code_set.add(c)
                coverage[neighbors_list[c]] += 1

    def try_kick(k):
        """
        Try to remove k random codewords and repair.
        Returns True if size decreased.
        """
        if len(code) < k:
            return False
            
        # Pick k distinct random indices
        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        
        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors_list[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal
            apply_code_change(removed, [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)
        
        new_size = len(code) - len(removed) + len(added)
        
        if new_size < len(code):
            # Apply change
            apply_code_change(removed, added)
            return True
        else:
            return False

    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 20 # Reduced to try harder kicks sooner
    
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for c in list(code):
                if is_redundant(c):
                    apply_code_change([c], [])
                    changed = True
                    if time.time() > deadline:
                        break
        
        # 2. Kicks
        kick_success = False
        # Try 1-removal
        if try_kick(1):
            kick_success = True
        # Try 2-removal
        elif try_kick(2):
            kick_success = True
        # Try 3-removal
        elif try_kick(3):
            kick_success = True
        # Try 4-removal if stuck
        elif no_improvement_count > 10 and try_kick(4):
            kick_success = True
        # Try 5-removal if really stuck
        elif no_improvement_count > 30 and try_kick(5):
            kick_success = True
            
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb: Remove a random codeword and force repair, even if size increases?
                # Or just reset the counter and continue.
                # Let's try a "swap" move: remove one, add one different.
                # If we can find a swap that keeps size same but changes structure, it might help.
                # For now, just continue with random kicks.
                pass

    result = [idx_to_tuple(c) for c in code]
    return result
