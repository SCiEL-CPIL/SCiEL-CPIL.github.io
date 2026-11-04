import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate all words and map to indices
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute balls as bitsets (Python integers)
    # ball[i] is an integer with bits set for all j such that dist(words[i], words[j]) <= radius
    # For radius 1, we just need i and its immediate neighbors.
    balls = [0] * N
    for i, w in enumerate(words):
        mask = 1 << i
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                mask |= (1 << idx)
                w_list[pos] = orig_val
        balls[i] = mask
        
    all_mask = (1 << N) - 1
    
    # Helper: Count bits (popcount)
    # Python 3.8+ has int.bit_count()
    def popcount(x):
        return x.bit_count()
        
    # Helper: Get list of indices from a bitset
    def bitset_to_list(mask):
        indices = []
        while mask:
            # Get least significant set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            indices.append(idx)
            mask ^= lsb
        return indices

    # 3. Greedy Initial Solution
    def greedy_init():
        covered = 0
        code = []
        # Precompute gain for each word? No, dynamic.
        # We can speed this up by keeping an array of uncovered counts?
        # Simpler: Just iterate.
        # To make it faster, we can use a random tie-breaker or random restarts.
        
        # We will do 5 random greedy runs and keep the best.
        best_code = None
        best_size = N + 1
        
        for _ in range(5):
            covered = 0
            code = []
            # Randomize the order of consideration?
            # Or just standard greedy.
            # Standard greedy: pick word with max new coverage.
            
            # Optimization: Maintain a list of candidate words?
            # With N=2187, iterating all N each step is 2187 * 186 ~ 400k ops. Fast.
            
            while covered != all_mask:
                if time.time() > deadline - 50: # Leave time for B&B
                    break
                
                best_idx = -1
                max_gain = 0
                # Randomize tie-breaking
                candidates = []
                for i in range(N):
                    gain = popcount(balls[i] & ~covered)
                    if gain > max_gain:
                        max_gain = gain
                        best_idx = i
                        candidates = [i]
                    elif gain == max_gain and gain > 0:
                        candidates.append(i)
                
                if max_gain == 0:
                    break
                    
                if len(candidates) > 1:
                    best_idx = random.choice(candidates)
                
                code.append(best_idx)
                covered |= balls[best_idx]
                
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
                
        return best_code if best_code else []

    # 4. Local Search (Swaps)
    def local_search(code):
        if not code:
            return code
        
        # Convert to set for O(1) lookup
        code_set = set(code)
        covered = 0
        for c in code:
            covered |= balls[c]
            
        improved = True
        iterations = 0
        max_iters = 100
        
        while improved and iterations < max_iters and time.time() < deadline - 20:
            improved = False
            iterations += 1
            random.shuffle(code)
            
            for i, c in enumerate(code):
                if time.time() > deadline - 20:
                    break
                    
                # Try to remove c
                temp_covered = covered
                # We need to know if removing c leaves any hole.
                # holes = ~(covered ^ balls[c]) & all_mask ? No.
                # If we remove c, new covered = covered & ~balls[c] ? No, other words might cover those bits.
                # We need counts.
                # For efficiency in local search, let's use a counts array?
                # Or just recompute coverage of the set without c?
                # That is O(|code|) per removal. |code| ~ 200. 200 * 200 = 40k. OK.
                
                # Let's check if c is redundant.
                # c is redundant if balls[c] is subset of union of balls of other words.
                # i.e., balls[c] & ~(covered ^ balls[c]) == 0 ?
                # Let rest_covered = 0
                # for c2 in code: if c2 != c: rest_covered |= balls[c2]
                # if balls[c] & ~rest_covered == 0: redundant.
                
                rest_covered = 0
                for c2 in code:
                    if c2 != c:
                        rest_covered |= balls[c2]
                
                if balls[c] & ~rest_covered == 0:
                    # Redundant, remove
                    code.remove(c)
                    covered = rest_covered
                    improved = True
                    break # Restart loop
                    
                # Try to swap c with some w not in code
                # w must cover the "critical" bits of c.
                # critical_bits = balls[c] & ~rest_covered
                # w must satisfy: critical_bits & ~balls[w] == 0
                # i.e., critical_bits is subset of balls[w]
                
                critical = balls[c] & ~rest_covered
                if critical == 0:
                    continue # Should have been removed
                
                # Find w not in code such that critical is subset of balls[w]
                # And we want to maximize gain? Or just find any?
                # To reduce size, we just need to find any valid swap?
                # No, swap keeps size same. We want to REMOVE.
                # So we only remove if redundant.
                # To reduce size, we need to find a w that allows removing c AND maybe another word?
                # Or just stick to removals and additions?
                # The "Kick and Repair" in seed was good.
                
        return code

    # 5. Branch and Bound
    # Global state
    best_size_global = N + 1
    best_code_global = []
    
    def bnb(covered_mask, current_size, current_set):
        nonlocal best_size_global, best_code_global
        
        if time.time() > deadline - 5:
            raise TimeoutError("Time limit exceeded")
            
        if current_size >= best_size_global:
            return
            
        if covered_mask == all_mask:
            best_size_global = current_size
            best_code_global = list(current_set)
            return
            
        # Lower bound
        uncovered = all_mask ^ covered_mask
        unc_count = popcount(uncovered)
        lb = (unc_count + 14) // 15  # Ceiling division by max ball size 15
        
        if current_size + lb >= best_size_global:
            return
            
        # Pick an uncovered vertex
        # Choose the one with the fewest neighbors in the "available" set?
        # Or just the first one.
        # Let's pick the first set bit in uncovered.
        lsb = uncovered & -uncovered
        u_idx = lsb.bit_length() - 1
        
        # Get all v such that u_idx is in balls[v]
        # This is equivalent to: v is a neighbor of u_idx.
        # We can precompute the list of neighbors for each u_idx.
        # neighbors_of[u_idx] = list of v such that bit u_idx is set in balls[v].
        
        # Let's precompute this outside.
        
    # Precompute neighbors_of
    neighbors_of = [[] for _ in range(N)]
    for v in range(N):
        mask = balls[v]
        while mask:
            lsb = mask & -mask
            u = lsb.bit_length() - 1
            neighbors_of[u].append(v)
            mask ^= lsb
            
    # Update BNB to use neighbors_of
    def bnb(covered_mask, current_size, current_set):
        nonlocal best_size_global, best_code_global
        
        if time.time() > deadline - 5:
            raise TimeoutError("Time limit exceeded")
            
        if current_size >= best_size_global:
            return
            
        if covered_mask == all_mask:
            best_size_global = current_size
            best_code_global = list(current_set)
            return
            
        uncovered = all_mask ^ covered_mask
        unc_count = popcount(uncovered)
        lb = (unc_count + 14) // 15
        
        if current_size + lb >= best_size_global:
            return
            
        # Pick an uncovered vertex
        # Heuristic: Pick the uncovered vertex with the smallest degree in the "uncovered subgraph"?
        # Simpler: Pick the one with the fewest neighbors that are NOT already in current_set?
        # Actually, just pick the first one for now, or the one with min popcount(balls[v] & uncovered) for v in neighbors?
        # Let's pick the first one.
        lsb = uncovered & -uncovered
        u_idx = lsb.bit_length() - 1
        
        # Get candidates
        candidates = neighbors_of[u_idx]
        
        # Sort candidates by gain (popcount(balls[v] & uncovered)) descending
        # This helps find a good solution quickly to prune.
        candidates.sort(key=lambda v: popcount(balls[v] & uncovered), reverse=True)
        
        for v in candidates:
            # If v is already in current_set, we don't need to add it again?
            # In a set cover, we don't add duplicates.
            if v in current_set:
                continue
                
            current_set.append(v)
            bnb(covered_mask | balls[v], current_size + 1, current_set)
            current_set.pop()
            
            if time.time() > deadline - 5:
                raise TimeoutError("Time limit exceeded")

    # Run Greedy
    initial_code = greedy_init()
    if not initial_code:
        # Fallback: return all words
        return [list(words[i]) for i in range(N)]
        
    # Run Local Search
    initial_code = local_search(initial_code)
    
    # Update Global Best
    best_size_global = len(initial_code)
    best_code_global = list(initial_code)
    
    # Run B&B
    try:
        bnb(0, 0, [])
    except TimeoutError:
        pass
    except RecursionError:
        # Increase recursion limit or handle differently
        sys.setrecursionlimit(10000)
        try:
            bnb(0, 0, [])
        except:
            pass

    # Final Polish
    if best_code_global:
        best_code_global = local_search(best_code_global)
        # Verify
        covered = 0
        for c in best_code_global:
            covered |= balls[c]
        if covered != all_mask:
            # If invalid, fallback to initial
            best_code_global = initial_code

    return [list(words[i]) for i in best_code_global]
