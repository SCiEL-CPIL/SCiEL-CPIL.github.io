# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

# Precompute the full space and neighborhood structure for the specific task (q=3, n=7, r=1)
# Space size: 3^7 = 2187
# Neighborhood size for each word: 1 (itself) + 7 * 2 (one position changed to 2 other values) = 15
# Total pairs: 2187 * 15 = 32805. This is small enough to precompute as bitsets.

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # If the problem is not the specific one we optimized for, fall back to a simple heuristic
    # to ensure correctness and avoid timeout, though the task specifies q=3, n=7, r=1.
    if q != 3 or n != 7 or radius != 1:
        # Simple greedy fallback for other parameters
        return _greedy_fallback(q, n, radius, deadline)

    # --- Optimized path for q=3, n=7, r=1 ---
    
    # 1. Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words) # 2187
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # 2. Precompute neighbors (including self) as bitsets (Python integers for bit operations)
    # neighbor_mask[i] is a bitset of all words within distance <= 1 from word i
    neighbor_masks = [0] * N
    
    for i, w in enumerate(all_words):
        mask = 1 << i  # Include self
        # Generate neighbors at distance 1
        w_list = list(w)
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val == old_val:
                    continue
                w_list[pos] = new_val
                nb = tuple(w_list)
                j = word_to_idx[nb]
                mask |= (1 << j)
            w_list[pos] = old_val
        neighbor_masks[i] = mask

    # 3. Greedy Set Cover to get an initial solution
    # We want to select a subset of words (codewords) such that the union of their neighbor_masks covers all bits (all words).
    
    def greedy_set_cover():
        uncovered = (1 << N) - 1
        selected = []
        remaining_masks = neighbor_masks[:]
        remaining_indices = list(range(N))
        
        while uncovered:
            # Find the word that covers the most uncovered elements
            best_idx = -1
            best_count = -1
            # Optimization: We can iterate through remaining_indices. 
            # To speed this up, we could maintain a heap or just iterate. 
            # N=2187, iterations ~100-200. 2187*150 = ~300k operations. Fast.
            for idx in remaining_indices:
                if not (uncovered & remaining_masks[idx]):
                    continue
                count = (uncovered & remaining_masks[idx]).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            if best_idx == -1:
                break # Should not happen if cover exists
            
            selected.append(best_idx)
            uncovered &= ~remaining_masks[best_idx]
            remaining_indices.remove(best_idx)
            
        return selected

    # 4. Local Search / Reduction
    # Try to remove codewords if they are redundant.
    def reduce_solution(solution):
        solution = list(solution)
        changed = True
        while changed:
            changed = False
            for i in range(len(solution)):
                # Check if removing solution[i] still covers everything
                # This is expensive if done naively. 
                # Better: For each codeword c, check if there is any word w in its neighborhood 
                # such that w is NOT covered by any other codeword in the set.
                
                # Let's compute the coverage of the set without solution[i].
                # Actually, a faster check:
                # A codeword c is redundant if for every word w in N(c), there exists another c' in solution \ {c} such that w in N(c').
                
                # Let's implement this check.
                redundant = True
                # We only need to check words covered by c.
                # But checking all 15 neighbors for each of ~180 codewords is 2700 checks.
                # Each check requires knowing if the neighbor is covered by others.
                
                # Alternative: 
                # Compute total coverage bitset for current solution.
                # If we remove c, the new coverage is (total_coverage_without_c).
                # It is valid if total_coverage_without_c == all_ones.
                
                # Let's maintain a count of how many codewords cover each word.
                # This allows O(1) check for redundancy.
                
                pass # We will rewrite this using a count array for efficiency.
        
        return solution

    # Let's use a more efficient local search strategy:
    # 1. Get initial solution via greedy.
    # 2. Use a "swap" or "remove" local search.
    #    - Try to remove any codeword. If valid, remove it.
    #    - If stuck, try to swap one codeword for another? Or just random restarts of greedy.
    
    # Given the small size, multiple randomized greedy restarts + reduction is likely best.
    
    def greedy_with_randomization(seed):
        random.seed(seed)
        uncovered = (1 << N) - 1
        selected = []
        remaining_indices = list(range(N))
        random.shuffle(remaining_indices) # Randomize tie-breaking order? 
        # Actually, standard greedy picks max coverage. 
        # Randomization can be done by adding small random noise to the coverage count.
        
        while uncovered:
            best_idx = -1
            best_score = -1
            for idx in remaining_indices:
                mask = remaining_masks[idx]
                if not (uncovered & mask):
                    continue
                count = (uncovered & mask).bit_count()
                # Add small random noise to break ties
                score = count + random.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            uncovered &= ~remaining_masks[best_idx]
            remaining_indices.remove(best_idx)
            
        return selected

    def is_valid_solution(solution):
        if not solution:
            return N == 0
        cover = 0
        for idx in solution:
            cover |= neighbor_masks[idx]
        return cover == (1 << N) - 1

    def try_reduce(solution):
        # Greedy removal
        solution = list(solution)
        changed = True
        while changed:
            changed = False
            # Compute coverage counts for current solution
            # counts[w] = number of codewords in solution that cover w
            counts = [0] * N
            for idx in solution:
                mask = neighbor_masks[idx]
                # Iterate through set bits in mask to increment counts
                m = mask
                while m:
                    # Get lowest set bit
                    lsb = m & -m
                    pos = lsb.bit_length() - 1
                    counts[pos] += 1
                    m ^= lsb
            
            # Try to remove each
            for i in range(len(solution)):
                idx = solution[i]
                mask = neighbor_masks[idx]
                # Check if any word covered by idx has count == 1 (only covered by idx)
                # If so, idx cannot be removed.
                m = mask
                can_remove = True
                while m:
                    lsb = m & -m
                    pos = lsb.bit_length() - 1
                    if counts[pos] == 1:
                        can_remove = False
                        break
                    m ^= lsb
                
                if can_remove:
                    solution.pop(i)
                    changed = True
                    # Recompute counts? Or update?
                    # For simplicity and correctness, recompute if removed.
                    # Or just break and restart the while loop.
                    break
            
        return solution

    best_solution = None
    best_size = float('inf')
    
    # Run several randomized greedy restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        sol = greedy_with_randomization(i)
        sol = try_reduce(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            
    # If we have time, try some local search swaps?
    # For now, rely on restarts.
    
    if best_solution is None:
        # Fallback to deterministic greedy
        best_solution = greedy_set_cover()
        best_solution = try_reduce(best_solution)
        
    # Convert indices back to words
    return [all_words[i] for i in best_solution]

def _greedy_fallback(q, n, radius, deadline):
    # Simple greedy for general case
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    def dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
    
    def get_neighbors(w):
        # Return set of indices within radius
        nbs = set()
        # Generate all words at distance <= radius is expensive for general q,n,r
        # For fallback, we just use a simple iterative check or assume small space
        # If space is too large, this will be slow, but the task is specific.
        # We'll just do a brute force check for the neighbor mask construction.
        for u in all_words:
            if dist(w, u) <= radius:
                nbs.add(word_to_idx[u])
        return nbs

    # Precompute neighbor masks (indices)
    neighbor_sets = [get_neighbors(w) for w in all_words]
    
    uncovered = set(range(N))
    selected = []
    remaining = list(range(N))
    
    while uncovered:
        best_idx = -1
        best_count = -1
        for idx in remaining:
            if not (uncovered & neighbor_sets[idx]):
                continue
            count = len(uncovered & neighbor_sets[idx])
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        selected.append(best_idx)
        uncovered -= neighbor_sets[best_idx]
        remaining.remove(best_idx)
        
    return [all_words[i] for i in selected]
# EVOLVE-BLOCK-END
