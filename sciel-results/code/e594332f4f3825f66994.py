"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # For the specific case q=3, n=6, radius=1
    total_words = q ** n
    
    def word_to_idx(word, q, n):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx, q, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute neighbors for each word (for radius=1)
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx, q, n)
        neighbors[idx].append(idx)  # self
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        nb_idx = word_to_idx(w, q, n)
                        neighbors[idx].append(nb_idx)
                w[pos] = orig
    
    # Convert neighbors to numpy arrays for faster operations
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    def greedy_cover(restart_seed=0, initial_codewords=None, symmetry_tie_break=False):
        """Greedy set cover with coverage counting and optional symmetry-guided tie-breaking."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(restart_seed)
        
        # Coverage count: how many codewords cover each word
        coverage = np.zeros(total_words, dtype=np.int32)
        codewords = []
        covered_mask = np.zeros(total_words, dtype=bool)
        
        # If we have initial codewords, add them
        if initial_codewords:
            for cw in initial_codewords:
                if cw not in codewords:
                    codewords.append(cw)
                    for nb in neighbors_np[cw]:
                        coverage[nb] += 1
                        covered_mask[nb] = True
        
        # Uncovered words as a boolean array (True = uncovered)
        uncovered_arr = np.ones(total_words, dtype=bool)
        for cw in codewords:
            uncovered_arr[neighbors_np[cw]] = False
        
        # Precompute all neighbor arrays as a 2D array for vectorized operations
        # neighbors_2d[i] is the list of neighbors of word i
        # For efficient counting, we'll use a different approach
        
        # Greedy: repeatedly pick the codeword that covers the most uncovered words
        while time.time() < deadline:
            # Check if all covered
            if not uncovered_arr.any():
                break
            
            # For each candidate, count how many of its neighbors are uncovered
            # We need to count for each candidate i: sum(uncovered_arr[neighbors_np[i]])
            # This is expensive to do naively. Let's use a smarter approach.
            
            # Instead of checking all 729 candidates each iteration, 
            # we can use the fact that only candidates near uncovered words matter.
            # But for now, let's do a vectorized version using precomputed structure.
            
            # Build a 2D array where neighbor_matrix[i, j] = 1 if j is a neighbor of i
            # This is 729x729 = 531441 entries, which is fine.
            # Actually, let's just do it more efficiently.
            
            # For each candidate, the count is the number of its neighbors that are uncovered.
            # We can compute this as: for each candidate i, count = sum(uncovered_arr[neighbors_np[i]])
            # To vectorize, we need a 2D boolean matrix.
            
            # Let's precompute a 2D array: is_neighbor[i, j] = True if j is in neighbors[i]
            # Then count[i] = sum(is_neighbor[i, :] & uncovered_arr)
            # This is a matrix-vector product essentially.
            
            # Precompute neighbor matrix once
            if not hasattr(greedy_cover, '_neighbor_matrix'):
                nm = np.zeros((total_words, total_words), dtype=bool)
                for i in range(total_words):
                    nm[i, neighbors_np[i]] = True
                greedy_cover._neighbor_matrix = nm
            
            nm = greedy_cover._neighbor_matrix
            
            # Count uncovered neighbors for each candidate
            # count[i] = sum(nm[i, :] & uncovered_arr)
            counts = nm.dot(uncovered_arr.astype(np.int32))
            
            # Find the candidate with max count
            best_count = counts.max()
            if best_count == 0:
                break
            
            best_cw = int(counts.argmax())
            
            # Add best_cw
            codewords.append(best_cw)
            uncovered_arr[neighbors_np[best_cw]] = False
        
        return codewords
    
    def remove_redundant(codewords):
        """Remove redundant codewords iteratively using coverage counts."""
        if time.time() > deadline:
            return codewords
        
        if not codewords:
            return codewords
        
        # Build coverage count: for each word, how many codewords cover it
        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for nb in neighbors_np[cw]:
                coverage[nb] += 1
        
        # Use the precomputed neighbor matrix for faster checks
        if not hasattr(greedy_cover, '_neighbor_matrix'):
            nm = np.zeros((total_words, total_words), dtype=bool)
            for i in range(total_words):
                nm[i, neighbors_np[i]] = True
            greedy_cover._neighbor_matrix = nm
        
        nm = greedy_cover._neighbor_matrix
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                
                cw = codewords[i]
                
                # Check if all words in ball(cw) are covered by at least 2 codewords
                # If so, cw is redundant
                # coverage[neighbors_np[cw]] gives coverage for all words cw covers
                # If min coverage >= 2, then cw is redundant
                if coverage[neighbors_np[cw]].min() >= 2:
                    # Remove cw and update coverage
                    coverage[neighbors_np[cw]] -= 1
                    codewords.pop(i)
                    changed = True
                    break
        
        return codewords
    
    def local_search(codewords):
        """Try to swap codewords to improve coverage or reduce count."""
        if time.time() > deadline:
            return codewords
        
        # Try removing each codeword and see if we can re-cover with fewer
        # This is essentially the same as remove_redundant but more aggressive
        
        # Try: for each pair (i, j), see if removing i and adding some k reduces count
        # This is expensive. Instead, just do multiple passes of redundancy removal.
        
        return remove_redundant(codewords)
    
    # Strategy:
    # 1. Start with the tetracode-based construction (81 words)
    # 2. Aggressively remove redundant codewords
    # 3. Try multiple random restarts of greedy to find better solutions
    # 4. Combine best results
    
    best_result = None
    best_count = float('inf')
    
    # First, try the tetracode construction with redundancy removal
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for tail_a in range(3):
        for tail_b in range(3):
            for h in tetracode:
                word = h + [tail_a, tail_b]
                idx = word_to_idx(word, q, n)
                initial_codewords.append(idx)
    
    # Remove redundant from initial
    reduced = remove_redundant(initial_codewords.copy())
    if len(reduced) < best_count:
        best_count = len(reduced)
        best_result = reduced
    
    # Try greedy from scratch with different seeds
    # The vectorized greedy is much faster, so we can do more restarts
    for seed in range(50):
        if time.time() > deadline:
            break
        
        result = greedy_cover(restart_seed=seed, initial_codewords=None, symmetry_tie_break=False)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Try hybrid: start with partial tetracode, then greedy
    for seed in range(20):
        if time.time() > deadline:
            break
        
        # Random subset of tetracode as initial
        rng = random.Random(seed)
        initial = rng.sample(initial_codewords, k=min(len(initial_codewords), rng.randint(30, 60)))
        result = greedy_cover(restart_seed=seed, initial_codewords=initial, symmetry_tie_break=False)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Try additional greedy restarts with different initial random seeds
    for seed in range(50, 100):
        if time.time() > deadline:
            break
        
        result = greedy_cover(restart_seed=seed, initial_codewords=None, symmetry_tie_break=False)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Final local search on best result
    if best_result is not None:
        best_result = local_search(best_result)
    
    # Convert back to lists
    if best_result is None:
        # Fallback: return the tetracode construction
        return [idx_to_word(idx, q, n) for idx in initial_codewords]
    
    return [idx_to_word(idx, q, n) for idx in best_result]
# EVOLVE-BLOCK-END
