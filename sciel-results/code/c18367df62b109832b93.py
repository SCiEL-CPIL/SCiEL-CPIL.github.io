import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for 150s limit
    
    # For the specific task q=3, n=7, radius=1
    # Total words = 3^7 = 2187
    # This is small enough to handle with bitsets or simple numpy ops efficiently.
    
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    N = len(all_words)
    
    # Precompute the covering sets for each word
    # cover_mask[i] is a boolean array of length N where True if word j is within radius of word i
    # However, storing N x N boolean matrix for 2187x2187 is ~4.7 MB, which is fine.
    # But we can be smarter. We just need to know which words are covered by the current set.
    
    # Let's use a bitset approach for speed in the greedy phase.
    # Python integers can be used as bitsets.
    
    # Map word index to bitset of words it covers
    # Actually, let's just use numpy arrays of indices for simplicity and speed in this small domain.
    
    # Precompute neighbors for each word
    # neighbors[i] = list of indices of words within Hamming distance <= radius
    neighbors = []
    # For radius 1, a word covers itself and words with 1 differing coordinate.
    # Number of neighbors = 1 + n*(q-1) = 1 + 7*2 = 15 for q=3, n=1.
    
    # To speed up, we can precompute the list of neighbor indices for each of the 2187 words.
    # This takes a bit of time but is done once.
    
    # Alternative: On-the-fly generation is slow. Precomputation is better.
    
    # Let's precompute the "coverage" as a list of lists of indices.
    # Or better, a boolean matrix C where C[i,j] = 1 if word j is covered by word i.
    # Then union of coverages is sum > 0.
    
    # Given N=2187, C is 2187x2187. 
    # Computing C:
    # For each i, for each j, check dist. O(N^2 * n) = 2187^2 * 7 ~ 33 million ops. 
    # This might take a few seconds in pure Python, but numpy can vectorize.
    
    # Vectorized distance computation:
    # diff = all_words[:, np.newaxis, :] - all_words[np.newaxis, :, :]
    # dist = np.sum(diff != 0, axis=2)
    # This creates a 2187 x 2187 x 7 array. 2187^2 * 7 * 1 byte (uint8) ~ 33 MB.
    # This is feasible.
    
    diff = all_words[:, np.newaxis, :] - all_words[np.newaxis, :, :]
    dist_matrix = np.sum(diff != 0, axis=2)
    coverage_matrix = (dist_matrix <= radius).astype(np.uint8)
    
    # Greedy Set Cover
    covered = np.zeros(N, dtype=bool)
    code_indices = []
    remaining = N
    
    while remaining > 0 and time.time() < deadline - 10.0:
        # Find word that covers the most uncovered words
        # new_covers = coverage_matrix[:, ~covered].sum(axis=1)
        # This is O(N^2) per step. N=2187, steps ~ 150-200.
        # 200 * 2187^2 ~ 950 million ops. Might be too slow in Python/NumPy loop.
        
        # Optimization: Keep track of uncovered count per word?
        # Or just use the matrix and mask.
        
        # Let's try to optimize the selection.
        # We can maintain an array `uncovered_counts` which is the number of uncovered words covered by each candidate.
        # When we pick a word, we decrement the counts for all words that it covers.
        
        # Let's re-initialize this approach.
        # uncovered_counts[i] = number of words j such that coverage_matrix[i,j]==1 and covered[j]==False
        
        uncovered_counts = coverage_matrix.sum(axis=1).astype(np.int32)
        
        while remaining > 0 and time.time() < deadline - 10.0:
            # Pick argmax
            best_idx = np.argmax(uncovered_counts)
            if uncovered_counts[best_idx] == 0:
                break # Should not happen if remaining > 0 and coverage is perfect, but safety
            
            code_indices.append(best_idx)
            
            # Update covered
            newly_covered = coverage_matrix[best_idx] & ~covered
            if newly_covered.any():
                remaining -= np.sum(newly_covered)
                covered |= newly_covered
            
            # Update uncovered_counts for all other candidates
            # For each candidate i, decrement by number of newly covered words it also covers.
            # new_covers_by_i = coverage_matrix[:, newly_covered_indices].sum(axis=1)
            # This is still O(N * size_of_newly_covered).
            
            newly_covered_indices = np.where(newly_covered)[0]
            if len(newly_covered_indices) > 0:
                # Count how many of these newly covered words are covered by each candidate i
                # coverage_matrix[:, newly_covered_indices] is N x k
                # Sum along axis 1
                decrement = coverage_matrix[:, newly_covered_indices].sum(axis=1)
                uncovered_counts -= decrement
                # Ensure no negative counts (due to double counting? No, newly_covered are disjoint from previous covered)
                uncovered_counts[uncovered_counts < 0] = 0
        
        if remaining > 0:
            # If greedy got stuck or time ran out, we might need to handle the rest, 
            # but for a covering code, greedy usually finishes or gets very close.
            # If time is tight, break.
            pass

    # Local Search (Pairwise Replacement)
    # Try to replace pairs of codewords with one, or replace one with another if it improves coverage.
    # Actually, standard local search for set cover: 
    # 1. Try to remove a codeword if its coverage is redundant.
    # 2. Try to replace two codewords with one different codeword if it reduces size? No, size is fixed in replacement.
    # 3. Try to replace one codeword with another to allow removal of another?
    
    # Simpler Local Search:
    # Iterate through current code.
    # For each word w in code:
    #    Check if w is redundant (all its neighbors are covered by others). If so, remove it.
    #    Check if replacing w with w' allows removal of another word? (More complex)
    
    # Let's do a simple "drop if redundant" pass first, then a "swap" pass.
    
    code_set = set(code_indices)
    
    # Pass 1: Drop redundant
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for w in list(code_set):
            # Check if all neighbors of w are covered by code_set - {w}
            # neighbors_of_w = np.where(coverage_matrix[w])[0]
            # covered_by_others = np.zeros(N, dtype=bool)
            # for v in code_set:
            #    if v != w:
            #        covered_by_others |= coverage_matrix[v]
            # if np.all(covered_by_others[neighbors_of_w]):
            #    code_set.remove(w)
            #    changed = True
            #    break # Restart loop to be safe, or continue
            
            # Vectorized check:
            # Mask out w
            mask_others = np.ones(N, dtype=bool)
            mask_others[w] = False
            others_in_code = [v for v in code_set if v != w]
            
            if not others_in_code:
                continue
                
            # Compute union of coverage of others_in_code
            # This is expensive if done naively.
            # Let's precompute union coverage of current code set?
            # current_coverage = np.zeros(N, dtype=bool)
            # for v in code_set:
            #     current_coverage |= coverage_matrix[v]
            
            # If we remove w, new_coverage = current_coverage & ~coverage_matrix[w]
            # Check if new_coverage[neighbors_of_w] are all True.
            
            # We need current_coverage.
            # Let's maintain it.
            
            # Actually, recomputing current_coverage every time is O(|Code| * N).
            # |Code| ~ 150, N ~ 2200. 150 * 2200 = 330k ops. Very fast.
            
            current_coverage = np.zeros(N, dtype=bool)
            for v in code_set:
                current_coverage |= coverage_matrix[v]
                
            if w not in code_set: continue
            
            # Coverage without w
            coverage_without_w = current_coverage & ~coverage_matrix[w]
            
            # Neighbors of w
            neighbors_of_w = np.where(coverage_matrix[w])[0]
            
            if np.all(coverage_without_w[neighbors_of_w]):
                code_set.remove(w)
                changed = True
                break # Restart to update coverage efficiently? Or just continue.
        
    # Pass 2: Swap
    # Try replacing one word with another to enable a drop?
    # Or just try replacing one word with a different one that covers more?
    # Greedy already maximized marginal gain.
    # A common improvement is to try to replace a pair of codewords with a single codeword if possible?
    # i.e., is there a word w such that coverage(w) covers all words covered by w1 and w2?
    # Unlikely for radius 1.
    
    # Let's try a simple random restart or perturbation if time allows.
    # But for now, the greedy + drop is likely to get close to the optimum.
    # Known best is 186. Greedy often gets within 10-20%.
    # 186 * 1.1 = 204.
    
    # Let's add a "local search" that tries to swap a codeword with a random other to see if it allows a drop.
    
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Current coverage
            current_coverage = np.zeros(N, dtype=bool)
            for v in code_set:
                current_coverage |= coverage_matrix[v]
            
            # For each w in code_set, try replacing it with w'
            # If coverage with w' (and rest of set) is still full, and we can drop another word?
            # Simpler: If we replace w with w', does the new set still cover everything?
            # And is the size the same? Yes.
            # But we want to minimize size.
            # So: Replace w with w'. Then run "drop redundant". If size decreases, accept.
            
            # This is expensive. Let's limit the number of swaps.
            
            # Alternative: Just return the current set.
            # The greedy set cover is a standard approximation.
            
            # Let's try a few random swaps to see if they help drop.
            # Pick a random w in code_set.
            # Pick a random w' not in code_set.
            # New set = (code_set - {w}) U {w'}
            # Check if new set covers all.
            # If yes, try to drop redundant from new set.
            # If size < old size, accept.
            
            # This is a hill-climbing step.
            
            # Given time constraints, let's stick to the greedy + drop.
            # It's robust and fast.
            
            pass
            
    return [list(all_words[i]) for i in code_indices]
