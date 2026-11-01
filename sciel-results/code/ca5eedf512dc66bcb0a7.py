"""Combined program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate all words in the space
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total_words = len(all_words)
    
    # If the space is too large for full greedy + post-processing, we might need to restrict, 
    # but for n=7, q=3, 2187 words is small enough.
    
    # Precompute coverage masks for all candidates
    # coverage[i] is a boolean array of length total_words, True if all_words[j] is within radius of all_words[i]
    # We can compute this efficiently using broadcasting or loops.
    # Distance calculation: sum(all_words[:, None, :] != all_words[None, :, :], axis=2) <= radius
    
    # For n=7, 2187x2187 matrix of distances is ~4.7M entries. 
    # Computing this in Python loop is slow. Let's use NumPy broadcasting.
    # all_words shape: (N, n)
    # diff shape: (N, N, n)
    # This might be memory heavy: 2187 * 2187 * 7 * 1 byte ~ 33 MB. Acceptable.
    
    # However, creating the full distance matrix might take a few seconds.
    # Let's check time.
    
    # Alternative: Loop over candidates and compute coverage.
    # For each candidate, compute distance to all words.
    # 2187 candidates * 2187 words * 7 comparisons = ~33M ops. In NumPy vectorized per candidate, it's fast.
    
    coverage = np.zeros((total_words, total_words), dtype=bool)
    
    # Vectorized distance calculation for all pairs is best if memory allows.
    # Let's try the full matrix approach first, it's often fastest for small N.
    # If it fails or is too slow, we fallback to loop.
    
    # Create indices for broadcasting
    # dist[i, j] = sum(all_words[i] != all_words[j])
    
    # To avoid huge intermediate arrays if N was larger, we chunk.
    # But for N=2187, it's fine.
    
    # all_words_expanded_1: (N, 1, n)
    # all_words_expanded_2: (1, N, n)
    # diff: (N, N, n)
    
    # Let's do it in chunks to be safe with memory and cache
    chunk_size = 500
    for i_start in range(0, total_words, chunk_size):
        i_end = min(i_start + chunk_size, total_words)
        # all_words[i_start:i_end] shape (C, n)
        # all_words shape (N, n)
        
        # Compute Hamming distances between chunk and all words
        # diff shape (C, N, n)
        diff = all_words[i_start:i_end, None, :] != all_words[None, :, :]
        dists = np.sum(diff, axis=2)
        coverage[i_start:i_end, :] = dists <= radius

    # Greedy Set Cover
    remaining = np.ones(total_words, dtype=bool)
    remaining_count = total_words
    selected_indices = []
    
    # To speed up greedy, we can maintain a count of uncovered points per candidate.
    # But updating it is expensive.
    # Standard greedy: at each step, find argmax of sum(remaining & coverage[i])
    
    # Optimization: If we have a list of candidates, we can pre-filter or use a priority queue.
    # For 2187 candidates, scanning all of them each step is O(N * N) ~ 4.7M ops per step.
    # If we select ~186 codewords, total ops ~ 186 * 4.7M ~ 870M. This might be too slow in Python/NumPy loop overhead.
    
    # Better Greedy:
    # 1. Pick the candidate that covers the most uncovered points.
    # 2. Mark those points as covered.
    # 3. Repeat.
    
    # We can optimize the "find best candidate" step.
    # Instead of scanning all 2187 candidates every time, we can just do it. 
    # NumPy operations are C-speed. 
    # remaining & coverage[i] -> sum.
    # We can vectorize the "count uncovered for all candidates" step?
    # counts = np.sum(remaining[None, :] & coverage, axis=1)
    # This creates a (N, N) boolean matrix and sums it.
    # 2187 * 2187 * 186 steps.
    # 2187*2187 ~ 4.7M booleans. Summing 4.7M booleans is fast (~1ms?).
    # 186 * 1ms = 0.18s. This is very fast!
    
    # So, we can just do:
    # counts = np.sum(remaining[None, :] & coverage, axis=1)
    # best_idx = np.argmax(counts)
    
    # Let's implement this vectorized greedy.
    
    while remaining_count > 0:
        if time.time() > deadline - 10.0:
            break
            
        # Calculate coverage count for all candidates against remaining words
        # remaining shape (N,)
        # coverage shape (N, N)
        # We want for each candidate i: sum(remaining & coverage[i])
        # This is equivalent to: coverage @ remaining (if coverage was 0/1 and we summed)
        # But coverage is boolean. 
        # np.dot(coverage, remaining.astype(np.int8)) would work and be BLAS accelerated.
        
        counts = coverage.dot(remaining.astype(np.int8))
        
        # Find candidate with max count
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]
        
        if best_count == 0:
            # No candidate covers any remaining word? Should not happen if coverage is complete.
            break
            
        selected_indices.append(best_idx)
        
        # Update remaining
        remaining &= ~coverage[best_idx]
        remaining_count = np.count_nonzero(remaining)
        
        # Optional: If remaining_count is small, we can switch to a different strategy
        # or just continue.
        
    # Convert indices to codewords
    selected_codewords = [all_words[i].tolist() for i in selected_indices]
    
    # Post-processing: Remove redundant codewords
    # A codeword is redundant if all words it covers are also covered by other selected codewords.
    # We can check this by calculating the coverage count of each word by the *selected* codewords.
    # If a word is covered by >= 2 selected codewords, it's "safe".
    # If a word is covered by exactly 1 selected codeword, that codeword is "essential" for that word.
    # A codeword can be removed if it does NOT cover any "essential" word (i.e., a word it is the unique cover for).
    
    if time.time() < deadline - 5.0:
        # Build a matrix of which selected codewords cover which words
        # selected_coverage shape (K, N) where K is number of selected
        K = len(selected_indices)
        if K > 0:
            sel_cov = coverage[selected_indices, :]
            
            # Count how many selected codewords cover each word
            cover_counts = np.sum(sel_cov, axis=0)
            
            # Words with cover_count == 1 are "essential" for the unique codeword covering them.
            # We need to identify which codeword is essential for which word.
            
            # Iterate to remove redundant codewords
            # We maintain a list of active indices
            active = list(range(K))
            
            # To speed up, we can pre-compute for each selected codeword, the set of words it is the *unique* cover for.
            # unique_cover_words[i] = set of words j where sel_cov[i, j] is True and cover_counts[j] == 1
            
            # However, cover_counts changes as we remove codewords.
            # So we must re-evaluate or do it smartly.
            
            # Heuristic:
            # 1. Calculate initial cover_counts.
            # 2. Identify "critical" codewords: those that are the unique cover for at least one word.
            # 3. Try to remove non-critical codewords.
            # 4. After removal, recalculate cover_counts and repeat.
            
            # Simpler approach:
            # Loop through selected codewords. For each, check if it can be removed.
            # To check if codeword i can be removed:
            #   Let S be the set of words covered by codeword i.
            #   For all w in S, check if there exists another active codeword j != i that covers w.
            #   If yes for all w, i can be removed.
            
            # This is O(K^2 * N) in worst case if done naively.
            # K ~ 186, N ~ 2187. 186^2 * 2187 ~ 75M ops. Might be slow in Python.
            
            # Vectorized check:
            # For a candidate removal index r:
            #   words_covered_by_r = sel_cov[r]
            #   We need to check if for all w where words_covered_by_r[w] is True,
            #   there is some other active codeword j covering w.
            #   Let other_cov = sum of sel_cov[active \ {r}]
            #   If sum(other_cov[w]) > 0 for all w in words_covered_by_r, then r is redundant.
            
            # We can optimize by only checking codewords that are "potentially" redundant.
            # Initially, codewords that don't cover any "unique" word are good candidates.
            
            # Let's implement a simple iterative removal.
            # To avoid O(K^2) full scans, we can prioritize.
            
            # Actually, since K is small (~186), a simple loop with vectorized inner check is fine.
            # Inner check: 
            #   mask = sel_cov[r]
            #   others = np.sum(sel_cov[[j for j in active if j != r]], axis=0)
            #   if np.all(others[mask] > 0): remove r
            
            # This is still O(K^2 * N) if we sum others every time.
            # We can maintain a global cover_count array.
            # When we remove a codeword, we decrement cover_count for the words it covered.
            
            active_set = set(selected_indices) # Indices into original all_words, but we work with local indices 0..K-1
            # Let's stick to local indices 0..K-1 for sel_cov
            
            local_cover_counts = np.sum(sel_cov, axis=0)
            
            # We want to remove codewords that don't protect any word with count 1.
            # A codeword i protects word w if sel_cov[i, w] is True and local_cover_counts[w] == 1.
            # If codeword i protects no word, it is redundant.
            
            # We can compute "protection" for all codewords at once?
            # protect[i] = any( sel_cov[i, w] and local_cover_counts[w] == 1 )
            
            # Let's iterate.
            removed = [False] * K
            
            # To be efficient, we can just loop through K times, and in each pass, try to remove one.
            # Or, we can identify all currently redundant ones and remove them, then recalculate.
            
            for _ in range(K):
                if time.time() > deadline - 2.0:
                    break
                
                # Find all codewords that are currently redundant
                # A codeword i is redundant if for all w it covers, local_cover_counts[w] > 1.
                # This is equivalent to: not any( sel_cov[i, w] and local_cover_counts[w] == 1 )
                
                # Let's check each active codeword
                to_remove = []
                for i in range(K):
                    if removed[i]:
                        continue
                    # Check if i covers any word with count 1
                    # words_covered = sel_cov[i]
                    # unique_words = words_covered & (local_cover_counts == 1)
                    # if np.any(unique_words):
                    #     continue # Essential
                    # else:
                    #     to_remove.append(i)
                    
                    # Vectorized check for i
                    if np.any(sel_cov[i] & (local_cover_counts == 1)):
                        continue
                    to_remove.append(i)
                
                if not to_remove:
                    break
                
                # Remove all identified redundant codewords
                for i in to_remove:
                    removed[i] = True
                    # Update local_cover_counts
                    local_cover_counts -= sel_cov[i]
                    
                # If we removed any, we might have made other codewords redundant.
                # The loop continues.
                
            # Final list of codewords
            final_indices = [i for i in range(K) if not removed[i]]
            selected_codewords = [all_words[selected_indices[i]].tolist() for i in final_indices]
            
    return selected_codewords
# EVOLVE-BLOCK-END
