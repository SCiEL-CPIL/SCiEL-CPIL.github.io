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

    # Generate candidate codewords using tetracode direct-sum construction
    # This gives 9 * 3^(n-4) candidates, much fewer than all_words
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        candidates = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        candidates = np.array(candidates, dtype=np.int8)
        num_candidates = len(candidates)
    else:
        # Fallback: use all words as candidates
        candidates = all_words
        num_candidates = total_words

    # Precompute coverage matrix: coverage[i, j] = True if candidates[i] covers all_words[j]
    # Use int32 for distance calculation to avoid overflow
    coverage = np.zeros((num_candidates, total_words), dtype=bool)
    
    # Vectorized distance calculation in chunks
    chunk_size = 256
    for i_start in range(0, num_candidates, chunk_size):
        i_end = min(i_start + chunk_size, num_candidates)
        # candidates[i_start:i_end] shape (C, n)
        # all_words shape (N, n)
        diff = candidates[i_start:i_end, None, :].astype(np.int32) != all_words[None, :, :].astype(np.int32)
        dists = np.sum(diff, axis=2)
        coverage[i_start:i_end, :] = dists <= radius

    # Greedy Set Cover using BLAS-accelerated dot product
    remaining = np.ones(total_words, dtype=np.int8)
    remaining_count = total_words
    selected_indices = []
    
    while remaining_count > 0:
        if time.time() > deadline - 10.0:
            break
            
        # Calculate coverage count for all candidates against remaining words
        # coverage shape (M, N), remaining shape (N,)
        # counts[i] = sum of coverage[i, j] where remaining[j] == 1
        counts = coverage.dot(remaining)
        
        # Find candidate with max count
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]
        
        if best_count == 0:
            break
            
        selected_indices.append(best_idx)
        
        # Update remaining
        remaining[coverage[best_idx]] = 0
        remaining_count = np.count_nonzero(remaining)
        
    # Convert indices to codewords
    selected_codewords = [candidates[i].tolist() for i in selected_indices]
    
    # Post-processing: Remove redundant codewords
    if time.time() < deadline - 5.0 and len(selected_indices) > 0:
        K = len(selected_indices)
        sel_cov = coverage[selected_indices, :]
        
        # Count how many selected codewords cover each word
        local_cover_counts = np.sum(sel_cov, axis=0)
        
        # Iteratively remove redundant codewords
        removed = [False] * K
        
        for _ in range(K):
            if time.time() > deadline - 2.0:
                break
                
            to_remove = []
            for i in range(K):
                if removed[i]:
                    continue
                # Check if i covers any word with count 1
                if np.any(sel_cov[i] & (local_cover_counts == 1)):
                    continue
                to_remove.append(i)
                
            if not to_remove:
                break
                
            for i in to_remove:
                removed[i] = True
                local_cover_counts -= sel_cov[i]
                
        # Final list of codewords
        final_indices = [i for i in range(K) if not removed[i]]
        selected_codewords = [candidates[selected_indices[i]].tolist() for i in final_indices]
        
    return selected_codewords
# EVOLVE-BLOCK-END
