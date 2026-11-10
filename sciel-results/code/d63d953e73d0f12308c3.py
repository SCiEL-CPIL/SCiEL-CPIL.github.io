import itertools
import numpy as np

def construct(q, n, radius):
    # Generate all possible words
    all_words = np.array(list(itertools.product(range(q), repeat=n)))
    num_words = len(all_words)
    
    # Pre-compute the ball for each word
    # is_in_ball[i, j] is True if word j is within distance 'radius' of word i
    
    # Vectorized distance calculation
    a = all_words[:, None, :]
    b = all_words[None, :, :]
    dist = np.sum(a != b, axis=2)
    
    is_in_ball = dist <= radius
    
    covered = np.zeros(num_words, dtype=bool)
    code = []
    
    # List of all candidate indices
    candidates = list(range(num_words))
    
    while covered.sum() < num_words:
        if not candidates:
            break
            
        # Filter candidates to only those that are currently uncovered
        # This is a heuristic: it is generally better to pick an uncovered word
        # as the next center because covered words are "already dominated" by existing centers
        # and their marginal gain is often lower than that of a well-chosen uncovered word.
        active_candidates = [c for c in candidates if not covered[c]]
        
        if not active_candidates:
            # Fallback: if all candidates are covered but we still have uncovered words (should not happen if is_in_ball is symmetric and complete)
            active_candidates = candidates
            
        # Calculate gain for active candidates
        # We need to sum is_in_ball[c, j] for j where covered[j] is False
        # is_in_ball shape: (N, N)
        # active_candidates is a list of indices
        # Extract rows for active candidates
        candidate_rows = is_in_ball[active_candidates] # (K, N)
        
        # Mask with uncovered
        # uncovered is (N,)
        # We want sum of True values in candidate_rows where uncovered is True
        gains = np.sum(candidate_rows & uncovered_mask, axis=1) if 'uncovered_mask' in locals() else np.sum(candidate_rows & (~covered)[None, :], axis=1)
        
        # Find the best candidate
        best_local_idx = np.argmax(gains)
        best_gain = gains[best_local_idx]
        
        if best_gain == 0:
            # No candidate covers any new word? This implies an error in logic or data
            # But if covered.sum() < num_words, there must be some uncovered word.
            # If all active candidates have 0 gain, it means their balls are fully covered.
            # This can happen if we restricted to uncovered candidates but their balls don't reach new territory?
            # Actually, if a word u is uncovered, it is in B(u). So B(u) contains u.
            # So gain for u is at least 1.
            # So best_gain should never be 0 if active_candidates is not empty and contains uncovered words.
            break
            
        best_global_idx = active_candidates[best_local_idx]
        
        code.append(all_words[best_global_idx].tolist())
        
        # Update covered
        covered |= is_in_ball[best_global_idx]
        
        # Update candidates: remove the selected one? 
        # We don't strictly need to remove it, but we filter by 'not covered' in the next iteration.
        # Since best_global_idx is now covered, it will be filtered out next time.
        # We can just keep the full list and filter each time, or maintain a set.
        # Filtering a list of 729 is cheap.
        
    return code
