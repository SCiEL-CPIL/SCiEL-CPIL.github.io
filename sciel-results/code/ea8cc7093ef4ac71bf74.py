import itertools
import numpy as np

def construct(q, n, radius):
    # Generate all possible words
    all_words = np.array(list(itertools.product(range(q), repeat=n)))
    num_words = len(all_words)
    
    # Pre-compute the ball for each word (words within Hamming distance 'radius')
    # For efficiency, we create a 2D boolean matrix where is_in_ball[i, j] is True
    # if word j is within distance 'radius' of word i.
    
    # Helper to calculate Hamming distance between two words
    # Since n is small (6) and q is small (3), we can just compute it directly.
    
    is_in_ball = np.zeros((num_words, num_words), dtype=bool)
    
    # We can optimize this by noting that for radius=1, the ball contains the word itself
    # and words that differ in exactly one position.
    # However, for generality and since N=729 is small, we can just compute distances.
    
    # Vectorized distance calculation for all pairs is N^2 * n operations.
    # 729^2 * 6 ~ 3 million operations. This is acceptable.
    
    # all_words shape: (729, 6)
    # To compute pairwise Hamming distance:
    # dist[i, j] = sum(all_words[i, k] != all_words[j, k] for k in range(n))
    
    # Using broadcasting:
    # a = all_words[:, None, :]  # (729, 1, 6)
    # b = all_words[None, :, :]  # (1, 729, 6)
    # dist = np.sum(a != b, axis=2) # (729, 729)
    
    a = all_words[:, None, :]
    b = all_words[None, :, :]
    dist = np.sum(a != b, axis=2)
    
    is_in_ball = dist <= radius
    
    covered = np.zeros(num_words, dtype=bool)
    code = []
    
    while covered.sum() < num_words:
        uncovered = ~covered
        
        # Calculate gain for each candidate
        # is_in_ball is (N, N), uncovered is (N,)
        # We want sum over j (columns) of is_in_ball[i, j] AND uncovered[j]
        # Broadcasting: is_in_ball & uncovered[None, :]
        
        gains = np.sum(is_in_ball & uncovered[None, :], axis=1)
        
        # Pick the candidate with maximum gain
        best_idx = np.argmax(gains)
        
        # If gain is 0, we are stuck (should not happen if covered.sum() < N)
        if gains[best_idx] == 0:
             break
            
        code.append(all_words[best_idx].tolist())
        
        # Update covered
        covered |= is_in_ball[best_idx]
        
    return code
