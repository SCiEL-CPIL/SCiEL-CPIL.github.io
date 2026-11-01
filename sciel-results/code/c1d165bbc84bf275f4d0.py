import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Total number of words
    N = q ** n
    
    # Generate all words and map to indices
    # words[i] is the tuple of digits for index i
    # idx_to_word is a list of lists/tuples
    # word_to_idx is a dict
    
    # For q=3, n=7, N=2187. This is small.
    # We can generate all combinations.
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors (Hamming distance <= radius)
    # balls[i] is a list of indices of words within distance 'radius' of words[i]
    # For radius 1, this is 1 + n*(q-1) neighbors.
    # For q=3, n=7, radius=1: 1 + 7*2 = 15 neighbors? 
    # Wait, Hamming distance 1 means changing exactly one position to a different value.
    # For ternary, each position has 2 other values. So 7*2 = 14 neighbors + self = 15.
    
    # Let's verify the ball size.
    # The problem statement says "differs from some chosen word in at most one position".
    # Yes, radius 1.
    
    # Precompute balls
    # balls will be an array of lists of indices.
    # For efficient overlap calculation, we might want a dense matrix or sparse.
    # Given N=2187, a dense matrix of size N x N is ~4.7M entries. 
    # If we store it as uint8, it's ~4.7MB. This is fine.
    
    # However, we don't need the full N x N overlap matrix for all pairs.
    # We only need `overlap[k, c]` for the greedy step where `c` is the selected codeword.
    # Actually, we can compute the "decrease" on the fly if we store the balls as sets or sorted lists.
    # But vectorized NumPy is faster.
    
    # Let's create a boolean matrix `is_in_ball` of shape (N, N) where is_in_ball[i, j] = 1 if j is in ball(i).
    # This is the same as the adjacency matrix of the covering graph.
    
    is_in_ball = np.zeros((N, N), dtype=np.uint8)
    
    for i in range(N):
        w = words[i]
        # Self
        is_in_ball[i, i] = 1
        # Neighbors
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    # Create new word
                    # Instead of creating tuple and looking up, we can compute index directly?
                    # Or just use the dict.
                    new_w_list = list(w)
                    new_w_list[pos] = new_val
                    new_w = tuple(new_w_list)
                    j = word_to_idx[new_w]
                    is_in_ball[i, j] = 1
        # Check time
        if time.time() > deadline:
             # Fallback to seed if precomputation takes too long (unlikely)
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Greedy Set Cover
    # counts[i] = number of points in ball(i) that are currently uncovered
    # Initially, all points are uncovered, so counts[i] = sum(is_in_ball[i, :]) = ball_size
    # Note: is_in_ball[i, :] has 1s where the ball covers.
    
    counts = is_in_ball.sum(axis=1).astype(np.int32)
    covered = np.zeros(N, dtype=bool)
    
    solution = []
    
    # We need to track which points are covered to stop correctly?
    # Actually, if counts[k] becomes 0 for all k, we are done?
    # No, it's possible that some points are covered, but no *single* ball adds new points?
    # No, if a point is uncovered, it belongs to some balls. Those balls would have count > 0.
    # So if max(counts) == 0, all points are covered.
    
    while True:
        if time.time() > deadline:
            break
        
        max_idx = np.argmax(counts)
        if counts[max_idx] == 0:
            break
            
        solution.append(max_idx)
        
        # Update covered status
        # points_in_ball = np.where(is_in_ball[max_idx, :] == 1)[0]
        # covered[points_in_ball] = True
        
        # Update counts for all other codewords
        # For any codeword k, the number of new points it covers decreases by
        # the number of points in ball(k) that are in ball(max_idx) AND were just covered.
        # Since we mark ALL points in ball(max_idx) as covered, the decrease is
        # exactly the intersection size of ball(k) and ball(max_idx).
        # This is is_in_ball[k, max_idx].
        
        counts -= is_in_ball[:, max_idx]
        
        # Optional: Set counts[max_idx] to 0 to avoid re-picking? 
        # It will be 0 because is_in_ball[max_idx, max_idx] is 1, so it decreased by 1.
        # But wait, if it was 15, it becomes 14? No.
        # The logic: `counts` represents "how many NEW points this codeword covers".
        # When we pick `max_idx`, we cover its ball.
        # For any `k`, the points in `ball(k) ∩ ball(max_idx)` are now covered.
        # So `k` covers `|ball(k) ∩ ball(max_idx)|` fewer new points.
        # This is correct.
        
        # Check if all covered
        # We can track total_covered, or just rely on max(counts)==0.
        # Is it possible that max(counts) > 0 but no progress is made? No.
        
    # Local Reduction
    # Maintain cover_count for the current solution
    # cover_count[w] = number of codewords in solution covering w
    
    if solution:
        cover_count = np.zeros(N, dtype=np.int32)
        for c in solution:
            cover_count[is_in_ball[c, :] == 1] += 1
        
        # Try to remove each codeword
        # We iterate multiple times until no removal is possible
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Iterate in reverse to handle index shifts if we remove, 
            # but we are removing from a list of indices, so we can just filter at the end
            # Or we can mark for removal
            
            removals = []
            for c in solution:
                # Check if removing c leaves any point uncovered
                # i.e., is there a point w in ball(c) such that cover_count[w] == 1?
                # If yes, removing c makes cover_count[w] == 0 -> Invalid.
                # If no, removing c is safe.
                
                # Get points in ball(c)
                # We can use the is_in_ball row
                # points = np.where(is_in_ball[c, :] == 1)[0]
                # if np.any(cover_count[points] == 1):
                #     pass
                # else:
                #     removals.append(c)
                
                # Vectorized check:
                # min_count_in_ball = np.min(cover_count[is_in_ball[c, :] == 1])
                # if min_count_in_ball > 1:
                #     removals.append(c)
                
                # To avoid creating boolean masks repeatedly, we can precompute the list of points for each ball?
                # Or just use the matrix. N=2187, so mask creation is fast.
                
                mask = is_in_ball[c, :] == 1
                if np.min(cover_count[mask]) > 1:
                    removals.append(c)
            
            if removals:
                changed = True
                for c in removals:
                    cover_count[is_in_ball[c, :] == 1] -= 1
                solution = [c for c in solution if c not in removals]
        
        # Final check: ensure validity
        # min(cover_count) should be >= 1
        if np.min(cover_count) < 1:
            # This shouldn't happen if logic is correct, but fallback to seed if it does
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert indices back to words (lists of ints)
    result = [list(words[i]) for i in solution]
    
    return result
