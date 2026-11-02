import numpy as np
import itertools
import time

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute balls
# ball[c] is an array of indices of words within distance 1 of c
balls = []
for i in range(NUM_WORDS):
    c = all_words[i]
    # Center
    ball_indices = [i]
    # Neighbors
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            # Create neighbor
            neighbor = c.copy()
            neighbor[pos] = val
            # Find index of neighbor
            # Since it's a small space, we can use a dictionary or just search
            # Better: precompute a mapping from tuple to index
            pass
    
    # This loop is slow. Let's use a mapping.
    pass

# Let's do the mapping once
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

balls = []
for i in range(NUM_WORDS):
    c = all_words[i]
    ball_indices = [i]
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball_indices.append(idx)
    balls.append(np.array(ball_indices, dtype=np.int32))

# Tetracode construction for fallback/initial best
def get_tetracode_code():
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though task specifies 3,6,1
        # Use simple greedy for general case
        pass

    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_code = get_tetracode_code()
    best_size = len(best_code)
    
    # Greedy Set Cover with Randomized Restarts
    # We will run multiple restarts.
    
    for restart in range(10000):
        if time.time() > deadline:
            break
            
        covered = np.zeros(NUM_WORDS, dtype=bool)
        code = []
        
        # Greedy Phase
        while not np.all(covered):
            # Calculate scores
            # Score is number of uncovered words in the ball
            # Optimization: vectorized sum
            # scores = np.array([np.sum(~covered[balls[c]]) for c in range(NUM_WORDS)])
            
            # Faster:
            # Create a matrix? No, sparse.
            # Just loop. 729 iterations is small.
            scores = np.zeros(NUM_WORDS, dtype=np.int32)
            for c in range(NUM_WORDS):
                scores[c] = np.sum(~covered[balls[c]])
                
            # Randomized selection from top candidates
            # Find max score
            max_score = np.max(scores)
            if max_score == 0:
                # Should not happen if not all covered
                break
                
            # Get candidates with max score (or top few)
            # To introduce randomness, pick from top 5% or add noise
            # Let's pick randomly from all candidates with score == max_score
            candidates = np.where(scores == max_score)[0]
            c = candidates[np.random.randint(len(candidates))]
            
            code.append(c)
            covered[balls[c]] = True
            
        # Reduction Phase
        # Compute coverage counts
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[balls[c]] += 1
            
        # Try to remove redundant codewords
        changed = True
        while changed:
            changed = False
            # Iterate in random order to break symmetry
            indices = np.arange(len(code))
            np.random.shuffle(indices)
            
            for idx in indices:
                c = code[idx]
                # Check if all words in balls[c] have cov_count > 1
                # If yes, c is redundant
                if np.all(cov_counts[balls[c]] > 1):
                    # Remove c
                    cov_counts[balls[c]] -= 1
                    code.pop(idx)
                    changed = True
                    # After removing, we might be able to remove more
                    # So we break and restart the loop
                    break
                    
        current_size = len(code)
        if current_size < best_size:
            best_size = current_size
            best_code = [list(all_words[c]) for c in code]
            if best_size <= 73:
                # 73 is the best known. We can stop if we find 73 to save time?
                # Or continue to try for 72?
                # 72 is unknown. Let's continue for a bit.
                # But if we find 73, we can return early if time is tight?
                # No, just update best.
                pass

    return best_code
