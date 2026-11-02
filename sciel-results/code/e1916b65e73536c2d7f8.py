import numpy as np
import itertools
import time

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls: ball_matrix[c, w] = 1 if w is within distance 1 of c
# We use a sparse-like approach: for each c, store the list of indices in its ball
# For vectorized scoring, we build a 0/1 matrix of shape (NUM_WORDS, NUM_WORDS)
# This is 729x729 = ~531k entries, which is manageable.

ball_matrix = np.zeros((NUM_WORDS, NUM_WORDS), dtype=np.int8)
ball_indices = []  # list of arrays, each containing indices in the ball

for i in range(NUM_WORDS):
    c = all_words[i]
    ball_indices.append([i])
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball_indices.append(idx) if False else None
            ball_indices[-1].append(idx)
            ball_matrix[i, idx] = 1

# Convert to list of arrays for fast indexing
ball_indices = [np.array(bi, dtype=np.int32) for bi in ball_indices]

def get_tetracode_code():
    """Fallback construction yielding 81 codewords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        # Simple greedy for general case
        if q is None or n is None or radius is None:
            return []
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        bm = np.zeros((nw, nw), dtype=np.int8)
        for i in range(nw):
            c = all_w[i]
            bm[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    bm[i, wti[tuple(nb)]] = 1
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            scores = bm.dot(~covered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[bm[c] == 1] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (150s limit)
    
    best_code = get_tetracode_code()
    best_size = len(best_code)
    
    # Vectorized greedy set cover with randomization
    # ball_matrix is (NUM_WORDS, NUM_WORDS), int8
    # covered is (NUM_WORDS,) bool
    # score[c] = sum of uncovered in ball[c] = ball_matrix[c].dot(~covered)
    
    for restart in range(100000):
        if time.time() > deadline:
            break
            
        covered = np.zeros(NUM_WORDS, dtype=bool)
        code = []
        
        # Greedy Phase
        while not np.all(covered):
            # Vectorized score calculation
            scores = ball_matrix.dot(~covered)
            max_score = np.max(scores)
            if max_score == 0:
                break
                
            # Randomized selection from top candidates
            # Pick from candidates with score >= max_score * 0.9 to add diversity
            threshold = max_score * 0.85
            candidates = np.where(scores >= threshold)[0]
            c = candidates[np.random.randint(len(candidates))]
            
            code.append(c)
            covered[ball_indices[c]] = True
            
        # Reduction Phase
        # Compute coverage counts
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_indices[c]] += 1
            
        # Try to remove redundant codewords
        changed = True
        while changed:
            changed = False
            indices = np.arange(len(code))
            np.random.shuffle(indices)
            
            for idx in indices:
                c = code[idx]
                if np.all(cov_counts[ball_indices[c]] > 1):
                    cov_counts[ball_indices[c]] -= 1
                    code.pop(idx)
                    changed = True
                    break
                    
        # Local Search Phase: Try 1-for-1 swaps to enable further reduction
        # We look for a codeword c_i that covers some unique words.
        # We try to replace it with c_j that covers those unique words,
        # which might allow other codewords to be removed.
        
        improved = True
        max_swap_iterations = 20
        swap_iter = 0
        while improved and swap_iter < max_swap_iterations:
            improved = False
            swap_iter += 1
            
            if not code:
                break
                
            # Recalculate coverage counts
            cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
            for c in code:
                cov_counts[ball_indices[c]] += 1
            
            # Randomly select a codeword to potentially swap
            idx_to_swap = np.random.randint(len(code))
            c_i = code[idx_to_swap]
            
            # Find words covered only by c_i
            ball_i = ball_indices[c_i]
            unique_words = ball_i[cov_counts[ball_i] == 1]
            
            if len(unique_words) == 0:
                continue
                
            # Pick a random unique word
            target_word = unique_words[np.random.randint(len(unique_words))]
            
            # Find candidates c_j that cover target_word
            in_code = np.zeros(NUM_WORDS, dtype=bool)
            for c in code:
                in_code[c] = True
            
            candidates = np.where(ball_matrix[:, target_word] == 1)[0]
            candidates = candidates[~in_code[candidates]]
            
            if len(candidates) == 0:
                continue
                
            # Try a few random candidates
            num_tries = min(5, len(candidates))
            tried = np.random.choice(candidates, size=num_tries, replace=False)
            
            for c_j in tried:
                # Check if c_j covers ALL unique words of c_i
                covers_all_unique = True
                for w in unique_words:
                    if ball_matrix[c_j, w] == 0:
                        covers_all_unique = False
                        break
                        
                if not covers_all_unique:
                    continue
                
                # Perform swap and reduction
                test_code = code.copy()
                test_code[idx_to_swap] = c_j
                
                # Compute coverage for test_code
                test_cov = np.zeros(NUM_WORDS, dtype=np.int32)
                for c in test_code:
                    test_cov[ball_indices[c]] += 1
                    
                # Reduce test_code
                t_changed = True
                while t_changed:
                    t_changed = False
                    t_indices = np.arange(len(test_code))
                    np.random.shuffle(t_indices)
                    for t_idx in t_indices:
                        tc = test_code[t_idx]
                        if np.all(test_cov[ball_indices[tc]] > 1):
                            test_cov[ball_indices[tc]] -= 1
                            test_code.pop(t_idx)
                            t_changed = True
                            break
                            
                if len(test_code) < len(code):
                    code = test_code
                    improved = True
                    break # Restart the swap search with the new code

        current_size = len(code)
        if current_size < best_size:
            best_size = current_size
            best_code = [list(all_words[c]) for c in code]
            if best_size <= 73:
                # We found the best known. Try to improve further.
                pass

    return best_code
