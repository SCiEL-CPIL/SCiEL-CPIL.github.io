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

# Precompute ball matrix
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
            ball_indices[-1].append(idx)
            ball_matrix[i, idx] = 1

# Convert to list of arrays for fast indexing
ball_indices = [np.array(bi, dtype=np.int32) for bi in ball_indices]
# Also create boolean masks for faster AND operations if needed, or just use ball_matrix rows
# ball_masks = [row == 1 for row in ball_matrix] -> Memory heavy? 729*729 bool is small.
# But ball_matrix.dot is fast.

def get_tetracode_code():
    """Fallback construction yielding 81 codewords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
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
    deadline = start_time + 110  # 110s safety margin

    best_code = get_tetracode_code()
    best_size = len(best_code)
    
    # Precompute boolean masks for balls for faster unique calculation?
    # ball_bool = ball_matrix == 1
    
    for restart in range(100000):
        if time.time() > deadline:
            break
            
        covered = np.zeros(NUM_WORDS, dtype=bool)
        code = []
        
        # Greedy Phase
        while not np.all(covered):
            scores = ball_matrix.dot(~covered)
            max_score = np.max(scores)
            if max_score == 0:
                break
                
            # Randomized selection
            threshold = max_score * 0.85
            candidates = np.where(scores >= threshold)[0]
            c = candidates[np.random.randint(len(candidates))]
            
            code.append(c)
            covered[ball_indices[c]] = True
            
        # Local Search Phase
        # State: code (list of indices), in_code (bool array), cov_counts (int array)
        in_code = np.zeros(NUM_WORDS, dtype=bool)
        for c in code:
            in_code[c] = True
            
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_indices[c]] += 1
            
        # Iterative improvement
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # 1. Delete redundant words
            # A word c is redundant if no word is covered exclusively by c.
            # i.e., for all w in ball(c), cov_counts[w] > 1.
            redundant_indices = []
            for i, c in enumerate(code):
                # Check if any word in ball(c) has count 1
                # unique_in_c = (cov_counts == 1) & (ball_matrix[c] == 1)
                # if not np.any(unique_in_c): redundant
                if not np.any((cov_counts == 1) & (ball_matrix[c] == 1)):
                    redundant_indices.append(i)
            
            if redundant_indices:
                # Remove all redundant words at once
                # We need to be careful if removing one makes another non-redundant?
                # No, if c is redundant, it doesn't cover any unique word.
                # Removing c decreases counts for words in ball(c).
                # This might make words that were count 2 become count 1.
                # But it cannot make a word that was NOT covered by c become covered by c.
                # So redundancy of other words is not affected in a way that creates new redundancies?
                # Actually, removing c might make some words unique to *other* codewords?
                # No, if w was covered by c and d, and we remove c, w is still covered by d.
                # If w was covered only by c, c was NOT redundant.
                # So removing redundant words is safe and doesn't create new redundancies immediately?
                # Wait. If c1 and c2 are both redundant, we can remove both.
                # If c1 is redundant, it covers no unique words.
                # Removing c1 reduces counts.
                # Could this make c2 redundant? Yes, if c2 was covering a word w that was also covered by c1, and w was the only word c2 covered uniquely?
                # No, if w was covered by c1 and c2, and c1 is redundant, it means w is covered by some other c3?
                # No, "c1 is redundant" means NO word is covered *only* by c1.
                # So every word in ball(c1) is covered by at least one other codeword.
                # So removing c1 does not uncover any word.
                # Thus, the coverage of the remaining code is the same as before.
                # So if c2 was not redundant before, it is not redundant after?
                # Actually, redundancy depends on the *set* of codewords.
                # If we remove redundant words, the coverage is unchanged.
                # So the set of redundant words in the new set is a subset of the old set?
                # Yes. So we can remove all identified redundant words in one go.
                
                # Update code and counts
                # We need to remove them from the list and update counts.
                # Since order doesn't matter for the set, we can just filter.
                new_code = [c for i, c in enumerate(code) if i not in set(redundant_indices)]
                # Update counts
                cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
                for c in new_code:
                    cov_counts[ball_indices[c]] += 1
                code = new_code
                for c in code:
                    in_code[c] = True
                for i in redundant_indices:
                    c = code[i] if i < len(code) else None # This logic is flawed if we modify list while iterating indices.
                    
                # Better way to update in_code:
                in_code = np.zeros(NUM_WORDS, dtype=bool)
                for c in code:
                    in_code[c] = True
                    
                improved = True
                continue # Restart loop to check for more redundancies or swaps

            # 2. Swap phase
            # Try to find a swap that allows future deletions or just improves structure?
            # Standard swap: replace c with c' such that coverage is maintained.
            # This is valid if c' covers all words that were uniquely covered by c.
            
            # Shuffle to try different c's
            indices = np.arange(len(code))
            np.random.shuffle(indices)
            
            swapped = False
            for idx in indices:
                c = code[idx]
                unique_mask = (cov_counts == 1) & (ball_matrix[c] == 1)
                target_count = np.sum(unique_mask)
                
                if target_count == 0:
                    # Should have been caught in delete phase, but just in case
                    continue
                
                # Find c' not in code that covers all unique words
                # scores[c'] = number of unique words covered by c'
                scores = ball_matrix.dot(unique_mask)
                candidates = np.where((scores == target_count) & (~in_code))[0]
                
                if len(candidates) > 0:
                    # Pick a random candidate
                    c_new = candidates[np.random.randint(len(candidates))]
                    
                    # Perform swap
                    # Remove c
                    in_code[c] = False
                    cov_counts[ball_indices[c]] -= 1
                    # Add c_new
                    in_code[c_new] = True
                    cov_counts[ball_indices[c_new]] += 1
                    # Update code list
                    code[idx] = c_new
                    
                    swapped = True
                    break # Restart delete/swap loop
            
            if not swapped:
                break # Local optimum reached

        current_size = len(code)
        if current_size < best_size:
            best_size = current_size
            best_code = [list(all_words[c]) for c in code]
            if best_size <= 73:
                # Best known is 73. If we hit 73, we might try to find 72?
                # But 71 is lower bound. 73 is best known.
                pass

    return best_code
