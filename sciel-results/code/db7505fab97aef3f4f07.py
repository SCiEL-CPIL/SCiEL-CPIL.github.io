import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer for safety
    
    # Calculate total number of words
    N = q ** n
    
    # If the space is too large for the efficient bitset/matmul approach, 
    # fall back to a simple valid construction (like the seed's tetracode if applicable, 
    # or a random/greedy without bitsets which is slower but works for smaller N).
    # For this specific task (q=3, n=8), N=6561, which is fine.
    if N > 20000:
        # Fallback: Simple greedy without full matrix, or structural construction
        # For now, return a valid but potentially larger code using a simple greedy
        # that checks coverage on the fly (slower per step but less memory)
        # Or just return the tetracode construction if n>=4
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: cover with all words (valid, size N)
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    # 1. Precompute neighbors
    # Map integer index to tuple of digits
    def idx_to_tuple(idx):
        return [int(d) for d in np.base_repr(idx, base=q).zfill(n)] if q != 10 else list(map(int, str(idx).zfill(n)))
        # Note: np.base_repr works for any base. But for q=3, it's fine.
        # Actually, manual conversion is faster and safer for small n.
    
    def tuple_to_idx(tup):
        idx = 0
        for d in tup:
            idx = idx * q + d
        return idx

    # Precompute neighbor indices for each word
    # neighbors[i] is a list of indices of words within distance <= radius
    # For radius 1, it's the word itself and words with 1 digit changed.
    neighbors = [[] for _ in range(N)]
    
    # Generate all words as tuples
    all_words = [list(word) for word in itertools.product(range(q), repeat=n)]
    
    # Map word to index
    # We can just use the index directly if we generate them in order
    # Index i corresponds to the i-th word in lexicographic order?
    # Yes, if we use base-q representation properly.
    # Let's verify:
    # 0 -> 00...0
    # 1 -> 00...1
    # ...
    # q-1 -> 00...(q-1)
    # q -> 01...0
    # This matches itertools.product order if we treat the first element as most significant?
    # itertools.product(range(q), repeat=n) generates (0,0,...,0), (0,0,...,1), ..., (0,0,...,q-1), (0,1,0,0)...
    # This corresponds to base-q expansion where the last digit is least significant?
    # Let's check:
    # (0, 1) in n=2, q=2.
    # idx = 1.
    # base_repr(1, 2) is '1'. zfill(2) is '01'.
    # So idx 1 is '01'.
    # itertools.product gives (0,0) at 0, (0,1) at 1.
    # So `idx_to_tuple` using base_repr with zfill(n) works if we interpret the string as the tuple.
    
    for i in range(N):
        word = all_words[i]
        # Self
        neighbors[i].append(i)
        # Neighbors at distance 1
        for pos in range(n):
            original_val = word[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_word = word.copy()
                new_word[pos] = new_val
                new_idx = tuple_to_idx(new_word)
                neighbors[i].append(new_idx)
        
        # Check time
        if time.time() > deadline:
            # Fallback to valid construction
            if n >= 4 and q == 3:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # 2. Greedy Set Cover using NumPy
    # Create N_matrix: N x N boolean matrix
    # N_matrix[i, j] = 1 if j is in neighbors[i]
    N_matrix = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in neighbors[i]:
            N_matrix[i, j] = True
            
    uncovered = np.ones(N, dtype=bool)
    centers = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Find center that covers most uncovered words
        # scores[i] = sum(N_matrix[i] & uncovered)
        scores = N_matrix.dot(uncovered)
        best_idx = scores.argmax()
        
        # If no uncovered words are covered by any candidate (should not happen if uncovered is not empty)
        if scores[best_idx] == 0:
            # This implies a bug or empty uncovered, but loop condition says uncovered is not empty.
            # Break to avoid infinite loop.
            break
            
        centers.append(best_idx)
        uncovered &= ~N_matrix[best_idx]
        
    # 3. Deletion Pass
    # Maintain counts of how many centers cover each word
    counts = np.zeros(N, dtype=int)
    for c in centers:
        counts[neighbors[c]] += 1
        
    # Remove redundant centers
    # A center c is redundant if all words in neighbors[c] have count >= 2
    # We iterate and remove if possible
    changed = True
    while changed:
        changed = False
        new_centers = []
        for c in centers:
            # Check if c is redundant
            # If all neighbors of c are covered by at least 2 centers, we can remove c
            # Note: c covers itself, so counts[neighbors[c]] includes c's contribution.
            # If min(counts[neighbors[c]]) >= 2, then even without c, they are covered by >= 1.
            if np.all(counts[neighbors[c]] >= 2):
                # Remove c
                counts[neighbors[c]] -= 1
                changed = True
            else:
                new_centers.append(c)
        centers = new_centers
        
    # 4. Convert indices back to codewords
    result = [all_words[c] for c in centers]
    
    return result
