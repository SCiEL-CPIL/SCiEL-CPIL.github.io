import numpy as np
import itertools

def construct(q, n, radius):
    # Specific optimization for q=3, n=8, radius=1 using Greedy Set Cover with Adjacency Matrix
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        # Simple greedy or direct sum
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(x) for x in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n
    # Generate all words as indices 0..6560
    # We need to map index <-> word to build adjacency
    
    # Precompute neighbors for each index
    # Instead of full matrix, we can build sparse structure or just the list of neighbors
    # But for the "Dot Product" trick, we need the matrix or equivalent.
    # Let's build the dense adjacency matrix.
    
    # 1. Map index to word (list of ints)
    # 2. For each index i, find all j such that dist(i, j) <= 1
    # 3. Set A[i, j] = 1
    
    # To build A efficiently:
    # A[i, i] = 1
    # For each pos k, for each delta in [-1, 1] (if valid), find j, set A[i, j]=1, A[j, i]=1
    
    # We need a helper to convert index to word and back, or just manipulate digits.
    
    # Let's create the matrix
    A = np.zeros((total_words, total_words), dtype=np.float32) # float for dot product
    # Or use int8 and cast? float32 is fine.
    
    # Helper to get digit at position k of index i
    # i = d0 * 3^7 + d1 * 3^6 + ... + d7
    # Actually, standard base-3 representation.
    
    # Precompute powers
    powers = [3**k for k in range(n)]
    
    # Fill diagonal
    A[np.arange(total_words), np.arange(total_words)] = 1.0
    
    # Fill off-diagonals
    # For each index i
    for i in range(total_words):
        # Get digits
        temp = i
        digits = [0] * n
        for k in range(n):
            digits[k] = temp % 3
            temp //= 3
        # digits[0] is least significant (pos 0)
        
        for k in range(n):
            val = digits[k]
            for delta in [-1, 1]:
                new_val = val + delta
                if 0 <= new_val < 3:
                    # Calculate new index
                    # j = i + delta * 3^k
                    j = i + delta * powers[k]
                    A[i, j] = 1.0
                    A[j, i] = 1.0 # Symmetric

    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=np.float32)
    code_indices = []
    
    # To ensure we get a good result, we might run this multiple times with different random seeds for tie-breaking
    # But the matrix is fixed. The randomness is in choosing among max-gain candidates.
    
    best_code = None
    best_size = total_words
    
    # Run a few iterations to find a smaller code
    for _ in range(3):
        uncovered = np.ones(total_words, dtype=np.float32)
        code_indices = []
        
        while np.any(uncovered > 0):
            # Calculate gains: for each candidate i, how many uncovered words does it cover?
            # gain[i] = sum(uncovered[j] for j in neighbors(i))
            # Since A[i, j] is 1 if j is neighbor of i, gain = A.dot(uncovered)
            gains = A.dot(uncovered)
            
            max_gain = gains.max()
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one to break ties and encourage diversity
            c = np.random.choice(candidates)
            
            code_indices.append(c)
            
            # Update uncovered
            # Mark all neighbors of c as covered (set to 0)
            # uncovered[A[c, :] == 1] = 0
            # This creates a boolean mask.
            neighbors_mask = A[c, :] == 1
            uncovered[neighbors_mask] = 0
            
        if len(code_indices) < best_size:
            best_size = len(code_indices)
            best_code = code_indices.copy()
            
    # Convert indices back to words
    final_codewords = []
    for idx in best_code:
        temp = idx
        word = [0] * n
        for k in range(n):
            word[k] = temp % 3
            temp //= 3
        final_codewords.append(word)
        
    return final_codewords
