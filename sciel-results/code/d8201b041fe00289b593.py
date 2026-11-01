import itertools
import time

def construct(q, n, radius):
    """
    Constructs a covering code using a greedy set cover approach.
    """
    # Generate all possible words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Map word to index for O(1) lookup
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (words within Hamming distance <= radius)
    # For q=3, n=6, radius=1, neighbors are the word itself and words differing in 1 pos.
    # Number of neighbors = 1 + n * (q - 1) = 1 + 6 * 2 = 19
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # self
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    # Create new word
                    w_list = list(w)
                    w_list[pos] = new_val
                    new_word = tuple(w_list)
                    neighbors[i].append(word_to_idx[new_word])
                    
    # Coverage state: True if covered, False if not
    # We want to cover all words.
    uncovered = [True] * N
    
    # List of selected codewords (indices)
    code_indices = []
    
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    while True:
        if time.time() > deadline:
            break
            
        # Check if all covered
        # Optimization: keep a count of uncovered words
        if not any(uncovered):
            break
            
        best_idx = -1
        best_gain = -1
        
        # Greedy step: find word that covers the most *currently uncovered* words
        for i in range(N):
            gain = 0
            for j in neighbors[i]:
                if uncovered[j]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            # No more uncovered words can be covered? Should not happen if graph is connected
            break
            
        code_indices.append(best_idx)
        
        # Mark neighbors as covered
        for j in neighbors[best_idx]:
            uncovered[j] = False
            
    # Convert indices back to word lists
    result = [list(all_words[i]) for i in code_indices]
    
    return result
