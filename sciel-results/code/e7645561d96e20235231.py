import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 100
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n
    words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        w.reverse()
        words.append(tuple(w))
        
    powers = [1] * n
    for i in range(n-2, -1, -1):
        powers[i] = powers[i+1] * q
        
    def get_idx(w):
        idx = 0
        for i in range(n):
            idx += w[i] * powers[i]
        return idx

    # Build coverage matrix: cover_matrix[i] is a boolean array of length total_words
    # where cover_matrix[i][j] = True if word j is within distance radius of word i
    # For radius=1, word i covers itself and all words at distance 1.
    # We'll use a list of sets for efficiency in the greedy phase.
    
    # Precompute neighbors: for each word, list of words within distance <= radius
    neighbors = []
    for i, w in enumerate(words):
        covered_set = set()
        covered_set.add(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = get_idx(new_w)
                covered_set.add(j)
        neighbors.append(covered_set)
    
    full_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = total_words
    
    # Run greedy with numpy-accelerated gain computation
    # Use bitsets represented as Python integers for the mask, but compute gains via numpy
    # Actually, let's use a different approach: maintain an array of uncovered counts.
    
    restarts = 80
    for r in range(restarts):
        if time.time() > deadline - 10:
            break
        
        # uncovered_count[j] = number of selected codewords that cover word j
        # Initially 0 for all
        uncovered_count = np.zeros(total_words, dtype=np.int32)
        selected = []
        
        # For gain computation, we need to know for each candidate c, how many new words it covers
        # gain(c) = sum of 1 for j in neighbors[c] where uncovered_count[j] == 0
        
        # We'll maintain a list of active candidates and recompute gains efficiently.
        # Key insight: when we add a codeword c, we decrement uncovered_count for all j in neighbors[c]
        # that were previously uncovered. This changes the gain for other candidates.
        
        # For n=7, total_words=2187, and each codeword covers 1 + 7*2 = 15 words.
        # So each addition affects at most 15 positions.
        
        # Let's use a simpler approach: maintain a list of candidates and compute gains on the fly
        # using the uncovered_count array. Since each codeword covers only 15 words, computing
        # gain(c) is O(15).
        
        cand_list = list(range(total_words))
        random.shuffle(cand_list)
        
        while True:
            # Check if all covered
            if uncovered_count.min() > 0:
                break
            
            # Find candidate with maximum gain
            best_gain = -1
            best_cand = -1
            best_idx_in_list = -1
            
            for idx, c in enumerate(cand_list):
                # Compute gain: number of neighbors of c that are not yet covered
                gain = 0
                for j in neighbors[c]:
                    if uncovered_count[j] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_cand = c
                    best_idx_in_list = idx
                elif gain == best_gain and best_cand != -1:
                    if random.random() < 0.5:
                        best_gain = gain
                        best_cand = c
                        best_idx_in_list = idx
            
            if best_cand == -1 or best_gain == 0:
                break
            
            selected.append(best_cand)
            
            # Update uncovered_count
            for j in neighbors[best_cand]:
                uncovered_count[j] += 1
            
            # Remove from candidate list
            cand_list[best_idx_in_list] = cand_list[-1]
            cand_list.pop()
        
        # Post-processing: Remove redundant codewords
        # Check if removing any selected codeword still covers everything
        improved = True
        while improved:
            improved = False
            for i in range(len(selected) - 1, -1, -1):
                # Check if selected[i] is the only one covering some word
                is_redundant = True
                for j in neighbors[selected[i]]:
                    # Count how many other selected codewords cover j
                    count = 0
                    for k in range(len(selected)):
                        if k != i and j in neighbors[selected[k]]:
                            count += 1
                            if count >= 1:
                                break
                    if count == 0:
                        is_redundant = False
                        break
                
                if is_redundant:
                    selected.pop(i)
                    improved = True
                    break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
    
    result = []
    for idx in best_code:
        result.append(list(words[idx]))
    
    return result
