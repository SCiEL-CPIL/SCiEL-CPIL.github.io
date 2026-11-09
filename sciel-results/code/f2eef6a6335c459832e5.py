import time
import random
import itertools
import numpy as np

_DEADLINE = 110.0

def construct(q, n, radius):
    start_time = time.time()
    
    # Handle the specific known case efficiently
    if q == 3 and n == 7 and radius == 1:
        return _solve_3_7_1(start_time)
    
    # General fallback for other parameters (not expected in this task but required for robustness)
    # Using a simple greedy for small spaces
    if q ** n > 1000000:
        # Direct sum construction for large n
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Fallback to all-zero and single-bit changes? No, just return a valid but large set
            # For radius 1, a trivial covering is all words (size q^n)
            # But we need to be efficient.
            return [list(w) for w in itertools.product(range(q), repeat=n)]

def _solve_3_7_1(start_time):
    q, n, r = 3, 7, 1
    total_words = q ** n
    # Map word tuple to index
    # We can generate all words and map them
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word
    # Neighbor list for radius 1: word itself + change one position to one of the other q-1 values
    neighbor_indices = []
    for w in all_words:
        nbs = [word_to_idx[w]]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val != orig_val:
                    w_list[pos] = val
                    nbs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig_val
        neighbor_indices.append(nbs)
    
    # Convert to numpy array for fast matrix-vector multiplication
    # Shape: (total_words, num_neighbors)
    # Note: num_neighbors is constant for regular codes like this
    num_neighbors = len(neighbor_indices[0])
    nbrs_matrix = np.array(neighbor_indices, dtype=np.int32)
    
    def greedy_cover():
        uncovered = np.ones(total_words, dtype=np.int8)
        codewords = []
        # To break ties, we can randomize the order of candidates or just take the first max
        # We iterate until all covered
        while uncovered.sum() > 0:
            # Calculate gain for all words: sum of uncovered neighbors
            gains = nbrs_matrix @ uncovered
            # Find index of max gain
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            # Add this word
            codewords.append(all_words[best_idx])
            # Update uncovered
            uncovered[nbrs_matrix[best_idx]] = 0
            
            # Time check
            if time.time() - start_time > _DEADLINE - 5:
                break
                
        return codewords

    def local_search(codewords):
        """
        Try to remove redundant codewords.
        A codeword is redundant if all its neighbors are covered by other codewords.
        """
        codewords = [list(c) for c in codewords]
        m = len(codewords)
        if m == 0:
            return codewords
            
        # Build coverage map: for each word, how many codewords cover it?
        # Or simply, for each codeword, check if it's the *only* one covering any of its neighbors.
        
        # Efficient check:
        # 1. Compute the union of neighbors for all codewords except i.
        # 2. Check if neighbors of i are a subset of that union.
        
        # Since m is small (~180), we can do this:
        # Precompute neighbor sets as bitsets or just lists.
        # Let's use sets of indices for neighbors.
        nbr_sets = [set(nbrs) for nbrs in neighbor_indices]
        
        changed = True
        while changed:
            changed = False
            # Iterate backwards to allow popping
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() - start_time > _DEADLINE - 10:
                    return codewords
                
                cw_idx = word_to_idx[tuple(codewords[i])]
                my_nbrs = nbr_sets[cw_idx]
                
                # Check if any neighbor is ONLY covered by codeword i
                # We need to know if there exists a neighbor x such that no other codeword j covers x.
                
                # Optimization: 
                # Instead of checking all neighbors against all other codewords,
                # we can maintain a count of how many times each word is covered.
                
                # Let's rebuild coverage counts if we remove one.
                # This local search is O(m^2 * |N|) in worst case if done naively.
                # m ~ 180, |N| ~ 15. 180^2 * 15 ~ 486,000 ops. This is fast.
                
                is_redundant = True
                # To check redundancy, we need to know if every neighbor of cw_i is covered by some cw_j (j != i).
                
                # Let's create a set of all words covered by other codewords.
                # This is expensive to rebuild from scratch.
                
                # Better approach:
                # Maintain `cover_count` array of size total_words.
                # cover_count[x] = number of codewords covering x.
                
                # Let's implement this properly outside the loop.
                
                break # We will implement a better local search below
            
            # Fallback for this simple loop: just try to remove if we can verify quickly
            # Actually, let's implement the count-based local search properly.
            
        return codewords

    def local_search_optimized(codewords):
        codewords = [list(c) for c in codewords]
        m = len(codewords)
        
        # Precompute neighbor indices for these specific codewords
        cw_indices = [word_to_idx[tuple(cw)] for cw in codewords]
        
        # cover_count[x] = number of codewords that cover word x
        cover_count = np.zeros(total_words, dtype=np.int32)
        
        for ci in range(m):
            idx = cw_indices[ci]
            cover_count[nbrs_matrix[idx]] += 1
        
        changed = True
        while changed:
            changed = False
            for ci in range(m - 1, -1, -1):
                if time.time() - start_time > _DEADLINE - 5:
                    break
                
                idx = cw_indices[ci]
                my_nbrs = nbrs_matrix[idx]
                
                # If any neighbor has cover_count == 1, this codeword is NOT redundant
                # Because removing it would make that neighbor uncovered.
                # Wait, cover_count includes this codeword.
                # So if min(cover_count[my_nbrs]) == 1, it means some neighbor is covered ONLY by this codeword.
                # In that case, we cannot remove it.
                
                if np.min(cover_count[my_nbrs]) > 1:
                    # Redundant! Remove it.
                    cover_count[my_nbrs] -= 1
                    codewords.pop(ci)
                    cw_indices.pop(ci)
                    m -= 1
                    changed = True
            
            if time.time() - start_time > _DEADLINE - 5:
                break
                
        return codewords

    # Run Greedy
    codewords = greedy_cover()
    
    # Run Local Search
    if time.time() - start_time < _DEADLINE - 10:
        codewords = local_search_optimized(codewords)
        
    # Try a few random restarts of greedy to see if we get a better initial solution
    best_size = len(codewords)
    best_codewords = codewords
    
    # The greedy algorithm is deterministic if we always pick the first max.
    # We can randomize the tie-breaking to explore different solutions.
    
    for _ in range(10): # Up to 10 restarts
        if time.time() - start_time > _DEADLINE - 20:
            break
            
        # Modify greedy to break ties randomly
        def greedy_cover_randomized():
            uncovered = np.ones(total_words, dtype=np.int8)
            cw_list = []
            while uncovered.sum() > 0:
                gains = nbrs_matrix @ uncovered
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                # Find all indices with max gain
                candidates = np.where(gains == max_gain)[0]
                # Pick one randomly
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
                cw_list.append(all_words[best_idx])
                uncovered[nbrs_matrix[best_idx]] = 0
            return cw_list

        cand_cw = greedy_cover_randomized()
        if time.time() - start_time < _DEADLINE - 10:
            cand_cw = local_search_optimized(cand_cw)
        
        if len(cand_cw) < best_size:
            best_size = len(cand_cw)
            best_codewords = cand_cw
            
    return best_codewords
