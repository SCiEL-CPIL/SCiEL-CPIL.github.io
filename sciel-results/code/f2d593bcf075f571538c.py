"""Improved covering code construction via greedy + local search."""
import time
import random
import numpy as np

def _hamming_dist(a, b):
    return int(np.count_nonzero(np.array(a) != np.array(b)))

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # leave 5s safety margin
    
    total_words = q ** n
    
    # Special case: if we can afford a perfect code construction
    # For q=3, n=7, radius=1: the tetracode direct sum gives 243
    # We want to do better via local search
    
    # Initialize with a random set of codewords
    # We'll use a greedy construction then try to prune/improve
    
    def vec_to_int(vec):
        val = 0
        for i in range(n):
            val = val * q + vec[i]
        return val
    
    def int_to_vec(val):
        vec = [0] * n
        for i in range(n - 1, -1, -1):
            vec[i] = val % q
            val //= q
        return vec
    
    # Precompute: for each word, its neighborhood (itself + all words at Hamming distance 1)
    # This lets us check coverage efficiently
    
    # Actually, let's think about this differently.
    # A codeword covers: itself + all words at distance 1 = 1 + n*(q-1) = 1 + 7*2 = 15 words
    
    # Strategy:
    # 1. Start with a random set of ~200 codewords
    # 2. Check coverage: is every word within distance 1 of some codeword?
    # 3. If not covered, add codewords to cover uncovered words
    # 4. Try to remove redundant codewords
    # 5. Iterate
    
    # To check if a word is covered by a set of codewords, we need to check
    # if any codeword is within distance 1.
    
    # For efficiency with numpy:
    # Represent all words as a matrix of shape (total_words, n)
    # Represent codewords as a matrix of shape (k, n)
    # Distance matrix would be (total_words, k) which is 2187 * 200 = 437k entries
    # That's manageable.
    
    # Let's precompute all words
    all_words = np.array([int_to_vec(i) for i in range(total_words)], dtype=np.int8)
    
    def is_covered(codeword_list):
        """Check if every word is within radius of some codeword.
        Returns list of uncovered words (as indices into all_words)."""
        if not codeword_list:
            return list(range(total_words))
        cw = np.array(codeword_list, dtype=np.int8)  # shape (k, n)
        # Compute distances: for each word, min distance to any codeword
        # dist[i, j] = sum(all_words[i] != cw[j])
        # We need dist[i] <= radius for all i
        # dist matrix is (total_words, k)
        # Let's compute it in chunks to save memory
        k = len(codeword_list)
        uncovered = []
        chunk_size = max(1, total_words // 10)
        for start in range(0, total_words, chunk_size):
            end = min(start + chunk_size, total_words)
            chunk = all_words[start:end]  # (chunk_len, n)
            # distances: (chunk_len, k)
            dists = np.sum(chunk[:, np.newaxis, :] != cw[np.newaxis, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            for idx in range(start, end):
                if min_dists[idx - start] > radius:
                    uncovered.append(idx)
        return uncovered
    
    def find_best_cw_for_word(word_idx, existing_cw):
        """Find a codeword (as int) that maximally covers uncovered words.
        For simplicity, just return the word itself."""
        return word_idx
    
    # Greedy construction:
    # Start with empty set, repeatedly add the codeword that covers the most uncovered words
    
    def greedy_construct(target_size=None):
        """Greedy: add codewords one at a time to maximize coverage."""
        uncovered = set(range(total_words))
        codewords = []
        
        while uncovered:
            # For each candidate codeword position, count how many uncovered words it covers
            # A codeword at position c covers: c itself + all words at distance 1 from c
            # But we only care about covering currently uncovered words
            # 
            # To find the best codeword, we could:
            # 1. For each uncovered word w, the best codeword to place near w is w itself
            #    (covers w and its neighbors)
            # 2. But a codeword might cover multiple uncovered words
            #
            # Greedy approach: pick the uncovered word w, place codeword at w.
            # This is a simple greedy set cover.
            
            # Pick a random uncovered word to break ties
            w_idx = random.choice(list(uncovered))
            cw = int_to_vec(w_idx)
            
            # Remove all words within distance 1 of cw from uncovered
            cw_arr = np.array(cw, dtype=np.int8)
            for u_idx in list(uncovered):
                if np.sum(all_words[u_idx] != cw_arr) <= radius:
                    uncovered.discard(u_idx)
            
            codewords.append(cw)
            
            if target_size and len(codewords) >= target_size:
                break
        
        return codewords
    
    # Local search: try to replace a codeword with a better one, or remove redundant ones
    
    def try_remove(codewords):
        """Try to remove codewords that are redundant."""
        removed_any = False
        for i in range(len(codewords)):
            if time.time() > deadline - 5:
                break
            # Try removing codeword i
            candidate = codewords[:i] + codewords[i+1:]
            if not candidate:
                continue
            uncovered = is_covered(candidate)
            if not uncovered:
                codewords = candidate
                removed_any = True
                break
        return codewords, removed_any
    
    def try_replace(codewords):
        """Try replacing a codeword with a random one to see if we can maintain coverage
        and then try to remove another."""
        if len(codewords) < 2:
            return codewords
        i = random.randint(0, len(codewords) - 1)
        # Try a random replacement
        new_cw = int_to_vec(random.randint(0, total_words - 1))
        candidate = codewords[:i] + [new_cw] + codewords[i+1:]
        uncovered = is_covered(candidate)
        if not uncovered:
            # Check if we can now remove some codeword
            for j in range(len(candidate)):
                if j == i:
                    continue
                cand2 = candidate[:j] + candidate[j+1:]
                if not cand2:
                    continue
                uncovered2 = is_covered(cand2)
                if not uncovered2:
                    return cand2
        return codewords
    
    # Main optimization loop
    best_code = greedy_construct()
    best_size = len(best_code)
    
    # Try to improve by removing redundant codewords
    while time.time() < deadline - 10:
        progress = False
        
        # Multiple removal passes
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            best_code, removed = try_remove(best_code)
            if removed:
                best_size = len(best_code)
                progress = True
        
        # Try replacements
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            new_code = try_replace(best_code)
            if len(new_code) < len(best_code):
                best_code = new_code
                best_size = len(best_code)
                progress = True
        
        if not progress:
            # Try a fresh random restart if stuck
            if time.time() > deadline - 20:
                break
            new_code = greedy_construct()
            if len(new_code) < len(best_code):
                best_code = new_code
                best_size = len(best_code)
    
    # Final cleanup: remove any remaining redundant codewords
    while time.time() < deadline - 2:
        best_code, removed = try_remove(best_code)
        if not removed:
            break
    
    return best_code
