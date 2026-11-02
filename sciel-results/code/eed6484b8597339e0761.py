"""Geometric Hyperplane Projection approach for K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    Uses geometric hyperplane projection to guide greedy selection.
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[i] for i in range(q)]
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)

    def word_to_int(word):
        val = 0
        for v in word:
            val = val * q + v
        return val

    # Precompute neighbors for each word index
    neighbors = []
    for w in all_words:
        w_int = word_to_int(w)
        nbs = [w_int]
        w_list = list(w)
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val == old_val:
                    continue
                w_list[pos] = new_val
                nbs.append(word_to_int(tuple(w_list)))
            w_list[pos] = old_val
        neighbors.append(nbs)

    # Geometric Hyperplane Projection Parameters
    # We use two linear functionals to create a 2D tile structure.
    weights1 = [1] * n
    weights2 = [(i + 1) % q if q > 1 else 0 for i in range(n)]
    
    # Precompute tile index for each word
    tile_of_word = [0] * total_words
    for idx, w in enumerate(all_words):
        p1 = sum(w[i] * weights1[i] for i in range(n)) % q
        p2 = sum(w[i] * weights2[i] for i in range(n)) % q
        tile_of_word[idx] = p1 * q + p2

    tile_size = q * q
    words_per_tile = [0] * tile_size
    for t in tile_of_word:
        words_per_tile[t] += 1

    def run_greedy(seed):
        rng = random.Random(seed)
        covered = [False] * total_words
        remaining_uncovered = total_words
        tile_covered_count = [0] * tile_size
        codewords = []
        
        # To break ties deterministically but differently per seed, we can use a hash
        # or just rely on the iteration order if we shuffle candidates?
        # Actually, we can add a small random perturbation to the score or use hash.
        
        for step in range(400):
            if remaining_uncovered == 0:
                break

            best_codeword = -1
            best_score = (-1, -1.0, -1)  # (newly, tile_score, tie_breaker)

            # Iterate through all candidates
            # Optimization: We can skip candidates that are already codewords? 
            # No, a codeword might cover new words if it wasn't fully covered before? 
            # Actually, if a word is a codeword, its neighbors are already covered. 
            # So newly_covered will be 0 for any existing codeword.
            # We can maintain a set of chosen words to skip them.
            
            chosen_set = set(codewords)

            for w_idx in range(total_words):
                if w_idx in chosen_set:
                    continue
                    
                newly = 0
                tile_newly = [0] * tile_size
                for nb in neighbors[w_idx]:
                    if not covered[nb]:
                        newly += 1
                        t = tile_of_word[nb]
                        tile_newly[t] += 1

                if newly == 0:
                    continue

                tile_score = 0.0
                for t in range(tile_size):
                    if tile_newly[t] > 0:
                        density = tile_covered_count[t] / words_per_tile[t] if words_per_tile[t] > 0 else 0
                        tile_score += tile_newly[t] * (1.0 - density)

                # Tie breaker: use a hash of the word index combined with seed
                # This ensures different runs explore different local optima
                tie_breaker = (w_idx * 31 + seed) % 10000
                
                score = (newly, tile_score, tie_breaker)
                if score > best_score:
                    best_score = score
                    best_codeword = w_idx

            if best_codeword == -1:
                break

            # Add the best codeword
            codewords.append(best_codeword)
            for nb in neighbors[best_codeword]:
                if not covered[nb]:
                    covered[nb] = True
                    remaining_uncovered -= 1
                    t = tile_of_word[nb]
                    tile_covered_count[t] += 1

        # Pruning phase: Remove redundant codewords
        # A codeword is redundant if all its neighbors are covered by other codewords.
        # We check this by counting how many codewords cover each neighbor.
        # If a neighbor is covered by >= 2 codewords, it's "safe".
        # If all neighbors of a codeword C are covered by >= 1 OTHER codeword, C is redundant.
        
        # Efficient pruning:
        # For each codeword, check if removing it leaves all its neighbors covered.
        # This is O(C * K_neighbors * C) if done naively.
        # Better: Maintain a count of how many codewords cover each word.
        
        cover_count = [0] * total_words
        for c in codewords:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        redundant = set()
        for c in codewords:
            is_redundant = True
            for nb in neighbors[c]:
                # If this neighbor is ONLY covered by c, then c is not redundant
                if cover_count[nb] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant.add(c)
        
        if redundant:
            # Remove redundant codewords and update cover_count
            new_codewords = [c for c in codewords if c not in redundant]
            # We need to recompute cover_count or just return new_codewords
            # Since we just need the list, we can return it.
            # However, removing one might make another redundant? 
            # Usually, greedy codes don't have deep redundancy chains, but let's do one pass.
            codewords = new_codewords
            
        return [list(all_words[c]) for c in codewords]

    # Multi-start greedy
    best_result = None
    best_len = float('inf')
    
    # Run a few times with different seeds to break symmetry
    for seed in range(10):
        res = run_greedy(seed)
        if len(res) < best_len:
            best_len = len(res)
            best_result = res
            
    return best_result
# EVOLVE-BLOCK-END
