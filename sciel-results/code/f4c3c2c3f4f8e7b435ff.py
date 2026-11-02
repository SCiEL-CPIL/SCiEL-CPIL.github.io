"""Geometric Hyperplane Projection approach for K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    Uses geometric hyperplane projection with multi-tile guidance and local search refinement.
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[i] for i in range(q)]
    if n == 2:
        # For n=2, q=3, radius=1: greedy is fine
        pass

    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)

    def word_to_int(word):
        val = 0
        for v in word:
            val = val * q + v
        return val

    # Precompute neighbors for each word index using numpy for speed
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

    # Convert neighbors to numpy arrays for faster computation
    neighbors_np = np.array(neighbors, dtype=np.int32)

    # Geometric hyperplane projection:
    # Use multiple projection schemes to create diverse tile views
    # For n=7, q=3, we use several weight vector pairs

    # Generate multiple projection schemes
    num_schemes = 3
    schemes = []
    for s in range(num_schemes):
        if s == 0:
            weights1 = [1] * n
            weights2 = [(i + 1) % q if q > 1 else 0 for i in range(n)]
        elif s == 1:
            weights1 = [(i % q) if q > 1 else 0 for i in range(n)]
            weights2 = [1] * n
        else:
            weights1 = [1, 2, 1, 2, 1, 2, 1][:n]
            weights2 = [2, 1, 2, 1, 2, 1, 2][:n]
        schemes.append((weights1, weights2))

    # Precompute tile indices for each scheme
    tile_of_word = []  # list of arrays, one per scheme
    for weights1, weights2 in schemes:
        tiles = np.zeros(total_words, dtype=np.int32)
        for idx, w in enumerate(all_words):
            p1 = sum(w[i] * weights1[i] for i in range(n)) % q
            p2 = sum(w[i] * weights2[i] for i in range(n)) % q
            tiles[idx] = p1 * q + p2
        tile_of_word.append(tiles)

    # Count words per tile for each scheme
    tile_size = q * q
    words_per_tile = []
    for tiles in tile_of_word:
        counts = np.bincount(tiles, minlength=tile_size).astype(np.float64)
        words_per_tile.append(counts)

    # Greedy with multi-geometric guidance and local search
    covered = np.zeros(total_words, dtype=bool)
    remaining_uncovered = total_words
    tile_covered_count = [np.zeros(tile_size, dtype=np.float64) for _ in range(num_schemes)]
    codewords = []

    # Use a priority queue-like approach: track uncovered count per candidate
    # For efficiency, we'll use numpy vectorized operations where possible

    max_steps = 250
    for step in range(max_steps):
        if remaining_uncovered == 0:
            break

        best_codeword = -1
        best_newly = -1
        best_tile_score = -1.0

        # Sample candidates for speedup when many words remain
        if remaining_uncovered > total_words * 0.5 and step < 50:
            # Sample a subset of candidates
            candidate_indices = np.random.choice(total_words, size=min(500, total_words), replace=False)
        else:
            candidate_indices = np.arange(total_words)

        for w_idx in candidate_indices:
            nbs = neighbors_np[w_idx]
            # Count newly covered words
            newly_mask = ~covered[nbs]
            newly = np.sum(newly_mask)

            if newly == 0:
                continue

            if newly < best_newly:
                continue
            if newly == best_newly and best_codeword != -1:
                # Only compute tile score if tied on newly
                pass

            # Compute multi-scheme tile deficiency score
            tile_score = 0.0
            for s in range(num_schemes):
                # Get tiles of newly covered words
                new_tiles = tile_of_word[s][nbs[newly_mask]]
                # Count how many new words in each tile
                tile_newly = np.bincount(new_tiles, minlength=tile_size)
                # Compute deficiency: newly_in_tile * (1 - current_density)
                density = tile_covered_count[s] / words_per_tile[s]
                # Avoid division by zero
                density = np.where(words_per_tile[s] > 0, density, 0.0)
                tile_score += np.sum(tile_newly * (1.0 - density))

            if newly > best_newly or (newly == best_newly and tile_score > best_tile_score):
                best_newly = newly
                best_tile_score = tile_score
                best_codeword = w_idx

        if best_codeword == -1:
            break

        # Add the best codeword
        codewords.append(all_words[best_codeword])
        nbs = neighbors_np[best_codeword]
        newly_mask = ~covered[nbs]
        for nb in nbs[newly_mask]:
            covered[nb] = True
            remaining_uncovered -= 1
            for s in range(num_schemes):
                t = tile_of_word[s][nb]
                tile_covered_count[s][t] += 1

    # Local search refinement: try to swap or remove codewords
    # Try removing each codeword and see if coverage is maintained
    final_codewords = list(codewords)
    for _ in range(3):  # Multiple passes
        improved = True
        while improved:
            improved = False
            for i in range(len(final_codewords)):
                # Try removing codeword i
                test_codewords = final_codewords[:i] + final_codewords[i+1:]
                if _is_valid_covering(test_codewords, q, n, radius, all_words, word_to_int):
                    final_codewords = test_codewords
                    improved = True
                    break

    return [list(w) for w in final_codewords]

def _is_valid_covering(codewords, q, n, radius, all_words, word_to_int):
    """Check if the given codewords form a valid covering code."""
    if not codewords:
        return False

    # Build a set of covered words
    covered_set = set()
    for cw in codewords:
        cw_int = word_to_int(cw)
        # Add all words within Hamming distance radius
        cw_list = list(cw)
        for perm in _hamming_sphere(cw_list, n, q, radius):
            covered_set.add(word_to_int(perm))

    # Check if all words are covered
    for w in all_words:
        if word_to_int(w) not in covered_set:
            return False
    return True

def _hamming_sphere(center, n, q, radius):
    """Generate all words within Hamming distance radius from center."""
    result = []
    # Use itertools for combinations of positions to change
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            # Generate all combinations of values for these positions
            center_list = list(center)
            if r == 0:
                result.append(tuple(center_list))
            else:
                # For each position, choose a different value
                value_options = []
                for pos in positions:
                    options = [v for v in range(q) if v != center_list[pos]]
                    value_options.append(options)
                for combo in itertools.product(*value_options):
                    new_word = list(center_list)
                    for pos, val in zip(positions, combo):
                        new_word[pos] = val
                    result.append(tuple(new_word))
    return result
# EVOLVE-BLOCK-END
