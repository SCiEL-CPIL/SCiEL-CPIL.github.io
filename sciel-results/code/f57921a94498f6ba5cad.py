"""
Ternary covering code K_3(8,1) construction.

Strategy:
1. Start with the direct product of the ternary tetracode (9 words, length 4, perfect
   covering radius 1) with all words on the remaining 4 coordinates: 9 * 3^4 = 729 words.
2. Greedy removal: iteratively remove redundant codewords (those whose removal
   preserves coverage).
3. Branch-and-bound on the conflict graph: find a maximum independent set among
   remaining codewords where edges connect words within Hamming distance 2.
   A valid radius-1 covering code requires no two codewords at distance 2
   (since a word at distance 1 from both would be uncovered if we remove either).
   Actually, that's not quite right - we need coverage, not just independence.
   Instead, use local search: try swaps (remove one, add another) to reduce size
   while maintaining coverage.
"""

import time
import numpy as np
from itertools import product
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit

    if q != 3 or n != 8 or radius != 1:
        # Fallback for general parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
        else:
            return [list(w) for w in product(range(q), repeat=n)]

    # Generate the initial 729-word product code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_words = [h + list(tail) for h in tetracode for tail in product(range(3), repeat=4)]
    words = set(tuple(w) for w in initial_words)

    # Precompute all 3^8 = 6561 possible words
    all_words_list = [tuple(w) for w in product(range(3), repeat=8)]
    all_words_set = set(all_words_list)
    word_to_idx = {w: i for i, w in enumerate(all_words_list)}
    num_words = len(all_words_list)

    def get_neighbors(word):
        """Return list of words within Hamming distance 1 of word (including self)."""
        result = [word]
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    new_word = list(word)
                    new_word[i] = v
                    result.append(tuple(new_word))
        return result

    # Build neighbor index lists for all words in the space
    # This is expensive but done once
    all_neighbor_indices = np.zeros((num_words, 17), dtype=np.int32)
    for w in all_words_list:
        idx = word_to_idx[w]
        nbs = get_neighbors(w)
        for j, nb in enumerate(nbs):
            all_neighbor_indices[idx, j] = word_to_idx[nb]

    # Initial codewords
    cw_indices = [word_to_idx[w] for w in words]
    num_cw = len(cw_indices)
    active = [True] * num_cw

    # Coverage count: how many active codewords cover each word
    cov_count = np.zeros(num_words, dtype=np.int32)
    for idx in cw_indices:
        for nb_idx in all_neighbor_indices[idx]:
            cov_count[nb_idx] += 1

    # Track which codewords cover each word (for swap search)
    # word_coverers[word_idx] = list of codeword indices that cover this word
    word_coverers = [set() for _ in range(num_words)]
    for ci, idx in enumerate(cw_indices):
        if not active[ci]:
            continue
        for nb_idx in all_neighbor_indices[idx]:
            word_coverers[nb_idx].add(ci)

    # Greedy removal phase
    def try_greedy_removal(max_iters=100000):
        nonlocal active, cov_count, word_coverers
        removed_count = 0
        for _ in range(max_iters):
            if time.time() > deadline - 20:
                break
            # Find a removable codeword
            removed = False
            for ci in range(num_cw):
                if not active[ci]:
                    continue
                idx = cw_indices[ci]
                # Check if all neighbors have coverage >= 2
                can_remove = True
                for nb_idx in all_neighbor_indices[idx]:
                    if cov_count[nb_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove codeword ci
                    active[ci] = False
                    removed_count += 1
                    for nb_idx in all_neighbor_indices[idx]:
                        cov_count[nb_idx] -= 1
                        word_coverers[nb_idx].discard(ci)
                    removed = True
                    break
            if not removed:
                break
        return removed_count

    try_greedy_removal()

    # Local search with swaps
    # Try to find: remove codeword A, add codeword B (not in current set)
    # such that coverage is maintained and we don't increase size (ideally decrease)
    # Since we can only remove (not add) to decrease size, we need:
    # Remove A and B, add C, where C covers everything A and B uniquely covered.
    # This is a 2-for-1 swap.

    def get_active_count():
        return sum(1 for a in active if a)

    def try_swap_search(max_iters=50000):
        nonlocal active, cov_count, word_coverers
        improved = 0
        random.seed(42)

        for _ in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Find a codeword that can be removed if we add a replacement
            # Strategy: pick a random active codeword, see what it uniquely covers
            active_cis = [ci for ci in range(num_cw) if active[ci]]
            if not active_cis:
                break

            ci = random.choice(active_cis)
            idx = cw_indices[ci]

            # Find words uniquely covered by this codeword
            unique_words = []
            for nb_idx in all_neighbor_indices[idx]:
                if cov_count[nb_idx] == 1:
                    unique_words.append(nb_idx)

            if not unique_words:
                # Can just remove
                active[ci] = False
                improved += 1
                for nb_idx in all_neighbor_indices[idx]:
                    cov_count[nb_idx] -= 1
                    word_coverers[nb_idx].discard(ci)
                continue

            # Need to find a replacement codeword that covers all unique_words
            # A replacement must be a word not in our current code
            # that is within distance 1 of all unique_words
            # The replacement's ball must contain all unique_words

            # For each unique word, the replacement must be within distance 1
            # So the replacement must be in the intersection of balls around unique_words
            # Ball around word w: w itself and all words at distance 1

            # Intersection of balls: find words within distance 1 of ALL unique_words
            if len(unique_words) > 3:
                continue  # Too many constraints, unlikely to find replacement

            # Candidate replacements: words in ball of first unique word
            candidate_replacements = set()
            first_ukw = unique_words[0]
            for cand in all_neighbor_indices[first_ukw]:
                candidate_replacements.add(cand)

            # Filter: must be within distance 1 of all unique words
            valid_replacements = []
            for cand in candidate_replacements:
                if cand in {cw_indices[ci2] for ci2 in range(num_cw) if active[ci2]}:
                    continue  # Already in code
                valid = True
                for ukw in unique_words:
                    # Check if cand is within distance 1 of ukw
                    # cand is in all_neighbor_indices[ukw] if within dist 1
                    if not any(all_neighbor_indices[ukw, k] == cand for k in range(17)):
                        valid = False
                        break
                if valid:
                    valid_replacements.append(cand)

            if not valid_replacements:
                continue

            # Try adding a valid replacement and removing ci
            for cand in valid_replacements:
                # Check: after removing ci and adding cand, is coverage maintained?
                # Words that only ci covered (and cand doesn't) would be uncovered
                # Words that only cand's ball covers but ci doesn't are fine (new coverage)

                # Simulate: decrement coverage for ci's neighbors, increment for cand's neighbors
                # Check if any word goes to 0

                temp_cov = cov_count.copy()
                for nb_idx in all_neighbor_indices[idx]:
                    temp_cov[nb_idx] -= 1
                for nb_idx in all_neighbor_indices[cand]:
                    temp_cov[nb_idx] += 1

                if np.all(temp_cov >= 1):
                    # Valid swap
                    active[ci] = False
                    for nb_idx in all_neighbor_indices[idx]:
                        cov_count[nb_idx] -= 1
                        word_coverers[nb_idx].discard(ci)

                    # Add new codeword
                    new_ci = len(cw_indices)
                    cw_indices.append(cand)
                    active.append(True)
                    for nb_idx in all_neighbor_indices[cand]:
                        cov_count[nb_idx] += 1
                        word_coverers[nb_idx].add(new_ci)

                    improved += 1
                    break

        return improved

    # Run swap search
    try_swap_search()

    # Another round of greedy removal after swaps
    try_greedy_removal()

    # Extract remaining codewords
    result = [list(all_words_list[idx]) for i, idx in enumerate(cw_indices) if active[i]]

    return result
