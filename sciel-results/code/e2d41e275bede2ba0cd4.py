"""Dual greedy set cover for ternary covering code K_3(8,1)."""
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    total_words = q ** n
    # Represent words as integers in [0, total_words)
    # Base-q representation: word w has digits w[i]

    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # For radius 1, a codeword covers itself and all words at Hamming distance 1
    # Number of words covered by one codeword = 1 + n*(q-1)
    cover_size = 1 + n * (q - 1)

    # We'll use a greedy approach with dual/shadow prices
    # uncovered: set of indices of uncovered words
    uncovered = set(range(total_words))

    # For each candidate codeword (word), precompute which words it covers
    # But that's total_words * cover_size which is 6561 * 25 = 164025, manageable

    # Actually, let's think about this differently.
    # For a given word w, the set of words it covers (within Hamming distance radius)
    # For radius 1: w itself, and for each position i, change w[i] to each other value in 0..q-1

    def get_cover(word_idx):
        """Return list of word indices covered by codeword at word_idx."""
        w = all_words[word_idx]
        covered = [word_idx]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_word = list(w)
                new_word[pos] = val
                new_idx = word_to_idx[tuple(new_word)]
                covered.append(new_idx)
        return covered

    # Precompute covers for all words
    # This is 6561 * 25 = 164025 entries, each a list of 25 ints
    covers = [get_cover(i) for i in range(total_words)]

    # For efficiency, also precompute for each word, which codewords cover it
    # word_covered_by[word_idx] = list of codeword indices that cover this word
    word_covered_by = [[] for _ in range(total_words)]
    for cw_idx in range(total_words):
        for covered_idx in covers[cw_idx]:
            word_covered_by[covered_idx].append(cw_idx)

    # Dual greedy: maintain shadow prices for uncovered words
    # Initially uniform. At each step, pick codeword maximizing sum of shadow prices of newly covered words.
    # Then update shadow prices: increase prices of words that have few remaining candidate coverers.

    # Track how many times each uncovered word is still "available" to be covered
    # i.e., how many codewords that could cover it haven't been chosen yet and still cover it
    # Actually, a codeword covers a word if Hamming distance <= radius.
    # Once we pick a codeword, words it covers become covered.
    # The "scarcity" of a word = number of codewords that cover it (among all possible codewords).
    # But as we pick codewords, some of those options are used up... but actually in set cover,
    # once a word is covered, it's done. The issue is: which uncovered words are "hard" to cover?
    # A word is hard if few codewords cover it. But in this symmetric space, every word is covered
    # by exactly cover_size codewords (the words within Hamming distance 1 of it).
    # Wait, no: word_covered_by[w] has size = number of codewords c such that dist(c,w) <= 1.
    # For radius 1, that's 1 + n*(q-1) = cover_size. So every word is covered by exactly cover_size codewords.
    # So the scarcity is uniform initially.

    # The dual greedy idea: shadow price p[w] for each uncovered word w.
    # Score of codeword c = sum of p[w] for w in cover(c) ∩ uncovered.
    # Pick c with max score. Add c to solution. Remove covered words from uncovered.
    # Update: for remaining uncovered words, increase p[w] based on how few codewords still "can" cover it.
    # But all codewords can still cover any uncovered word (we haven't removed codewords).
    # So the dual update is: p[w] *= (1 + 1/remaining_coverers[w]) or something.
    # Actually, remaining_coverers[w] = number of codewords that cover w. This is constant = cover_size.
    # Hmm, that doesn't help.

    # Let me reconsider. The standard greedy picks max coverage. The "dual" version might use
    # a different scoring. Let me try: score(c) = sum over w in cover(c) ∩ uncovered of (1 / len(word_covered_by[w])).
    # Since all have same len, this is just |cover(c) ∩ uncovered|. Same as greedy.

    # Alternative: random restarts with greedy, and keep best.
    # Or: greedy with random tie-breaking.

    # Let's do: multiple greedy passes with random tie-breaking, keep best.
    # Also try: after greedy, do local improvement (swap codewords to reduce count).

    best_solution = None
    best_size = float('inf')

    # Run greedy with random tie-breaking multiple times
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        random.seed(restart * 12345 + 7)
        uncovered = set(range(total_words))
        solution = []

        while uncovered:
            if time.time() > deadline - 5.0:
                break

            # Find codeword covering most uncovered words
            best_cw = -1
            best_count = -1
            # To speed up, we can sample or use a smarter approach
            # For 6561 words and ~500 iterations, checking all 6561 codewords each time is 3.2M ops per restart
            # Each check involves iterating over 25 covered words. So 80M ops per restart. Might be too slow.

            # Optimize: precompute for each codeword the list of covered words.
            # For each codeword, count how many of its covered words are in uncovered.
            # We can do this efficiently.

            # Actually, let's just do it and see if it's fast enough.
            # 6561 codewords * 25 covered words = 164025 checks per iteration.
            # ~500 iterations * 164025 = 82M checks. In Python, this might take ~10-20s per restart.
            # Too slow for 50 restarts.

            # Better: use numpy or a more efficient approach.
            # Or: reduce the search space. Note that by symmetry, we can fix the first word to 0...
            # Actually, let's just do greedy but with a smarter data structure.

            # Alternative approach: instead of checking all codewords, note that
            # the best codeword to add is one that covers the most uncovered words.
            # We can maintain for each codeword a count of how many uncovered words it covers.
            # When we add a codeword and remove its covered words, we update counts for all
            # codewords that share covered words.

            # Let me use a different strategy: precompute everything and use numpy for speed.

            break  # This approach is too slow in pure Python

        # This restart didn't complete, skip
        continue

    # Let me use a numpy-based approach for speed

    import numpy as np

    # Represent words as integers 0..total_words-1
    # Precompute cover sets as numpy arrays
    covers_arr = np.array(covers, dtype=np.int32)  # shape (total_words, cover_size)

    # For each word, the list of codewords that cover it
    # word_covered_by_arr: for each word, list of codeword indices
    # This is a ragged array, but all have same length = cover_size
    word_covered_by_arr = np.array(word_covered_by, dtype=np.int32)  # shape (total_words, cover_size)

    # Greedy with numpy:
    # uncovered_mask: boolean array of length total_words
    # For each codeword c, coverage = sum(uncovered_mask[covers_arr[c]])
    # Pick c with max coverage

    for restart in range(30):
        if time.time() > deadline - 10.0:
            break

        random.seed(restart * 9999 + 13)
        uncovered_mask = np.ones(total_words, dtype=bool)
        solution = []

        while uncovered_mask.any():
            if time.time() > deadline - 10.0:
                break

            # Compute coverage of each codeword
            # covers_arr shape: (total_words, cover_size)
            # For each row, sum uncovered_mask at those positions
            # This is: uncovered_mask[covers_arr].reshape(total_words, cover_size).sum(axis=1)
            coverage = uncovered_mask[covers_arr].reshape(total_words, cover_size).sum(axis=1)

            # Find max with random tie-breaking
            max_val = coverage.max()
            candidates = np.where(coverage == max_val)[0]
            if len(candidates) > 1:
                chosen = candidates[random.randint(0, len(candidates) - 1)]
            else:
                chosen = candidates[0]

            solution.append(chosen)
            # Remove covered words
            uncovered_mask[covers_arr[chosen]] = False

        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = list(solution)

        if time.time() > deadline - 10.0:
            break

    if best_solution is None:
        # Fallback: use the original tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert integer indices back to word lists
    result = [list(all_words[idx]) for idx in best_solution]

    # Optional: try local improvement (swap) to reduce size further
    # For now, return what we have

    return result
