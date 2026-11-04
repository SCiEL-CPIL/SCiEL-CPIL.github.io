import time
import itertools
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0

    # Total number of words in the space
    total = q ** n

    # Precompute all words as integer indices 0..total-1
    # Each word is a base-q number of n digits
    # We'll use a coverage bitmask: bit i is set if word i is covered

    # For efficiency, we represent coverage as a numpy bool array of length total
    # and for each candidate codeword, we precompute the set of words it covers

    # The tetracode direct-sum gives us a good initial set of candidates
    # but we want to do greedy set cover from scratch for potentially better results

    # Actually, let's use the tetracode direct-sum as a starting point,
    # then prune redundancies. This is the key structural change:
    # 1. Start with the known 486-word code (tetracode direct sum)
    # 2. Verify coverage
    # 3. Prune: remove codewords whose coverage is entirely redundant

    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    # Build the full code: tetracode x all tails of length n-4
    if n == 4:
        code = [list(t) for t in tetracode]
    else:
        code = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                code.append(h + list(tail))

    # Convert to numpy for fast operations
    code_arr = np.array(code, dtype=np.int8)
    m = len(code_arr)

    # Precompute: for each codeword, which words does it cover?
    # A word is within Hamming distance <= radius of a codeword.
    # For radius=1, a codeword covers itself + all words differing in exactly 1 position.
    # Number of words covered per codeword = 1 + n*(q-1)

    # Represent each word as a base-q integer
    # word index = sum(word[i] * q^(n-1-i))

    # Precompute powers of q
    q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    # Convert codewords to integer indices
    code_indices = code_arr.astype(np.int64) @ q_powers

    # For each codeword, compute the set of covered word indices
    # covered[c] = list of word indices within distance <= radius of codeword c

    # For radius=1: covered words are the codeword itself plus all single-position changes
    # This is efficient: 1 + n*(q-1) words per codeword

    covered_sets = []
    for c in range(m):
        cw = code_arr[c]
        covered = [code_indices[c]]
        for pos in range(n):
            for val in range(q):
                if val == cw[pos]:
                    continue
                # Compute the index of the word with position pos changed to val
                # delta = (val - cw[pos]) * q_powers[pos]
                idx = code_indices[c] + (val - cw[pos]) * q_powers[pos]
                covered.append(int(idx))
        covered_sets.append(np.array(covered, dtype=np.int32))

    # Now do redundancy pruning:
    # A codeword c is redundant if every word it covers is also covered by at least one other codeword.
    # We process codewords in order and try to remove each one if possible.

    # For efficiency, maintain a coverage count: for each word, how many codewords cover it.
    # If we remove codeword c, word w is still covered iff count[w] >= 2 (before removal).

    # Build coverage count array
    cov_count = np.zeros(total, dtype=np.int32)
    for c in range(m):
        cov_count[covered_sets[c]] += 1

    # Try to prune: iterate through codewords, try to remove each
    # A codeword can be removed if min(cov_count[covered_sets[c]]) >= 2
    # After removal, decrement cov_count for its covered words

    active = np.ones(m, dtype=bool)
    removed_count = 0

    # Multiple passes to find more redundancies
    for pass_num in range(3):
        if time.time() > deadline - 5.0:
            break
        for c in range(m):
            if not active[c]:
                continue
            # Check if all words covered by c are still covered by others
            cs = covered_sets[c]
            if np.all(cov_count[cs] >= 2):
                # Remove codeword c
                cov_count[cs] -= 1
                active[c] = False
                removed_count += 1

    # Build the final code
    final_code = [list(code_arr[c]) for c in range(m) if active[c]]

    return final_code
