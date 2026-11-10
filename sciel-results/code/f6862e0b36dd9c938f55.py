"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n > 4:
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    elif n == 4:
        code = [h[:] for h in tetracode]
    else:
        # For n < 4, use all words (brute force)
        code = [list(w) for w in itertools.product(range(q), repeat=n)]
        return code

    # Precompute powers for fast encoding
    q_powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    total_words = q ** n

    def word_to_int(w):
        return int(np.dot(np.array(w, dtype=np.int64), q_powers))

    def int_to_word(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    # Convert code to integer representations
    code_ints = np.array([word_to_int(c) for c in code], dtype=np.int64)
    m = len(code_ints)

    # Precompute neighbor lists for each codeword
    # For radius=1: each codeword covers 1 + n*(q-1) words
    all_neighbors = []
    for ci in code_ints:
        word = int_to_word(int(ci))
        neighbors = [int(ci)]  # distance 0
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    word[pos] = new_val
                    neighbors.append(word_to_int(word))
                word[pos] = orig
        all_neighbors.append(np.array(neighbors, dtype=np.int64))

    # Build coverage counts
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for neighbors in all_neighbors:
        coverage_count[neighbors] += 1

    # Greedy removal with improved heuristic:
    # Sort candidates by "redundancy score" = number of neighbors with coverage > 1
    # Higher score = safer to remove
    removed = np.zeros(m, dtype=bool)
    improvement_found = True
    while improvement_found and time.time() < deadline:
        improvement_found = False

        # Compute redundancy score for each non-removed codeword
        # Score = number of neighbors that have coverage > 1
        scores = np.zeros(m, dtype=np.int32)
        for idx in range(m):
            if removed[idx]:
                continue
            neighbors = all_neighbors[idx]
            # Count neighbors with coverage > 1
            scores[idx] = np.sum(coverage_count[neighbors] > 1)

        # Try removing in order of highest score first
        indices = np.argsort(-scores)
        for idx in indices:
            if removed[idx] or time.time() > deadline:
                continue
            neighbors = all_neighbors[idx]
            # Check if removing this codeword leaves any word with coverage 0
            if np.any(coverage_count[neighbors] == 1):
                continue
            # Remove this codeword
            removed[idx] = True
            coverage_count[neighbors] -= 1
            improvement_found = True

    # Second pass: try random order removal for any remaining redundant codewords
    for _ in range(3):
        if time.time() > deadline:
            break
        improvement_found = False
        indices = np.random.permutation(m)
        for idx in indices:
            if removed[idx]:
                continue
            neighbors = all_neighbors[idx]
            if np.any(coverage_count[neighbors] == 1):
                continue
            removed[idx] = True
            coverage_count[neighbors] -= 1
            improvement_found = True
        if not improvement_found:
            break

    # Build final code
    final_code = [code[i] for i in range(m) if not removed[i]]

    return final_code
# EVOLVE-BLOCK-END
