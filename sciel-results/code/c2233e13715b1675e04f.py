"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

# Deadline: evaluator kills at 150s wall-clock. We target returning well before that.
_DEADLINE_SECONDS = 135.0


def _hamming_distance(a, b):
    """Hamming distance between two lists/tuples of equal length."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _generate_neighbors(word, q, n):
    """Yield all words within Hamming distance <= radius (here radius=1) from word.
    We only need distance 0 and 1 for radius=1."""
    # distance 0
    yield tuple(word)
    # distance 1
    for i in range(n):
        orig = word[i]
        for v in range(q):
            if v == orig:
                continue
            new_word = list(word)
            new_word[i] = v
            yield tuple(new_word)


def _build_initial_code(q, n, radius, rng):
    """
    Start with a random linear code of dimension 4 over GF(3) (81 codewords)
    if q == 3 and n >= 4. Otherwise fall back to a few random words.
    """
    if q == 3 and n >= 4:
        # Generate a random 4 x n matrix over GF(3)
        G = [[rng.randint(0, 2) for _ in range(n)] for _ in range(4)]
        codewords = set()
        for coeffs in itertools.product(range(3), repeat=4):
            word = [0] * n
            for i in range(4):
                c = coeffs[i]
                if c == 0:
                    continue
                for j in range(n):
                    word[j] = (word[j] + c * G[i][j]) % 3
            codewords.add(tuple(word))
        return list(codewords)
    else:
        # Fallback: a few random words
        count = min(q ** 2, q ** n)
        codewords = set()
        while len(codewords) < count:
            w = tuple(rng.randint(0, q - 1) for _ in range(n))
            codewords.add(w)
        return list(codewords)


def _greedy_cover(q, n, radius, initial_code, rng, deadline):
    """
    Greedy set-cover heuristic to expand the code until every word in {0..q-1}^n
    is within Hamming distance <= radius of some codeword.
    
    We track which words are covered. For efficiency, we represent the universe
    as a boolean array of size q^n (feasible for q=3, n=7: 2187 words).
    """
    total_words = q ** n
    # Map word tuple to index
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    # Precompute all words and their indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_idx = {w: i for i, w in enumerate(all_words)}

    # Covered array
    covered = np.zeros(total_words, dtype=bool)

    # Add initial codewords
    codewords = []
    for w in initial_code:
        idx = word_idx.get(tuple(w))
        if idx is None:
            continue
        codewords.append(tuple(w))
        # Mark all words within radius as covered
        for nb in _generate_neighbors(list(w), q, n):
            if radius >= 1:
                pass  # _generate_neighbors already yields dist 0 and 1
            nb_idx = word_idx.get(nb)
            if nb_idx is not None:
                covered[nb_idx] = True

    uncovered_count = total_words - int(covered.sum())

    # Precompute for each word, the list of words within distance <= radius
    # For radius=1, each word has 1 + n*(q-1) neighbors.
    # We'll compute on the fly to save memory, but for efficiency in greedy,
    # we can precompute the neighbor indices for each word.
    # Given total_words is small (2187), let's precompute.
    neighbor_indices = []
    for w in all_words:
        nbs = []
        for nb in _generate_neighbors(list(w), q, n):
            nb_idx = word_idx.get(nb)
            if nb_idx is not None:
                nbs.append(nb_idx)
        neighbor_indices.append(nbs)

    # Greedy: repeatedly pick the word that covers the most uncovered words
    # To speed up, we can use a heuristic: pick a random uncovered word, then
    # find the candidate codeword (any word) that covers it and maximizes new coverage.
    
    start_time = time.time()
    iteration = 0

    while uncovered_count > 0:
        if time.time() > deadline:
            break

        iteration += 1

        # Find an uncovered word
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        # Pick a random uncovered word
        target_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
        target_word = all_words[target_idx]

        # Candidate codewords: all words within distance <= radius of target_word
        # These are the only words that could cover target_word.
        candidates = neighbor_indices[target_idx]

        # For each candidate, compute how many currently uncovered words it would cover
        best_candidate = None
        best_gain = 0
        best_new_covered = None

        for cand_idx in candidates:
            # Count how many of cand_idx's neighbors are currently uncovered
            gain = 0
            new_covered_list = []
            for nb_idx in neighbor_indices[cand_idx]:
                if not covered[nb_idx]:
                    gain += 1
                    new_covered_list.append(nb_idx)
            if gain > best_gain:
                best_gain = gain
                best_candidate = cand_idx
                best_new_covered = new_covered_list

        if best_candidate is None or best_gain == 0:
            # Should not happen if uncovered_count > 0, but safety check
            break

        # Add the best candidate as a new codeword
        codewords.append(all_words[best_candidate])
        for nb_idx in best_new_covered:
            covered[nb_idx] = True
        uncovered_count -= best_gain

    return codewords


def _remove_redundant_codewords(q, n, radius, codewords, deadline):
    """
    Try to remove codewords that are redundant (i.e., removing them doesn't leave
    any word uncovered). This is a post-processing step.
    """
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbor indices
    neighbor_indices = []
    for w in all_words:
        nbs = []
        for nb in _generate_neighbors(list(w), q, n):
            nb_idx = word_idx.get(nb)
            if nb_idx is not None:
                nbs.append(nb_idx)
        neighbor_indices.append(nbs)

    # Count how many codewords cover each word
    coverage_count = np.zeros(total_words, dtype=int)
    for cw in codewords:
        cw_idx = word_idx.get(tuple(cw))
        if cw_idx is None:
            continue
        for nb_idx in neighbor_indices[cw_idx]:
            coverage_count[nb_idx] += 1

    # Try to remove codewords in random order
    rng = random.Random(42)
    indices = list(range(len(codewords)))
    rng.shuffle(indices)

    removed = set()
    for idx in indices:
        if time.time() > deadline:
            break
        if idx in removed:
            continue
        cw = codewords[idx]
        cw_idx = word_idx.get(tuple(cw))
        if cw_idx is None:
            continue
        # Check if any word is covered only by this codeword
        redundant = True
        for nb_idx in neighbor_indices[cw_idx]:
            if coverage_count[nb_idx] == 1:
                redundant = False
                break
        if redundant:
            # Remove it
            removed.add(idx)
            for nb_idx in neighbor_indices[cw_idx]:
                coverage_count[nb_idx] -= 1

    if removed:
        new_codewords = [cw for i, cw in enumerate(codewords) if i not in removed]
        return new_codewords
    return codewords


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius
    `radius`. Returns a list of codewords (each a list of n integers in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS

    rng = random.Random(12345)

    # Step 1: Generate initial code using random linear code
    initial_code = _build_initial_code(q, n, radius, rng)

    # Step 2: Greedy cover to satisfy the covering constraint
    codewords = _greedy_cover(q, n, radius, initial_code, rng, deadline)

    # Step 3: Remove redundant codewords
    if time.time() < deadline - 10:  # Leave some margin
        codewords = _remove_redundant_codewords(q, n, radius, codewords, deadline)

    # Convert tuples to lists for the return interface
    return [list(cw) for cw in codewords]

# EVOLVE-BLOCK-END
