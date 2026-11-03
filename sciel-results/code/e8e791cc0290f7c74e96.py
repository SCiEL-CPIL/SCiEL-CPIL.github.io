"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Total number of words in the space
    total_words = q ** n
    if total_words == 0:
        return []

    # If radius >= n, one word covers everything
    if radius >= n:
        return [[0] * n]

    # Precompute all words as a flat array of indices 0..total_words-1
    # Word i in base-q representation
    # We'll work with flat indices for efficiency

    # Ball of radius 1 around a word: the word itself plus n*(q-1) neighbors
    ball_size = 1 + n * (q - 1)

    # Strategy: Greedy set cover with removal
    # 1. Build initial cover greedily
    # 2. Try to remove redundant codewords
    # 3. Try to swap codewords to reduce count

    # Map from flat index to word (list of digits)
    def idx_to_word(idx):
        word = [0] * n
        for pos in range(n - 1, -1, -1):
            word[pos] = idx % q
            idx //= q
        return word

    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    # Precompute ball membership: for each word, which codeword indices cover it
    # We'll track coverage counts

    # Initial greedy construction
    covered = np.zeros(total_words, dtype=np.uint8)  # 1 if covered
    codewords = []  # list of flat indices
    code_to_idx = {}  # flat_idx -> position in codewords list

    # Precompute all words as numpy array of shape (total_words, n) for neighbor computation
    # For memory efficiency with q=3, n=6: 729*6 = 4374 ints, fine
    all_words = np.zeros((total_words, n), dtype=np.int8)
    for i in range(total_words):
        w = idx_to_word(i)
        all_words[i] = w

    # For each word, precompute its ball (all words within Hamming distance radius)
    # For radius=1: the word + all single-position changes
    balls = []
    for i in range(total_words):
        ball = [i]
        if radius >= 1:
            w = all_words[i]
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_w = w.copy()
                    new_w[pos] = val
                    ball.append(word_to_idx(new_w))
        balls.append(ball)

    # Greedy: repeatedly pick the word that covers the most uncovered words
    # For efficiency, we'll iterate
    uncovered_count = total_words
    while uncovered_count > 0:
        if time.time() > deadline:
            break

        # Find the word with max coverage of uncovered words
        best_idx = -1
        best_count = 0
        for i in range(total_words):
            if covered[i]:
                continue
            cnt = 0
            for j in balls[i]:
                if not covered[j]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
            if time.time() > deadline:
                break

        if best_idx == -1:
            break

        # Add best_idx to codewords
        code_to_idx[best_idx] = len(codewords)
        codewords.append(best_idx)

        # Mark covered
        for j in balls[best_idx]:
            if not covered[j]:
                covered[j] = 1
                uncovered_count -= 1

    # Now try to remove redundant codewords
    # A codeword is redundant if removing it doesn't uncover any word
    # For each codeword, check if all words in its ball are covered by other codewords

    def is_redundant(code_pos):
        """Check if codeword at position code_pos is redundant."""
        ci = codewords[code_pos]
        for j in balls[ci]:
            # Check if j is covered by some other codeword
            found = False
            for other_pos, other_ci in enumerate(codewords):
                if other_pos == code_pos:
                    continue
                if j in balls[other_ci]:
                    found = True
                    break
            if not found:
                return False
        return True

    # Better: track coverage counts
    # coverage[j] = number of codewords whose ball contains j
    coverage = np.zeros(total_words, dtype=np.int32)
    for ci in codewords:
        for j in balls[ci]:
            coverage[j] += 1

    removed = True
    while removed:
        if time.time() > deadline:
            break
        removed = False
        for pos in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            ci = codewords[pos]
            # Check if removing this codeword leaves any word uncovered
            redundant = True
            for j in balls[ci]:
                if coverage[j] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for j in balls[ci]:
                    coverage[j] -= 1
                codewords.pop(pos)
                removed = True

    # Try to merge: if two codewords' union of balls is covered by remaining, etc.
    # Actually, let's try a different improvement: local search / simulated annealing
    # Try replacing a codeword with another to potentially allow removal of others

    # Simple local improvement: try removing each codeword and see if we can
    # re-cover any exposed words with a single new codeword that also covers
    # words from another codeword, allowing that other to be removed.
    # This is complex; let's stick with the greedy + removal for now.

    # Final: convert back to list of lists
    result = [idx_to_word(ci) for ci in codewords]

    # Ensure distinctness (should be guaranteed)
    seen = set()
    unique_result = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            unique_result.append(w)

    return unique_result
# EVOLVE-BLOCK-END
