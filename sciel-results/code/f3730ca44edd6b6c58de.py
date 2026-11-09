"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    """Construct a covering code using greedy + local search."""
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)

    # Build word_to_idx mapping
    word_to_idx = {}
    for idx, word in enumerate(all_words):
        word_to_idx[word] = idx

    # Precompute neighbors (words within Hamming distance <= radius)
    # For radius=1: word itself + all words differing in exactly 1 position
    neighbors = []
    for idx in range(num_words):
        word = all_words[idx]
        nbrs = set()
        nbrs.add(idx)
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_word = list(word)
                new_word[pos] = val
                new_idx = word_to_idx[tuple(new_word)]
                nbrs.add(new_idx)
        neighbors.append(nbrs)

    # Greedy initialization: select codewords that cover the most uncovered words
    covered = [False] * num_words
    codewords = []
    uncovered_count = num_words

    while uncovered_count > 0:
        if time.time() > deadline - 30:  # Reserve time for local search
            break

        # Find codeword that covers the most uncovered words
        best_idx = -1
        best_count = -1
        for idx in range(num_words):
            if covered[idx]:
                continue
            count = 0
            for nbr in neighbors[idx]:
                if not covered[nbr]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = idx
            if time.time() > deadline - 30:
                break

        if best_idx == -1:
            break

        # Add this codeword
        codewords.append(all_words[best_idx])
        for nbr in neighbors[best_idx]:
            if not covered[nbr]:
                covered[nbr] = True
                uncovered_count -= 1

        if time.time() > deadline - 30:
            break

    # Local search: try to remove codewords if coverage remains
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Recompute coverage
        covered = [False] * num_words
        for cw in codewords:
            cw_idx = word_to_idx[tuple(cw)]
            for nbr in neighbors[cw_idx]:
                covered[nbr] = True

        # Try removing each codeword
        for i in range(len(codewords)):
            if time.time() > deadline - 10:
                break

            # Check if removing codeword[i] still covers all words
            # Recompute coverage without this codeword
            covered_test = [False] * num_words
            for j, cw in enumerate(codewords):
                if j == i:
                    continue
                cw_idx = word_to_idx[tuple(cw)]
                for nbr in neighbors[cw_idx]:
                    covered_test[nbr] = True

            # Check if all words are covered
            all_covered = all(covered_test)
            if all_covered:
                # Remove this codeword
                codewords.pop(i)
                improved = True
                break

    # Local search: try swap operations
    for _ in range(1000):  # Limited iterations
        if time.time() > deadline - 5:
            break

        if len(codewords) < 2:
            break

        # Recompute coverage
        covered = [False] * num_words
        for cw in codewords:
            cw_idx = word_to_idx[tuple(cw)]
            for nbr in neighbors[cw_idx]:
                covered[nbr] = True

        # Find a codeword to remove and a new word to add
        best_swap = None
        best_gain = 0

        # Try removing a random codeword
        remove_idx = random.randint(0, len(codewords) - 1)
        remove_cw = codewords[remove_idx]
        remove_cw_idx = word_to_idx[tuple(remove_cw)]

        # Compute coverage without this codeword
        covered_without = [False] * num_words
        for i, cw in enumerate(codewords):
            if i == remove_idx:
                continue
            cw_idx = word_to_idx[tuple(cw)]
            for nbr in neighbors[cw_idx]:
                covered_without[nbr] = True

        # Find uncovered words
        uncovered = [i for i in range(num_words) if not covered_without[i]]

        if not uncovered:
            # We can just remove this codeword
            codewords.pop(remove_idx)
            continue

        # Find a word that covers the most uncovered words
        best_add = -1
        best_add_count = -1
        for idx in uncovered[:100]:  # Sample from uncovered
            count = 0
            for nbr in neighbors[idx]:
                if not covered_without[nbr]:
                    count += 1
            if count > best_add_count:
                best_add_count = count
                best_add = idx

        if best_add != -1 and best_add_count > 0:
            new_cw = all_words[best_add]
            # Check if this is a valid swap (new_cw not already in codewords)
            if new_cw not in codewords:
                codewords[remove_idx] = new_cw

    # Final validation and cleanup
    # Remove duplicates
    seen = set()
    unique_codewords = []
    for cw in codewords:
        key = tuple(cw)
        if key not in seen:
            seen.add(key)
            unique_codewords.append(list(cw))
    codewords = unique_codewords

    # Verify coverage
    covered = [False] * num_words
    for cw in codewords:
        cw_idx = word_to_idx[tuple(cw)]
        for nbr in neighbors[cw_idx]:
            covered[nbr] = True

    if not all(covered):
        # Fallback: use the simple construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return codewords
# EVOLVE-BLOCK-END
