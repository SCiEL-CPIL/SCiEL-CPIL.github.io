"""Rewritten program for the suite task; improved greedy set-cover with time guard."""
# EVOLVE-BLOCK-START
import time
import itertools
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # For the specific task parameters (q=3, n=6, radius=1) we know the answer is 73.
    # We use a greedy set-cover approach to find a covering of size as close to 73 as possible.
    
    # Generate all words in the space
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = q ** n
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (words within Hamming distance <= radius) for each word
    # For radius=1, each word has 1 + n*(q-1) neighbors
    neighbors = [None] * total_words
    for i, w in enumerate(all_words):
        covered = set()
        covered.add(i)
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    covered.add(idx)
        if radius >= 2:
            # For radius 2, add all pairs of positions
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1, orig2 = w[pos1], w[pos2]
                    for val1 in range(q):
                        for val2 in range(q):
                            if val1 == orig1 and val2 == orig2:
                                continue
                            if val1 == orig1 or val2 == orig2:
                                continue  # Already counted in radius 1
                            new_w = list(w)
                            new_w[pos1] = val1
                            new_w[pos2] = val2
                            idx = word_to_idx[tuple(new_w)]
                            covered.add(idx)
        neighbors[i] = covered
    
    # Greedy set cover: repeatedly pick the center covering most uncovered words
    uncovered = set(range(total_words))
    codewords = []
    
    # To speed up the greedy selection, we can use a priority-queue-like approach
    # but for 729 words, a simple scan is fast enough.
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        # Randomize the order slightly to avoid deterministic pitfalls, but keep it reproducible
        # Actually, for covering codes, greedy is quite stable. Let's just scan.
        for i in range(total_words):
            if not uncovered:
                break
            # Count how many uncovered words are in the neighborhood of i
            count = 0
            for j in neighbors[i]:
                if j in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
                if count == len(uncovered):
                    break
        if best_idx == -1:
            break
        codewords.append(list(all_words[best_idx]))
        uncovered -= neighbors[best_idx]
    
    # If we timed out and haven't covered everything, fall back to a valid construction
    if uncovered:
        # Direct product construction as fallback
        if n >= 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            fallback = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
            return fallback
        else:
            return [list(w) for w in all_words]
    
    return codewords
# EVOLVE-BLOCK-END
