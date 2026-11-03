import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code using greedy set cover, trimming, and
    simulated annealing local search with swap moves.
    Uses inverse-index (reverse ball) for O(1) candidate enumeration.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    total_words = q ** n

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute balls using a tight Python loop
    balls_list = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d, q))
            d[pos] = orig
        balls_list[i] = neighbors

    balls = np.array(balls_list, dtype=np.int32)

    # Precompute inverse index: for each word, list of centers whose ball contains it
    # This enables O(1) candidate enumeration for covering a set of unique words
    inv_ball = [None] * total_words
    for c in range(total_words):
        for w in balls_list[c]:
            if inv_ball[w] is None:
                inv_ball[w] = []
            inv_ball[w].append(c)

    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=np.int32)

    while np.any(uncovered) and time.time() < deadline:
        gains = uncovered[balls].sum(axis=1)
        best_center = int(np.argmax(gains))
        max_gain = gains[best_center]

        if max_gain == 0:
            break

        code.append(best_center)
        uncovered[balls[best_center]] = False
        cover_count[balls[best_center]] += 1

    # Trimming: remove redundant codewords
    def trim_code():
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.any(cover_count[balls[c]] == 1):
                    continue
                to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True

    trim_code()

    # Simulated annealing local search with inverse-index candidate enumeration
    def get_unique_words(c):
        return balls[c][cover_count[balls[c]] == 1]

    def local_search(seed, max_rounds=200):
        nonlocal code, cover_count
        random.seed(seed)
        code_set = set(code)
        
        improved = True
        swap_round = 0
        temp = 1.0  # temperature for simulated annealing

        while improved and time.time() < deadline and swap_round < max_rounds:
            improved = False
            swap_round += 1
            temp *= 0.99  # cool down

            # Try removing each codeword first
            for ci in range(len(code) - 1, -1, -1):
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    code.pop(ci)
                    cover_count[balls[i]] -= 1
                    code_set.discard(i)
                    improved = True
                    break

            if improved:
                continue

            # Try swaps using inverse-index for fast candidate enumeration
            code_indices = list(range(len(code)))
            random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    continue

                # Use inverse index to find all centers that cover ALL unique words
                # Start with the first unique word's candidates, intersect with others
                unique_list = unique.tolist()
                if len(unique_list) == 0:
                    continue

                # Get candidates from the first unique word
                candidate_set = set(inv_ball[unique_list[0]])
                
                # Intersect with candidates from remaining unique words
                for w in unique_list[1:]:
                    candidate_set &= set(inv_ball[w])
                    if not candidate_set:
                        break

                # Remove current code words from candidates
                candidate_set -= code_set

                if not candidate_set:
                    continue

                # Try a limited number of candidates (shuffled for diversity)
                candidates = list(candidate_set)
                random.shuffle(candidates)
                n_candidates = min(50, len(candidates))
                candidates = candidates[:n_candidates]

                best_swap = None
                best_gain = -1

                for j in candidates:
                    if time.time() > deadline:
                        return
                    # Calculate gain: how many words does j uniquely cover after swap?
                    # After removing i and adding j, j's unique coverage = 
                    # words in ball[j] that are covered only by i and j (i.e., cover_count == 2 
                    # where one is i) plus words covered only by j (cover_count == 1, but j not in code yet)
                    # More precisely: after swap, for each word w in ball[j]:
                    #   new_cover_count = cover_count[w] - (1 if w in ball[i] else 0) + 1
                    #   j is unique for w if new_cover_count == 1
                    # Since j is not currently in code, cover_count[w] doesn't include j.
                    # After adding j: cover_count[w] + 1 - (1 if w in ball[i] else 0)
                    # j unique iff cover_count[w] + 1 - (1 if w in ball[i] else 0) == 1
                    # i.e., cover_count[w] - (1 if w in ball[i] else 0) == 0
                    # i.e., cover_count[w] == (1 if w in ball[i] else 0)
                    
                    # Faster: compute gain = number of words w in ball[j] where
                    # cover_count[w] == (1 if w in ball[i] else 0)
                    # This means: w is covered by exactly the codewords in ball[i] ∩ ball[j] 
                    # that are currently in code, plus j after addition.
                    
                    # Let's use numpy for speed:
                    bj = balls[j]
                    bi = balls[i]
                    # For each word in bj, check if its cover_count equals 
                    # (1 if it's in bi else 0)
                    in_bi = np.zeros(total_words, dtype=bool)
                    in_bi[bi] = True
                    # gain = sum over w in bj of [cover_count[w] == in_bi[w]]
                    gain = int(np.sum(cover_count[bj] == in_bi[bj]))
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = j

                if best_swap is not None:
                    # Accept swap: always accept if gain >= 0 (it's neutral or better)
                    # The swap is valid because all unique words of i are covered by j
                    j = best_swap
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    break

    # Run local search with different seeds
    for seed in [42, 123, 456, 789, 1024]:
        if time.time() < deadline:
            local_search(seed, max_rounds=150)
            trim_code()

    # Final trimming
    trim_code()

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result
