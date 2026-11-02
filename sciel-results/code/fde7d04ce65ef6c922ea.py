import time
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using fast greedy set cover (NumPy vectorized),
    followed by removal of redundant codewords, then a randomized local search
    that swaps one codeword for another to maximize uniquely covered words.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

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

    # Precompute ball sets for fast subset checking
    ball_sets = [set(balls[i].tolist()) for i in range(total_words)]

    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=np.int32)

    while np.any(uncovered) and time.time() < deadline:
        # Find the center that covers the most uncovered words
        gains = uncovered[balls].sum(axis=1)
        best_center = int(np.argmax(gains))
        max_gain = gains[best_center]

        if max_gain == 0:
            break

        code.append(best_center)
        uncovered[balls[best_center]] = False
        cover_count[balls[best_center]] += 1

    # Trimming: remove redundant codewords
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

    # Randomized local search with swaps
    code_set = set(code)
    all_indices = set(range(total_words))
    
    # Precompute for each codeword, the words it uniquely covers
    def get_unique_words(c, cover_count):
        return balls[c][cover_count[balls[c]] == 1]

    # Local search: try swaps
    np.random.seed(42)
    max_swap_rounds = 100
    improved = True
    swap_round = 0

    while improved and time.time() < deadline and swap_round < max_swap_rounds:
        improved = False
        swap_round += 1
        
        # Try removing each codeword first
        for ci in range(len(code) - 1, -1, -1):
            if time.time() > deadline:
                break
            i = code[ci]
            unique = get_unique_words(i, cover_count)
            if len(unique) == 0:
                code.pop(ci)
                cover_count[balls[i]] -= 1
                code_set.discard(i)
                improved = True
                break

        if improved:
            continue

        # Try swaps: for each codeword, try replacing with a candidate
        code_indices = list(range(len(code)))
        np.random.shuffle(code_indices)

        for ci in code_indices:
            if time.time() > deadline:
                break
            i = code[ci]
            unique = get_unique_words(i, cover_count)
            if len(unique) == 0:
                continue

            # Find candidates that cover all unique words
            # Sample from non-code words
            non_code = list(all_indices - code_set)
            if not non_code:
                break

            # Try a limited number of candidates
            n_candidates = min(200, len(non_code))
            np.random.shuffle(non_code)
            non_code = non_code[:n_candidates]

            for j in non_code:
                if time.time() > deadline:
                    break
                # Check if j covers all unique words of i
                j_ball_set = ball_sets[j]
                if all(w in j_ball_set for w in unique):
                    # Valid swap!
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    break

            if improved:
                break

    # Final trimming pass
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

    # Second round of local search with different seed
    if time.time() < deadline:
        np.random.seed(123)
        code_set = set(code)
        all_indices = set(range(total_words))
        
        improved = True
        max_swap_rounds2 = 50
        swap_round = 0

        while improved and time.time() < deadline and swap_round < max_swap_rounds2:
            improved = False
            swap_round += 1
            
            # Try removing each codeword first
            for ci in range(len(code) - 1, -1, -1):
                if time.time() > deadline:
                    break
                i = code[ci]
                unique = get_unique_words(i, cover_count)
                if len(unique) == 0:
                    code.pop(ci)
                    cover_count[balls[i]] -= 1
                    code_set.discard(i)
                    improved = True
                    break

            if improved:
                continue

            # Try swaps
            code_indices = list(range(len(code)))
            np.random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    break
                i = code[ci]
                unique = get_unique_words(i, cover_count)
                if len(unique) == 0:
                    continue

                non_code = list(all_indices - code_set)
                if not non_code:
                    break

                n_candidates = min(100, len(non_code))
                np.random.shuffle(non_code)
                non_code = non_code[:n_candidates]

                for j in non_code:
                    if time.time() > deadline:
                        break
                    j_ball_set = ball_sets[j]
                    if all(w in j_ball_set for w in unique):
                        code[ci] = j
                        code_set.discard(i)
                        code_set.add(j)
                        cover_count[balls[i]] -= 1
                        cover_count[balls[j]] += 1
                        improved = True
                        break

                if improved:
                    break

        # Final trimming
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

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result
