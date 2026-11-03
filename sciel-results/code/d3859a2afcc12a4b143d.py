import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Specific optimization for the known task: q=3, n=6, radius=1
    if not (q == 3 and n == 6 and radius == 1):
        return _generic_greedy(q, n, radius)

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # --- Precomputation ---
    N = q ** n  # 729

    # Efficient index conversion using powers of q
    powers = [q ** i for i in range(n)]

    def word_to_idx(w):
        idx = 0
        for i, val in enumerate(w):
            idx += val * powers[i]
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append((idx // powers[i]) % q)
        return w

    all_words = [idx_to_word(i) for i in range(N)]

    # Precompute balls (neighbors within Hamming distance radius)
    # For q=3, n=6, radius=1: each ball has 1 + 6*2 = 13 words
    balls = [None] * N
    balls_set = [None] * N
    for i, w in enumerate(all_words):
        ball = [i]  # self
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    # Efficient update
                    new_idx = i + (new_val - orig_val) * powers[pos]
                    ball.append(new_idx)
        balls[i] = ball
        balls_set[i] = set(ball)

    # Precompute coverers: for each word, list of codewords that cover it
    # coverers[i] = list of cw indices j such that i is in balls[j]
    coverers = [None] * N
    for i in range(N):
        coverers[i] = []
    for cw in range(N):
        for neighbor in balls[cw]:
            coverers[neighbor].append(cw)

    # --- Initial Solution: Tetracode construction ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            cw = h + list(tail)
            initial_codewords.append(word_to_idx(tuple(cw)))

    current_code = set(initial_codewords)

    # --- Coverage Management ---
    cover_counts = np.zeros(N, dtype=np.int32)

    def add_codeword(cw_idx):
        current_code.add(cw_idx)
        for neighbor in balls[cw_idx]:
            cover_counts[neighbor] += 1

    def remove_codeword(cw_idx):
        current_code.remove(cw_idx)
        for neighbor in balls[cw_idx]:
            cover_counts[neighbor] -= 1

    # Initialize coverage
    for cw in current_code:
        for neighbor in balls[cw]:
            cover_counts[neighbor] += 1

    def greedy_reduce():
        """Remove redundant codewords until none remain."""
        removed = True
        while removed:
            removed = False
            for cw in list(current_code):
                # Check if redundant: all neighbors have count > 1
                redundant = True
                for neighbor in balls[cw]:
                    if cover_counts[neighbor] <= 1:
                        redundant = False
                        break
                if redundant:
                    remove_codeword(cw)
                    removed = True

    greedy_reduce()

    best_size = len(current_code)
    best_code_snapshot = list(current_code)

    if best_size == 73:
        return [idx_to_word(idx) for idx in best_code_snapshot]

    # --- Local Search with Swaps ---
    # Strategy: try to swap a non-redundant codeword with another that covers
    # the uncovered words, enabling further reduction.

    # Precompute for faster candidate selection
    # For each word, the set of codewords covering it is coverers[word]

    temperature = 1.0
    cooling = 0.995
    min_temp = 0.05

    # Track which codewords are in current_code for fast lookup
    in_code = np.zeros(N, dtype=bool)
    for cw in current_code:
        in_code[cw] = True

    def mark_in_code(cw):
        in_code[cw] = True

    def mark_out_code(cw):
        in_code[cw] = False

    # Update in_code for initial state
    for cw in current_code:
        in_code[cw] = True

    while time.time() < deadline:
        greedy_reduce()

        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code_snapshot = list(current_code)
            if best_size == 73:
                break

        if len(current_code) <= 73:
            break

        # Pick a random codeword to try to remove/swap
        cw_out = random.choice(list(current_code))

        # Check if it's redundant
        is_red = True
        for neighbor in balls[cw_out]:
            if cover_counts[neighbor] <= 1:
                is_red = False
                break

        if is_red:
            remove_codeword(cw_out)
            mark_out_code(cw_out)
            continue

        # It is NOT redundant. Remove it temporarily.
        remove_codeword(cw_out)
        mark_out_code(cw_out)

        # Find uncovered words
        uncovered = []
        for neighbor in balls[cw_out]:
            if cover_counts[neighbor] == 0:
                uncovered.append(neighbor)

        if not uncovered:
            # Actually redundant, keep removed
            continue

        # Find candidate cw_in that covers ALL uncovered words
        # Use intersection of coverers for uncovered words
        # Start with coverers of first uncovered word
        candidates = set(coverers[uncovered[0]])
        for u in uncovered[1:]:
            candidates &= set(coverers[u])
            if not candidates:
                break

        # Filter out codewords already in current_code
        valid_candidates = [c for c in candidates if not in_code[c]]

        if valid_candidates:
            # Pick a candidate (random or best)
            cw_in = random.choice(valid_candidates)

            # Add it
            add_codeword(cw_in)
            mark_in_code(cw_in)

            # Try greedy reduce to see if we can shrink
            greedy_reduce()

            # Check if we improved
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_snapshot = list(current_code)
            elif len(current_code) <= 73:
                best_size = len(current_code)
                best_code_snapshot = list(current_code)
                break
        else:
            # No valid swap, revert
            add_codeword(cw_out)
            mark_in_code(cw_out)

        temperature *= cooling
        if temperature < min_temp:
            temperature = 1.0  # Reset to explore again

    # Reconstruct best code
    best_words = []
    for idx in best_code_snapshot:
        best_words.append(idx_to_word(idx))

    return best_words


def _generic_greedy(q, n, radius):
    """Fallback greedy set cover for other parameter settings."""
    N = q ** n
    powers = [q ** i for i in range(n)]

    def word_to_idx(w):
        idx = 0
        for i, val in enumerate(w):
            idx += val * powers[i]
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append((idx // powers[i]) % q)
        return w

    all_words = [idx_to_word(i) for i in range(N)]

    # Compute balls
    balls = []
    for w in all_words:
        ball = set()
        for delta in range(radius + 1):
            if delta == 0:
                ball.add(word_to_idx(w))
            else:
                for positions in itertools.combinations(range(n), delta):
                    for new_vals in itertools.product(range(q), repeat=delta):
                        new_w = list(w)
                        for pos, val in zip(positions, new_vals):
                            new_w[pos] = val
                        ball.add(word_to_idx(tuple(new_w)))
        balls.append(ball)

    # Greedy set cover
    uncovered = set(range(N))
    codewords = []

    while uncovered:
        # Find codeword covering most uncovered words
        best_cw = -1
        best_count = -1
        for cw in range(N):
            count = len(balls[cw] & uncovered)
            if count > best_count:
                best_count = count
                best_cw = cw
        if best_count <= 0:
            break
        codewords.append(best_cw)
        uncovered -= balls[best_cw]

    return [idx_to_word(idx) for idx in codewords]
