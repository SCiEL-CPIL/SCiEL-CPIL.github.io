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

    # Precompute all balls as numpy arrays of shape (total_words, ball_size)
    # For q=3, n=8, radius=1: ball_size = 1 + 8*2 = 17
    # total_words = 6561, so array is 6561 x 17 = ~111K integers - fine

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

    # Precompute balls using vectorized approach
    # For each center c, its ball contains c and all words at Hamming distance 1
    # We can compute this efficiently:
    # For position p and new value v (v != c[p]), the neighbor is c with digit p changed to v

    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    balls[:, 0] = np.arange(total_words, dtype=np.int32)

    # Precompute the digit values for each position
    # digits[i][p] = (i // (q**(n-1-p))) % q
    powers = [q ** (n - 1 - p) for p in range(n)]

    # For each position p, and for each possible change, compute the neighbor
    # neighbor = c - c[p]*powers[p] + new_val*powers[p]
    col = 1
    for p in range(n):
        pw = powers[p]
        # Get digit at position p for all centers
        digits_p = (np.arange(total_words, dtype=np.int64) // pw) % q
        for new_val in range(q):
            # Only include if new_val != digits_p
            mask = digits_p != new_val
            if not np.any(mask):
                continue
            # For centers where mask is true, the neighbor is c - digit*pw + new_val*pw
            # But we need to fill column col only for those centers
            # This is tricky with columnar storage. Let's use a different approach.
            pass

    # Alternative: build balls row by row but vectorize the inner loop
    # Actually, for 6561 rows and 17 cols, even a loop is fast enough.
    # Let's use a smarter vectorized approach:

    # For each position p, the change from digit d to digit d' shifts by (d'-d)*pw
    # We can precompute for each center and each position the two neighbors (for q=3)
    # But let's just do it cleanly:

    centers = np.arange(total_words, dtype=np.int64)
    balls_flat = np.zeros((total_words, ball_size), dtype=np.int32)
    balls_flat[:, 0] = np.arange(total_words, dtype=np.int32)

    col = 1
    for p in range(n):
        pw = powers[p]
        digits_p = (centers // pw) % q
        for new_val in range(q):
            mask = digits_p != new_val
            if not np.any(mask):
                continue
            # Compute neighbor indices for masked centers
            neighbors = centers[mask] - digits_p[mask] * pw + new_val * pw
            # We need to place these in column 'col' for the masked rows
            # This requires scattered assignment which is slow in numpy
            # Let's just do a simple loop for the final assembly
            pass

    # Given the complexity of vectorized scattered assignment, let's use a hybrid:
    # Precompute the offset table: for each center and each (position, new_val), the neighbor
    # Actually, let's just build it with a tight Python loop - 6561*16 = ~105K operations, fast

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

    # Convert to numpy array
    balls = np.array(balls_list, dtype=np.int32)

    # Greedy Set Cover using numpy for speed
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=np.int32)

    while np.any(uncovered):
        if time.time() > deadline:
            break

        # Find the center that covers the most uncovered words
        # gain[i] = number of neighbors of i that are uncovered
        # We can compute this as: for each center i, sum of uncovered[balls[i]]
        # This is: uncovered[balls].sum(axis=1)
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
            # Check if any word in balls[c] has cover_count == 1
            if np.any(cover_count[balls[c]] == 1):
                continue
            to_remove.append(i)

        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                c = code[idx]
                code.pop(idx)
                cover_count[balls[c]] -= 1
            changed = True

    # Randomized local search: swap codewords to improve
    # Objective: minimize |code| while maintaining coverage
    # Swap: remove codeword i, add codeword j (not in code)
    # A swap is valid if after removal, all words still covered, and j covers some newly needed words
    # Actually, a swap that reduces size: remove i, don't add j (that's removal)
    # A swap that maintains size: remove i, add j - only useful if it enables future removals
    # Better: try removals first (done above), then try swaps that might enable further removals

    # Let's do a more targeted local search:
    # For each codeword i, try replacing it with a candidate j not in code
    # Check if the new set still covers everything
    # If yes, accept if it leads to more redundancy (more words with cover_count=1 elsewhere)

    # Actually, the most effective local search for covering codes:
    # 1. Try to remove each codeword (done)
    # 2. Try to swap: for each codeword i, for each candidate j not in code,
    #    check if replacing i with j still covers all words
    #    Accept the swap if it results in a better "redundancy profile"

    # Given time constraints, let's do a limited number of swap attempts
    code_set = set(code)
    candidates = list(set(range(total_words)) - code_set)

    # Precompute for faster checking
    # For a swap (remove i, add j):
    #   words that lose coverage: balls[i] where cover_count == 1 (only covered by i)
    #   words that gain coverage: balls[j] where cover_count == 0 (not covered at all) - but we need full coverage
    #   Actually, after removing i, some words might become uncovered (cover_count becomes 0)
    #   After adding j, those must be covered by j
    #   So the swap is valid iff: {w in balls[i] : cover_count[w]==1} ⊆ balls[j]

    swap_attempts = 0
    max_swap_attempts = 50000
    np.random.seed(42)

    improved = True
    while improved and time.time() < deadline and swap_attempts < max_swap_attempts:
        improved = False
        # Shuffle code order for randomization
        code_indices = list(range(len(code)))
        np.random.shuffle(code_indices)

        for ci in code_indices:
            if time.time() > deadline:
                break
            i = code[ci]
            # Words only covered by i
            only_by_i = balls[i][cover_count[balls[i]] == 1]
            if len(only_by_i) == 0:
                # Can remove i (should have been caught by trimming, but double-check)
                code.pop(ci)
                cover_count[balls[i]] -= 1
                code_set.discard(i)
                improved = True
                break

            # Try to find a candidate j that covers all of only_by_i
            # and is not already in code
            # Sample some candidates
            n_samples = min(50, len(candidates))
            if n_samples == 0:
                break
            sample_idx = np.random.choice(len(candidates), size=n_samples, replace=False)
            for si in sample_idx:
                swap_attempts += 1
                j = candidates[si]
                if j in code_set:
                    continue
                # Check if j covers all of only_by_i
                j_ball = balls[j]
                # Check if only_by_i is subset of j_ball
                # Use set intersection or boolean mask
                # For small sets, use set
                if len(only_by_i) <= 20:
                    j_set = set(j_ball.tolist())
                    if all(w in j_set for w in only_by_i):
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

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result
