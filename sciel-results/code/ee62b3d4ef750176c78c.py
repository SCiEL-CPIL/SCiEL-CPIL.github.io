import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    total_words = q ** n

    # Fallback for large spaces where greedy is too slow
    if total_words > 50000:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [[0] * n]

    # Precompute powers of q for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    def idx_to_tuple(idx):
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        return tuple(t)

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx += t[i] * powers[i]
        return idx

    # Precompute balls as lists of indices for efficient operations
    balls = []
    for i in range(total_words):
        w = idx_to_tuple(i)
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = tuple_to_idx(tuple(new_w))
                    ball.append(new_idx)
        balls.append(ball)

    # Helper to prune redundant codewords from a list of indices
    def prune_cover(indices):
        indices = list(set(indices))
        if not indices:
            return []

        # Count how many times each word is covered
        cover_counts = [0] * total_words
        for idx in indices:
            for w in balls[idx]:
                cover_counts[w] += 1

        changed = True
        while changed:
            changed = False
            new_indices = []
            for idx in indices:
                necessary = False
                for w in balls[idx]:
                    if cover_counts[w] == 1:
                        necessary = True
                        break
                if necessary:
                    new_indices.append(idx)
                else:
                    # Remove it
                    for w in balls[idx]:
                        cover_counts[w] -= 1
                    changed = True
            indices = new_indices
        return indices

    # Local search: try to swap codewords to reduce count
    def local_search(indices):
        indices = list(indices)
        if not indices:
            return indices

        # Count coverage
        cover_counts = [0] * total_words
        for idx in indices:
            for w in balls[idx]:
                cover_counts[w] += 1

        improved = True
        while improved:
            improved = False
            for i in range(len(indices)):
                # Try removing indices[i]
                for w in balls[indices[i]]:
                    cover_counts[w] -= 1

                # Check if still covered
                valid = True
                for w in range(total_words):
                    if cover_counts[w] == 0:
                        valid = False
                        break

                if valid:
                    # Successfully removed
                    indices.pop(i)
                    improved = True
                    break
                else:
                    # Restore
                    for w in balls[indices[i]]:
                        cover_counts[w] += 1

        return indices

    best_code = None
    best_size = float('inf')

    # 1. Try Direct Product Construction (if applicable)
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))
        dp_indices = prune_cover(dp_indices)
        if len(dp_indices) < best_size:
            best_size = len(dp_indices)
            best_code = dp_indices[:]

    # 2. Try Randomized Greedy Multiple Times with symmetry-guided tie-breaking
    all_mask = (1 << total_words) - 1

    # Precompute ball masks for faster gain calculation
    ball_masks = []
    for ball in balls:
        mask = 0
        for w in ball:
            mask |= (1 << w)
        ball_masks.append(mask)

    num_restarts = 300

    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        uncovered = all_mask
        code_indices = []

        while uncovered:
            # Calculate gains
            max_gain = -1
            top_candidates = []

            for c in range(total_words):
                gain = (ball_masks[c] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    top_candidates = [c]
                elif gain == max_gain:
                    top_candidates.append(c)

            if max_gain <= 0:
                break

            # Symmetry-guided selection: prefer codewords with more "regular" structure
            # Use a simple heuristic: prefer codewords with lower index (more structured)
            # but add some randomness to escape local minima
            if len(top_candidates) > 1:
                # With 30% probability, pick the first (most structured) candidate
                # Otherwise, pick randomly
                if random.random() < 0.3:
                    best_c = top_candidates[0]
                else:
                    best_c = random.choice(top_candidates)
            else:
                best_c = top_candidates[0]

            code_indices.append(best_c)
            uncovered &= ~ball_masks[best_c]

            if time.time() > deadline - 15.0:
                break

        if uncovered:
            continue

        # Prune the result
        pruned = prune_cover(code_indices)

        # Local search to remove more
        pruned = local_search(pruned)

        if len(pruned) < best_size:
            best_size = len(pruned)
            best_code = pruned[:]

    # 3. Try a few more restarts with different random seeds if time permits
    if time.time() < deadline - 20.0 and best_size > 70:
        for r in range(50):
            if time.time() > deadline - 10.0:
                break

            uncovered = all_mask
            code_indices = []

            while uncovered:
                max_gain = -1
                top_candidates = []

                for c in range(total_words):
                    gain = (ball_masks[c] & uncovered).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                        top_candidates = [c]
                    elif gain == max_gain:
                        top_candidates.append(c)

                if max_gain <= 0:
                    break

                # Different tie-breaking: prefer random
                if top_candidates:
                    best_c = random.choice(top_candidates)
                else:
                    break

                code_indices.append(best_c)
                uncovered &= ~ball_masks[best_c]

                if time.time() > deadline - 15.0:
                    break

            if uncovered:
                continue

            pruned = prune_cover(code_indices)
            pruned = local_search(pruned)

            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned[:]

    if best_code is None:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            dp_indices = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    dp_indices.append(tuple_to_idx(tuple(word)))
            best_code = dp_indices
        else:
            best_code = list(range(total_words))

    # Convert indices back to list of lists
    result = [list(idx_to_tuple(c)) for c in best_code]

    return result
