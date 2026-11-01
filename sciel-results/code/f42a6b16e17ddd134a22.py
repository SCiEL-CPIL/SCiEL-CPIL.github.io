<<<<<<< SEARCH
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
=======
    # 2. Try Randomized Greedy Multiple Times with symmetry-guided tie-breaking
    # Use numpy for vectorized gain computation
    num_restarts = 300

    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        uncovered = np.ones(total_words, dtype=np.int32)
        code_indices = []

        while uncovered.any():
            # Vectorized gain calculation: for each candidate, count how many uncovered words it covers
            # ball_arrays shape: (total_words, max_ball_size)
            # We need: for each c, sum over ball of (uncovered[w] for w in ball[c])
            # Use matrix multiplication with masking
            max_bs = int(ball_sizes.max())
            gains = np.zeros(total_words, dtype=np.int32)
            for c in range(total_words):
                bs = ball_sizes[c]
                if bs > 0:
                    gains[c] = int(np.sum(uncovered[ball_arrays[c, :bs]]))

            max_gain = int(gains.max())
            if max_gain <= 0:
                break

            # Get all candidates with max gain
            top_candidates = list(np.where(gains == max_gain)[0])

            # Symmetry-guided selection: prefer codewords with more "regular" structure
            # Use a simple heuristic: prefer codewords with lower index (more structured)
            # but add some randomness to escape local minima
            if len(top_candidates) > 1:
                if random.random() < 0.3:
                    best_c = top_candidates[0]
                else:
                    best_c = random.choice(top_candidates)
            else:
                best_c = top_candidates[0]

            code_indices.append(best_c)
            uncovered[ball_arrays[best_c, :ball_sizes[best_c]]] = 0

            if time.time() > deadline - 15.0:
                break

        if uncovered.any():
            continue

        # Prune the result
        pruned = prune_cover(code_indices)

        # Local search to remove more
        pruned = local_search(pruned)

        if len(pruned) < best_size:
            best_size = len(pruned)
            best_code = pruned[:]
>>>>>>> REPLACE

