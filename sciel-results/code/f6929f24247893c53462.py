import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 6 and radius == 1:
        # Precompute index conversions
        def idx_to_tuple(idx):
            t = [0] * 6
            rem = idx
            for i in range(6):
                t[i] = rem % 3
                rem //= 3
            return tuple(t)

        def tuple_to_idx(t):
            idx = 0
            pow3 = 1
            for i in range(6):
                idx += t[i] * pow3
                pow3 *= 3
            return idx

        total_words = 729

        # Precompute balls as lists (faster iteration) and sets (for membership)
        balls_list = [None] * total_words
        balls_set = [None] * total_words
        for i in range(total_words):
            w = idx_to_tuple(i)
            ball = [i]
            for pos in range(6):
                orig_val = w[pos]
                for new_val in range(3):
                    if new_val != orig_val:
                        new_w = list(w)
                        new_w[pos] = new_val
                        new_idx = tuple_to_idx(tuple(new_w))
                        ball.append(new_idx)
            balls_list[i] = ball
            balls_set[i] = set(ball)

        # Helper to check if a set of indices covers the space
        def is_cover(indices):
            covered = set()
            for idx in indices:
                covered.update(balls_list[idx])
            return len(covered) == total_words

        # Helper to prune redundant codewords
        def prune_cover(indices):
            indices = list(indices)
            cover_counts = [0] * total_words
            for idx in indices:
                for w in balls_list[idx]:
                    cover_counts[w] += 1

            changed = True
            while changed:
                changed = False
                new_indices = []
                for idx in indices:
                    necessary = False
                    for w in balls_list[idx]:
                        if cover_counts[w] == 1:
                            necessary = True
                            break
                    if necessary:
                        new_indices.append(idx)
                    else:
                        for w in balls_list[idx]:
                            cover_counts[w] -= 1
                        changed = True
                indices = new_indices
            return indices

        # Local search: try to remove one codeword and add another to reduce size
        def local_search(code_indices):
            if time.time() > deadline - 15.0:
                return code_indices
            code_indices = list(code_indices)
            improved = True
            while improved:
                improved = False
                # Try removing each codeword and see if we can cover the gap with one addition
                for i in range(len(code_indices)):
                    if time.time() > deadline - 15.0:
                        return code_indices
                    candidate = code_indices[i]
                    # Compute uncovered words if we remove candidate
                    remaining = [c for j, c in enumerate(code_indices) if j != i]
                    covered = set()
                    for c in remaining:
                        covered.update(balls_list[c])
                    uncovered = set(range(total_words)) - covered
                    if not uncovered:
                        # Removing candidate still covers everything
                        code_indices = remaining
                        improved = True
                        break
                    # Try adding one codeword to cover all uncovered
                    # Find codewords that cover all uncovered words
                    best_add = -1
                    for c in range(total_words):
                        if c == candidate:
                            continue
                        if uncovered.issubset(balls_set[c]):
                            # Check if adding c and removing candidate reduces size or keeps same
                            # Actually we're replacing candidate with c, so size stays same
                            # But we want to see if we can get a better structure
                            # For now, just accept if it covers
                            best_add = c
                            break
                    if best_add != -1:
                        code_indices[i] = best_add
                        improved = True
                        break
            return prune_cover(code_indices)

        best_code = None
        best_size = float('inf')

        # Strategy 1: Direct Product Construction (Tetracode based)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))

        pruned_dp = prune_cover(dp_indices)
        if len(pruned_dp) < best_size:
            best_size = len(pruned_dp)
            best_code = pruned_dp

        # Strategy 2: Randomized Greedy with symmetry-guided tie-breaking
        num_restarts = 200  # Increased restarts with faster greedy

        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break

            uncovered = set(range(total_words))
            code_indices = []

            while uncovered:
                # Find max gain using list-based balls for speed
                max_gain = 0
                top_candidates = []
                for c in range(total_words):
                    gain = 0
                    for w in balls_list[c]:
                        if w in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        top_candidates = [c]
                    elif gain == max_gain and gain > 0:
                        top_candidates.append(c)

                if not top_candidates:
                    break

                # Symmetry-guided tie-breaking: prefer codewords that are more "central"
                # or have a more regular pattern to encourage structured solutions
                if len(top_candidates) > 1:
                    # Use a deterministic hash-based tie-break with some randomness
                    # to encourage exploration while maintaining some structure
                    random.shuffle(top_candidates)
                    best_c = top_candidates[0]
                else:
                    best_c = top_candidates[0]

                code_indices.append(best_c)
                for w in balls_list[best_c]:
                    uncovered.discard(w)

                if time.time() > deadline - 10.0:
                    break

            if uncovered:
                continue

            # Prune and then local search
            pruned = prune_cover(code_indices)
            pruned = local_search(pruned)

            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned

        if best_code is None:
            best_code = pruned_dp
        else:
            if not is_cover(best_code):
                best_code = pruned_dp

        result = [list(idx_to_tuple(c)) for c in best_code]
        return result

    else:
        # Generic fallback for other parameters
        # If the space is small, use greedy
        total_words = q ** n
        if total_words <= 10000:
            # Re-use logic from above but parameterized? 
            # For simplicity and safety, just return a large set or use a simple greedy
            # Since the task is specific to 3,6,1, this branch is unlikely.
            # But let's implement a basic greedy to be safe.
            
            powers = [1] * n
            for i in range(1, n):
                powers[i] = powers[i-1] * q
                
            def idx_to_tuple_gen(idx):
                t = [0] * n
                rem = idx
                for i in range(n):
                    t[i] = rem % q
                    rem //= q
                return tuple(t)
                
            def tuple_to_idx_gen(t):
                idx = 0
                for i in range(n):
                    idx += t[i] * powers[i]
                return idx

            balls = [None] * total_words
            for i in range(total_words):
                w = idx_to_tuple_gen(i)
                ball = {i}
                for pos in range(n):
                    orig_val = w[pos]
                    for new_val in range(q):
                        if new_val != orig_val:
                            new_w = list(w)
                            new_w[pos] = new_val
                            new_idx = tuple_to_idx_gen(tuple(new_w))
                            ball.add(new_idx)
                balls[i] = ball

            uncovered = set(range(total_words))
            code_indices = []
            
            while uncovered:
                best_c = -1
                max_gain = 0
                for c in range(total_words):
                    gain = len(balls[c] & uncovered)
                    if gain > max_gain:
                        max_gain = gain
                        best_c = c
                if best_c == -1:
                    break
                code_indices.append(best_c)
                uncovered -= balls[best_c]
                
                if time.time() > deadline:
                    break
            
            result = [list(idx_to_tuple_gen(c)) for c in code_indices]
            return result
        else:
            # Fallback for large spaces: return a grid or something
            # This is a placeholder
            return [[0] * n]
