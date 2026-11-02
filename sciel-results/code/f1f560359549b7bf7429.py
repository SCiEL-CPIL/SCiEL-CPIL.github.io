import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    N = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute balls as a matrix: balls_matrix[i] = list of 17 neighbors of i
    balls_matrix = np.zeros((N, ball_size), dtype=np.int32)
    for w in range(N):
        neighbors = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors.append(new_w)
        balls_matrix[w] = neighbors

    # Precompute distance-2 graph: for each word, list of words at Hamming distance exactly 2
    # This is used for efficient swap candidate generation
    dist2_neighbors = []
    for w in range(N):
        nbrs = []
        for pos1 in range(n):
            w1 = weights[pos1]
            d1 = (w // w1) % 3
            for pos2 in range(pos1 + 1, n):
                w2 = weights[pos2]
                d2 = (w // w2) % 3
                for nd1 in range(q):
                    if nd1 == d1:
                        continue
                    for nd2 in range(q):
                        if nd2 == d2:
                            continue
                        new_w = w + (nd1 - d1) * w1 + (nd2 - d2) * w2
                        nbrs.append(new_w)
        dist2_neighbors.append(nbrs)

    # Precompute for each word, the set of words at distance 1 (for fast overlap checks)
    # dist1_neighbors[i] = list of 16 words at Hamming distance 1 from i
    dist1_matrix = np.zeros((N, 2 * n), dtype=np.int32)
    for w in range(N):
        nbrs = []
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                nbrs.append(new_w)
        dist1_matrix[w] = nbrs

    def idx_to_list(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]

    def greedy_construct(cover_count=None, uncovered=None, selected=None):
        """Greedy set cover with random tie-breaking."""
        if cover_count is None:
            cover_count = np.zeros(N, dtype=np.int32)
        if uncovered is None:
            uncovered = np.ones(N, dtype=bool)
        if selected is None:
            selected = []

        while np.any(uncovered):
            if time.time() > deadline - 12.0:
                break

            gains = uncovered[balls_matrix].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(best_idx)
            ball = balls_matrix[best_idx]
            cover_count[ball] += 1
            uncovered[ball] = False

        return selected, cover_count, uncovered

    def prune(selected, cover_count):
        """Remove redundant codewords. Returns (new_selected, new_cover_count)."""
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls_matrix[c]]) > 1:
                    cover_count[balls_matrix[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count

    def try_swap(selected, cover_count, u_idx, v):
        """Try to swap selected[u_idx] with v. Returns (success, new_selected, new_cover_count)."""
        u = selected[u_idx]
        if u == v:
            return False, selected, cover_count

        # Check: all words in B(u) \ B(v) must have cover_count >= 2
        ball_u = balls_matrix[u]
        ball_v = balls_matrix[v]

        # Create a set for fast lookup
        ball_v_set = set(ball_v.tolist())
        risk_words = [x for x in ball_u.tolist() if x not in ball_v_set]

        if risk_words:
            risk_arr = np.array(risk_words, dtype=np.int32)
            if np.min(cover_count[risk_arr]) <= 1:
                return False, selected, cover_count

        # Perform swap
        cover_count[ball_u] -= 1
        cover_count[ball_v] += 1
        selected[u_idx] = v
        return True, selected, cover_count

    def local_search(selected, cover_count, time_budget):
        """Local search with swaps and pruning."""
        end_time = min(deadline - 3.0, time.time() + time_budget)
        mask_u = np.zeros(N, dtype=bool)
        mask_v = np.zeros(N, dtype=bool)

        while time.time() < end_time:
            if not selected:
                break

            # Pick a random codeword to potentially swap
            i = np.random.randint(len(selected))
            u = selected[i]

            # Generate candidates: prefer words at distance 2 from u (most likely to enable pruning)
            if random.random() < 0.7:
                # Pick from distance-2 neighbors
                candidates = dist2_neighbors[u]
                if candidates:
                    v = candidates[np.random.randint(len(candidates))]
                else:
                    v = np.random.randint(N)
            else:
                v = np.random.randint(N)

            if v == u:
                continue

            # Check safety using masks
            mask_u.fill(False)
            mask_u[balls_matrix[u]] = True
            mask_v.fill(False)
            mask_v[balls_matrix[v]] = True

            risk = mask_u & ~mask_v
            if np.any(risk):
                if np.min(cover_count[risk]) <= 1:
                    continue

            # Perform swap
            cover_count[balls_matrix[u]] -= 1
            cover_count[balls_matrix[v]] += 1
            selected[i] = v

            # Prune
            changed = True
            while changed and time.time() < end_time:
                changed = False
                new_selected = []
                for c in selected:
                    if np.min(cover_count[balls_matrix[c]]) > 1:
                        cover_count[balls_matrix[c]] -= 1
                        changed = True
                    else:
                        new_selected.append(c)
                selected = new_selected

        return selected, cover_count

    def double_swap_search(selected, cover_count, time_budget):
        """Try to replace two codewords with one, or one with two, to reduce size."""
        end_time = min(deadline - 3.0, time.time() + time_budget)
        mask = np.zeros(N, dtype=bool)

        while time.time() < end_time:
            if len(selected) < 2:
                break

            # Try: remove two codewords, add one
            i, j = np.random.choice(len(selected), 2, replace=False)
            u, v = selected[i], selected[j]

            # Union of balls of u and v
            mask.fill(False)
            mask[balls_matrix[u]] = True
            mask[balls_matrix[v]] = True

            # Find words in union with cover_count == 1
            risk = mask & (cover_count <= 1)
            if np.any(risk):
                continue

            # Find a word w that covers all risk words
            # w must be in B(x) for all x in risk
            # For efficiency, pick a random risk word and find candidates
            risk_words = np.where(risk)[0]
            if len(risk_words) == 0:
                # No risk, just remove both
                cover_count[balls_matrix[u]] -= 1
                cover_count[balls_matrix[v]] -= 1
                new_selected = [c for k, c in enumerate(selected) if k != i and k != j]
                selected, cover_count = prune(new_selected, cover_count)
                continue

            # Pick a random risk word and try to find a codeword covering it
            sample_risk = risk_words[np.random.randint(len(risk_words))]
            # Candidates: words at distance <= 1 from sample_risk
            candidates = balls_matrix[sample_risk]
            w = candidates[np.random.randint(len(candidates))]

            # Check if w covers all risk words
            ball_w = balls_matrix[w]
            ball_w_set = set(ball_w.tolist())
            all_covered = all(x in ball_w_set for x in risk_words.tolist())

            if all_covered:
                cover_count[balls_matrix[u]] -= 1
                cover_count[balls_matrix[v]] -= 1
                cover_count[ball_w] += 1
                new_selected = [c for k, c in enumerate(selected) if k != i and k != j]
                if w not in new_selected:
                    new_selected.append(w)
                selected, cover_count = prune(new_selected, cover_count)

        return selected, cover_count

    best_size = N
    best_set = None

    # Multiple restarts with different strategies
    restart_count = 0
    while time.time() < deadline - 15.0:
        restart_count += 1

        # Strategy varies by restart
        if restart_count % 3 == 1:
            # Standard greedy + local search
            selected, cover_count, uncovered = greedy_construct()
            if np.any(uncovered):
                continue
            selected, cover_count = prune(selected, cover_count)
            selected, cover_count = local_search(selected, cover_count, 15.0)
        elif restart_count % 3 == 2:
            # Greedy + double swap + local search
            selected, cover_count, uncovered = greedy_construct()
            if np.any(uncovered):
                continue
            selected, cover_count = prune(selected, cover_count)
            selected, cover_count = double_swap_search(selected, cover_count, 10.0)
            selected, cover_count = local_search(selected, cover_count, 10.0)
        else:
            # Greedy + more aggressive local search
            selected, cover_count, uncovered = greedy_construct()
            if np.any(uncovered):
                continue
            selected, cover_count = prune(selected, cover_count)
            selected, cover_count = local_search(selected, cover_count, 25.0)

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            # Try to polish the best solution
            if time.time() < deadline - 20.0:
                selected, cover_count = local_search(selected, cover_count, 10.0)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected.copy()

    if best_set is None:
        return [idx_to_list(w) for w in range(N)]

    return [idx_to_list(w) for w in best_set]
