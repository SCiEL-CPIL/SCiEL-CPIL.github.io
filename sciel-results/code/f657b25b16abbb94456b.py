import time
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using hypergraph vertex cover formulation with
    targeted local search and multi-start restarts.
    
    Vertices: 6561 words
    Hyperedges: 17-word spheres centered at each potential codeword
    Objective: minimize number of hyperedges (codewords) such that all vertices are covered
    """
    start_time = time.time()
    deadline = start_time + 145.0

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

    # Precompute balls
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
    ball_sets = [frozenset(balls[i].tolist()) for i in range(total_words)]

    # Precompute overlap matrix: overlap[i][j] = |balls[i] ∩ balls[j]|
    # This is expensive (6561^2 * 17), but we can compute it incrementally
    # Actually, let's use a different approach: for each word, precompute which centers cover it
    # center_covers[word] = list of centers whose ball contains word
    center_covers = [[] for _ in range(total_words)]
    for i in range(total_words):
        for w in balls[i]:
            center_covers[w].append(i)

    def greedy_cover(seed_offset=0):
        """Greedy set cover with optional randomization."""
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)

        # Randomize the order of centers for diversity
        if seed_offset > 0:
            rng = np.random.RandomState(seed_offset)
        else:
            rng = np.random.RandomState(0)

        while np.any(uncovered):
            if time.time() > deadline - 20:
                break

            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break

            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                best_center = int(best_centers[rng.randint(len(best_centers))])

            code.append(best_center)
            uncovered[balls[best_center]] = False
            cover_count[balls[best_center]] += 1

        return code, cover_count

    def trim_code(code, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed and time.time() < deadline - 10:
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
        return code, cover_count

    def targeted_local_search(code, cover_count, max_iter=500000):
        """
        Aggressive targeted local search using hypergraph vertex cover perspective.
        For each codeword, find candidates that cover its uniquely-covered words.
        """
        code_set = set(code)
        candidates = list(set(range(total_words)) - code_set)
        cand_arr = np.array(candidates, dtype=np.int32) if candidates else np.array([], dtype=np.int32)
        n_cand = len(cand_arr)

        improved = True
        iter_count = 0
        while improved and time.time() < deadline - 10 and iter_count < max_iter:
            improved = False
            iter_count += 1

            # Process codewords in order of fewest unique words (most replaceable first)
            code_indices = list(range(len(code)))
            unique_counts = []
            for ci in code_indices:
                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                unique_counts.append(np.sum(only_mask))
            
            # Sort by unique count ascending (try to remove/replace most redundant first)
            code_indices.sort(key=lambda x: unique_counts[x])

            for ci in code_indices:
                if time.time() > deadline - 10:
                    break
                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                # Targeted search: find candidates that cover all of only_by_i
                found = False
                if n_cand == 0:
                    break

                # Score candidates by how many of only_by_i they cover
                # Use vectorized computation for speed
                # For each candidate, count overlap with only_by_i
                if len(only_by_i) <= 5:
                    # Small deficit: check all candidates efficiently
                    # Create a lookup: for each word in only_by_i, which candidates cover it
                    candidate_scores = np.zeros(n_cand, dtype=np.int8)
                    for w in only_by_i:
                        # Find candidates that cover w
                        covering_cands = center_covers[w]
                        # Map to candidate indices
                        for cc_idx in covering_cands:
                            if cc_idx in code_set:
                                continue
                            # Find position in cand_arr
                            # This is slow; instead, precompute a mapping
                            pass
                    
                    # Simpler: sample and check
                    sample_size = min(500, n_cand)
                    sample_idx = np.random.choice(n_cand, size=sample_size, replace=False)
                    
                    for si in sample_idx:
                        j = int(cand_arr[si])
                        if j in code_set:
                            continue
                        j_set = ball_sets[j]
                        valid = all(w in j_set for w in only_by_i)
                        if valid:
                            code[ci] = j
                            code_set.discard(i)
                            code_set.add(j)
                            cover_count[i_ball] -= 1
                            cover_count[balls[j]] += 1
                            # Update candidates list
                            candidates = [c for c in candidates if c != j]
                            if i in candidates:
                                candidates.append(i)
                            cand_arr = np.array(candidates, dtype=np.int32) if candidates else np.array([], dtype=np.int32)
                            n_cand = len(cand_arr)
                            improved = True
                            found = True
                            break
                else:
                    # Larger deficit: sample candidates
                    sample_size = min(200, n_cand)
                    sample_idx = np.random.choice(n_cand, size=sample_size, replace=False)
                    
                    for si in sample_idx:
                        j = int(cand_arr[si])
                        if j in code_set:
                            continue
                        j_set = ball_sets[j]
                        valid = all(w in j_set for w in only_by_i)
                        if valid:
                            code[ci] = j
                            code_set.discard(i)
                            code_set.add(j)
                            cover_count[i_ball] -= 1
                            cover_count[balls[j]] += 1
                            candidates = [c for c in candidates if c != j]
                            if i in candidates:
                                candidates.append(i)
                            cand_arr = np.array(candidates, dtype=np.int32) if candidates else np.array([], dtype=np.int32)
                            n_cand = len(cand_arr)
                            improved = True
                            found = True
                            break

                if found:
                    break

        # Final trim
        code, cover_count = trim_code(code, cover_count)
        return code, cover_count

    def simulated_annealing_polish(code, cover_count, max_iters=300000):
        """
        Simulated annealing with hypergraph-aware moves.
        Moves: single swap, pair swap, removal
        Acceptance based on change in "effective coverage" (smooth surrogate)
        """
        code_set = set(code)
        candidates = list(set(range(total_words)) - code_set)
        cand_arr = np.array(candidates, dtype=np.int32) if candidates else np.array([], dtype=np.int32)
        n_cand = len(cand_arr)
        cc = cover_count.copy()

        T_start = 3.0
        T_end = 0.05
        n_sa_iters = 0
        improved = True

        while improved and n_sa_iters < max_iters and time.time() < deadline - 5:
            improved = False
            T = T_start * (T_end / T_start) ** (n_sa_iters / max(1, max_iters))
            n_sa_iters += 1

            if len(code) == 0 or n_cand == 0:
                break

            move_type = np.random.random()

            if move_type < 0.4:
                # Single swap
                i = np.random.randint(len(code))
                old_i = code[i]
                i_ball = balls[old_i]
                only_mask = cc[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(i)
                    cc[i_ball] -= 1
                    code_set.discard(old_i)
                    candidates.append(old_i)
                    cand_arr = np.array(candidates, dtype=np.int32)
                    n_cand = len(cand_arr)
                    improved = True
                    continue

                a = np.random.randint(n_cand)
                j = int(cand_arr[a])
                if j in code_set:
                    continue

                j_set = ball_sets[j]
                deficit = sum(1 for w in only_by_i if w not in j_set)

                if deficit == 0:
                    code[i] = j
                    code_set.discard(old_i)
                    code_set.add(j)
                    cc[i_ball] -= 1
                    cc[balls[j]] += 1
                    candidates[a] = old_i
                    improved = True
                elif deficit <= 2 and np.random.random() < np.exp(-deficit / T):
                    code[i] = j
                    code_set.discard(old_i)
                    code_set.add(j)
                    cc[i_ball] -= 1
                    cc[balls[j]] += 1
                    candidates[a] = old_i
                    improved = True

            elif move_type < 0.6:
                # Pair swap
                if len(code) < 2:
                    continue
                i = np.random.randint(len(code))
                j_idx = np.random.randint(len(code))
                if j_idx == i:
                    continue
                a = np.random.randint(n_cand)
                b = np.random.randint(n_cand)
                if a == b:
                    continue
                new_a = int(cand_arr[a])
                new_b = int(cand_arr[b])
                if new_a in code_set or new_b in code_set:
                    continue

                old_i = code[i]
                old_j = code[j_idx]
                i_ball = balls[old_i]
                j_ball = balls[old_j]

                only_i = i_ball[cc[i_ball] == 1]
                only_j = j_ball[cc[j_ball] == 1]
                required = set(only_i.tolist()) | set(only_j.tolist())

                if len(required) == 0:
                    # Remove both
                    if j_idx > i:
                        code.pop(j_idx)
                        cc[balls[old_j]] -= 1
                        code_set.discard(old_j)
                        code.pop(i)
                        cc[balls[old_i]] -= 1
                        code_set.discard(old_i)
                    else:
                        code.pop(i)
                        cc[balls[old_i]] -= 1
                        code_set.discard(old_i)
                        code.pop(j_idx)
                        cc[balls[old_j]] -= 1
                        code_set.discard(old_j)
                    candidates.extend([old_i, old_j])
                    cand_arr = np.array(candidates, dtype=np.int32)
                    n_cand = len(cand_arr)
                    improved = True
                    continue

                na_set = ball_sets[new_a]
                nb_set = ball_sets[new_b]
                deficit = sum(1 for w in required if w not in na_set and w not in nb_set)

                if deficit == 0:
                    code[i] = new_a
                    code[j_idx] = new_b
                    code_set.discard(old_i)
                    code_set.discard(old_j)
                    code_set.add(new_a)
                    code_set.add(new_b)
                    cc[i_ball] -= 1
                    cc[j_ball] -= 1
                    cc[balls[new_a]] += 1
                    cc[balls[new_b]] += 1
                    candidates[a] = old_i
                    candidates[b] = old_j
                    improved = True
                elif deficit <= 2 and np.random.random() < np.exp(-deficit / T):
                    code[i] = new_a
                    code[j_idx] = new_b
                    code_set.discard(old_i)
                    code_set.discard(old_j)
                    code_set.add(new_a)
                    code_set.add(new_b)
                    cc[i_ball] -= 1
                    cc[j_ball] -= 1
                    cc[balls[new_a]] += 1
                    cc[balls[new_b]] += 1
                    candidates[a] = old_i
                    candidates[b] = old_j
                    improved = True

            else:
                # Try removal
                i = np.random.randint(len(code))
                old_i = code[i]
                i_ball = balls[old_i]
                if np.any(cc[i_ball] == 1):
                    continue
                code.pop(i)
                cc[i_ball] -= 1
                code_set.discard(old_i)
                candidates.append(old_i)
                cand_arr = np.array(candidates, dtype=np.int32)
                n_cand = len(cand_arr)
                improved = True

        code, cc = trim_code(code, cc)
        return code, cc

    # Multi-start with different random seeds
    best_code = None
    best_count = float('inf')
    np.random.seed(42)

    n_starts = 5
    for start_i in range(n_starts):
        if time.time() > deadline - 40:
            break

        code, cover_count = greedy_cover(seed_offset=start_i * 1000 + 1)
        code, cover_count = trim_code(code, cover_count)
        code, cover_count = targeted_local_search(code, cover_count, max_iter=200000)
        code, cover_count = simulated_annealing_polish(code, cover_count, max_iters=100000)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Final polish on best solution
    if best_code is not None:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1
        best_code, cover_count = targeted_local_search(best_code, cover_count, max_iter=300000)
        best_code, cover_count = simulated_annealing_polish(best_code, cover_count, max_iters=200000)

    if best_code is None:
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result
