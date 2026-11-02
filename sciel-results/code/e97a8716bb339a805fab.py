import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)

    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]

    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 1 + 7*2 = 15

    # Precompute all words as index arrays
    all_words = np.zeros((N, n), dtype=np.int32)
    for i in range(N):
        temp = i
        for j in range(n - 1, -1, -1):
            all_words[i, j] = temp % q
            temp //= q

    # Precompute base values for index arithmetic
    base = np.zeros(n, dtype=np.int32)
    base[n - 1] = 1
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = []
    for i in range(N):
        w = all_words[i]
        ball = [i]
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball.append(new_idx)
        ball_indices.append(np.array(ball, dtype=np.int32))

    # Precompute coverers: coverers[point] = list of codeword indices that cover point
    coverers = [[] for _ in range(N)]
    for cw_idx in range(N):
        for pt in ball_indices[cw_idx]:
            coverers[pt].append(cw_idx)

    # Precompute bitsets for fast coverage tracking
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in ball_indices[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask

    all_mask = (1 << N) - 1

    # Helper: greedy set cover with optional randomization
    def greedy_build(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        covered = np.zeros(N, dtype=bool)
        code = []
        current_cover_mask = 0

        while current_cover_mask != all_mask:
            best_i = -1
            best_count = 0
            candidates = list(range(N))
            rng.shuffle(candidates)

            for i in candidates:
                new_bits = ball_bitsets[i] & ~current_cover_mask
                if new_bits:
                    count = new_bits.bit_count()
                    if count > best_count:
                        best_count = count
                        best_i = i
                        if best_count == ball_size:
                            break

            if best_i == -1:
                break

            code.append(best_i)
            current_cover_mask |= ball_bitsets[best_i]

        return code

    # Multiple greedy restarts
    best_initial = None
    best_initial_size = float('inf')

    init_code = greedy_build(seed=0)
    best_initial = set(init_code)
    best_initial_size = len(init_code)

    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 60.0:
            break
        init_code = greedy_build(seed=seed)
        if len(init_code) < best_initial_size:
            best_initial_size = len(init_code)
            best_initial = set(init_code)

    current_code = list(best_initial)
    best_code = list(current_code)
    best_size = len(current_code)

    # Initialize cover counts
    cnt = np.zeros(N, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices[c]] += 1

    def try_remove_redundant(codewords, cover_count):
        """Remove redundant codewords. Returns number removed."""
        removed = 0
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 15.0:
                break
            cw = codewords[i]
            b = ball_indices[cw]
            if np.min(cover_count[b]) > 1:
                cover_count[b] -= 1
                codewords.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    # Initial redundancy removal
    try_remove_redundant(current_code, cnt)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)

    # Local Search Loop
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 20

    while time.time() < deadline - 5.0:
        improved = False

        # 1. Try Removals
        removed = try_remove_redundant(current_code, cnt)
        if removed > 0:
            improved = True
            no_improvement_count = 0
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            continue

        # 2. Try Swaps
        if current_code:
            # Pick a random codeword to try swapping
            idx = random.randint(0, len(current_code) - 1)
            old_cw = current_code[idx]

            # Find critical points: points in ball(old_cw) with count == 1
            b_old = ball_indices[old_cw]
            critical_mask = cnt[b_old] == 1
            critical_points = b_old[critical_mask]

            if len(critical_points) == 0:
                # Should have been removed by redundancy check
                cnt[b_old] -= 1
                current_code.pop(idx)
                improved = True
                no_improvement_count = 0
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                continue

            # Find candidates that cover ALL critical points
            # Use intersection of coverers
            candidate_set = set(coverers[critical_points[0]])
            for p in critical_points[1:]:
                candidate_set.intersection_update(coverers[p])

            # Remove old_cw and existing codewords from candidates
            candidate_set.discard(old_cw)
            for cw in current_code:
                candidate_set.discard(cw)

            if candidate_set:
                # Pick the best candidate using vectorized cost evaluation
                candidates_list = list(candidate_set)
                if len(candidates_list) > 30:
                    random.shuffle(candidates_list)
                    candidates_list = candidates_list[:30]

                # Vectorized cost: sum of cover_count in each candidate's ball
                # Lower cost = less redundancy added
                best_new_cw = -1
                min_cost = float('inf')

                for new_cw in candidates_list:
                    b_new = ball_indices[new_cw]
                    cost = np.sum(cnt[b_new])
                    if cost < min_cost:
                        min_cost = cost
                        best_new_cw = new_cw

                if best_new_cw != -1:
                    # Perform swap
                    cnt[b_old] -= 1
                    b_new = ball_indices[best_new_cw]
                    cnt[b_new] += 1
                    current_code[idx] = best_new_cw
                    improved = True
                    no_improvement_count = 0
                    # Try to remove redundancies after swap
                    try_remove_redundant(current_code, cnt)
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    continue

        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                # Perturbation: Add a random codeword
                rand_cw = random.randint(0, N - 1)
                if rand_cw not in current_code:
                    current_code.append(rand_cw)
                    cnt[ball_indices[rand_cw]] += 1
                    no_improvement_count = 0
                else:
                    no_improvement_count = 0

    # Symmetry-based polish: apply coordinate and digit permutations
    # The ternary Hamming space has high symmetry. We can try to exploit it.
    if time.time() < deadline - 3.0:
        # Generate some random symmetry transformations
        # Coordinate permutations (7! = 5040, sample a few)
        # Digit permutations (S3^7 = 6^7, sample a few)
        
        best_code_set = set(best_code)
        best_code_size = len(best_code)
        
        # Try a few random symmetry transformations on the best solution
        for _ in range(10):
            if time.time() > deadline - 3.0:
                break
            
            # Random coordinate permutation
            coord_perm = list(range(n))
            random.shuffle(coord_perm)
            
            # Random digit permutations for each coordinate
            digit_perms = []
            for _ in range(n):
                perm = list(range(q))
                random.shuffle(perm)
                digit_perms.append(perm)
            
            # Apply transformation to all codewords
            transformed = set()
            for cw in best_code:
                word = all_words[cw]
                new_word = np.zeros(n, dtype=np.int32)
                for i in range(n):
                    new_word[coord_perm[i]] = digit_perms[i][word[i]]
                # Find index of transformed word
                idx = 0
                for j in range(n):
                    idx = idx * q + new_word[j]
                transformed.add(idx)
            
            # Take union with original
            combined = list(best_code_set | transformed)
            
            # Re-optimize the combined solution
            if len(combined) > best_code_size:
                cnt_temp = np.zeros(N, dtype=np.int32)
                for c in combined:
                    cnt_temp[ball_indices[c]] += 1
                
                # Remove redundancies
                try_remove_redundant(combined, cnt_temp)
                
                if len(combined) < best_code_size:
                    best_code_size = len(combined)
                    best_code = list(combined)
                    best_code_set = set(combined)

    # Final Polish
    final_codewords = list(best_code)
    final_cover_count = np.zeros(N, dtype=np.int32)
    for cw in final_codewords:
        final_cover_count[ball_indices[cw]] += 1

    # Run redundancy removal on the best solution found
    if np.min(final_cover_count) > 0:
        try_remove_redundant(final_codewords, final_cover_count)

    # Verify final validity
    if np.min(final_cover_count) > 0:
        return [list(all_words[cw]) for cw in final_codewords]
    else:
        # Fallback: Return the initial greedy solution if something went wrong
        return [list(all_words[cw]) for cw in current_code]
