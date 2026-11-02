import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    total = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    # Precompute balls as numpy array
    def build_balls():
        # For each word index, compute all words within Hamming distance radius
        balls = np.empty((total, ball_size), dtype=np.int32)
        for w in range(total):
            # Convert to word
            digits = []
            tmp = w
            for i in range(n):
                digits.append(tmp % q)
                tmp //= q
            digits = digits[::-1]  # now digits[0] is most significant

            ball = [w]
            for pos in range(n):
                orig = digits[pos]
                for nv in range(q):
                    if nv != orig:
                        new_word = digits[:]
                        new_word[pos] = nv
                        # Convert back to int
                        val = 0
                        for d in new_word:
                            val = val * q + d
                        ball.append(val)
            balls[w] = ball
        return balls

    balls = build_balls()

    # Greedy feasible cover
    covered = np.zeros(total, dtype=np.bool_)
    covered_count = 0
    codewords = []

    while covered_count < total:
        if time.time() > deadline - 30:
            break

        # Compute gain for each word
        covered_in_balls = covered[balls]  # (total, ball_size)
        covered_per_ball = covered_in_balls.sum(axis=1)
        gains = ball_size - covered_per_ball

        best_word = int(np.argmax(gains))
        best_new = int(gains[best_word])
        if best_new == 0:
            break

        codewords.append(best_word)
        bp = balls[best_word]
        newly = ~covered[bp]
        if newly.any():
            covered[bp] = True
            covered_count += int(newly.sum())

    if not codewords:
        return []

    # Convert to set for fast membership
    cw_set = set(codewords)

    # Build coverage count array
    cov_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        cov_count[balls[cw]] += 1

    # Local search: try to remove redundant codewords first
    changed = True
    while changed and time.time() < deadline - 20:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline - 20:
                break
            cw = codewords[i]
            if (cov_count[balls[cw]] > 1).all():
                # Remove
                cov_count[balls[cw]] -= 1
                cw_set.discard(cw)
                codewords.pop(i)
                changed = True
                break

    # Pairwise exchange local search
    # Try to replace one codeword with another to reduce size or improve coverage
    # Strategy: for each codeword, try removing it and see if we can add a better one
    # that covers more uniquely-covered points

    # Precompute: for each codeword, how many points it uniquely covers
    # unique_covered[cw] = number of points x where cov_count[x] == 1 and x is in ball(cw)

    def get_unique_count(cw):
        bp = balls[cw]
        return int((cov_count[bp] == 1).sum())

    # Try swaps: remove cw_i, add cw_j, check if still feasible
    # Feasibility: after removing cw_i, all points in ball(cw_i) that were uniquely covered
    # must still be covered by remaining codewords
    # After adding cw_j, all points must still be covered

    # More efficient: try to find a codeword that can be removed (redundant)
    # Then try to find a codeword whose removal + another's addition reduces size

    # Phase 1: aggressive removal of redundant codewords
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        removed_any = False
        for i in range(len(codewords)):
            if time.time() > deadline - 10:
                break
            cw = codewords[i]
            bp = balls[cw]
            if (cov_count[bp] > 1).all():
                cov_count[bp] -= 1
                cw_set.discard(cw)
                codewords.pop(i)
                removed_any = True
                break
        if not removed_any:
            break

    # Phase 2: Try pairwise exchanges
    # For each pair (i, j), try removing i and adding j
    # Check if the result is still a valid cover
    # Use randomized ordering to escape local optima

    rng = random.Random(42)
    cw_list = list(codewords)

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        rng.shuffle(cw_list)

        for i_idx, cw_i in enumerate(cw_list):
            if time.time() > deadline - 5:
                break
            if cw_i not in cw_set:
                continue

            # Try removing cw_i
            bp_i = balls[cw_i]
            # Points uniquely covered by cw_i
            unique_pts = bp_i[cov_count[bp_i] == 1]

            if len(unique_pts) == 0:
                # cw_i is redundant, just remove
                cov_count[bp_i] -= 1
                cw_set.discard(cw_i)
                if cw_i in codewords:
                    codewords.remove(cw_i)
                improved = True
                continue

            # Try to find a replacement: a word not in cw_set that covers all unique_pts
            # and whose addition doesn't create issues
            # For efficiency, only try a limited set of candidates
            # Candidates: words in the ball of each unique point (since they must cover those points)

            candidates = set()
            for up in unique_pts:
                # Words that cover up are those whose ball contains up
                # For radius 1, word w covers up if dist(w, up) <= 1
                # So candidates are up itself and all words at distance 1 from up
                candidates.add(int(up))
                # Generate neighbors of up
                up_word = []
                tmp = int(up)
                for pos in range(n):
                    up_word.append(tmp % q)
                    tmp //= q
                up_word = up_word[::-1]
                for pos in range(n):
                    orig = up_word[pos]
                    for nv in range(q):
                        if nv != orig:
                            nw = up_word[:]
                            nw[pos] = nv
                            val = 0
                            for d in nw:
                                val = val * q + d
                            candidates.add(val)

            # Filter candidates: must not be in cw_set, and must cover all unique_pts
            best_candidate = None
            best_score = -1

            for cand in candidates:
                if cand in cw_set:
                    continue
                # Check if cand covers all unique_pts
                bp_cand = balls[cand]
                # Check coverage of unique_pts
                # A point p is covered by cand if p is in ball(cand)
                # Equivalently, dist(p, cand) <= 1
                # For efficiency, check if all unique_pts are in bp_cand
                # This is expensive, so only do it for a limited number of candidates
                if len(candidates) > 100:
                    # Limit check
                    if best_score >= 0:
                        break

                # Check if cand covers all unique points
                covers_all = True
                for up in unique_pts:
                    # Check if up is in ball(cand)
                    # dist(up, cand) <= 1
                    # Compute Hamming distance
                    up_digits = []
                    tmp = int(up)
                    for pos in range(n):
                        up_digits.append(tmp % q)
                        tmp //= q
                    up_digits = up_digits[::-1]
                    cand_digits = []
                    tmp = int(cand)
                    for pos in range(n):
                        cand_digits.append(tmp % q)
                        tmp //= q
                    cand_digits = cand_digits[::-1]
                    dist = sum(1 for a, b in zip(up_digits, cand_digits) if a != b)
                    if dist > 1:
                        covers_all = False
                        break
                if not covers_all:
                    continue

                # Score: how many new points does cand cover (not already covered)
                # Points in ball(cand) that are not covered by other codewords
                # After removing cw_i, coverage of points in bp_i decreases by 1
                # So a point p is uncovered after removal if cov_count[p] == 1 (and p in bp_i)
                # cand covers p if p in bp_cand
                # New coverage from cand: points in bp_cand that would be uncovered without cand
                # = points in bp_cand where (cov_count[p] - (1 if p in bp_i else 0)) == 0
                # i.e., cov_count[p] == 0 or (cov_count[p] == 1 and p in bp_i)

                # Simplify: just count how many points in bp_cand have cov_count <= 1
                # (after removing cw_i, their count drops by 1 if in bp_i)
                bp_cand_arr = balls[cand]
                # Effective coverage after removing cw_i:
                # eff_count[p] = cov_count[p] - (1 if p in bp_i else 0)
                # We need eff_count[p] >= 1 for all p, and cand covers p if p in bp_cand
                # So cand helps if eff_count[p] == 0 for some p in bp_cand
                # Number of such p = sum over p in bp_cand of [eff_count[p] == 0]
                # eff_count[p] == 0 means cov_count[p] == 0 or (cov_count[p] == 1 and p in bp_i)

                # For speed, approximate: count points in bp_cand with cov_count <= 1
                score = int((cov_count[bp_cand_arr] <= 1).sum())

                if score > best_score:
                    best_score = score
                    best_candidate = cand

            if best_candidate is not None and best_score > 0:
                # Perform swap: remove cw_i, add best_candidate
                cov_count[bp_i] -= 1
                cw_set.discard(cw_i)
                if cw_i in codewords:
                    codewords.remove(cw_i)

                bp_new = balls[best_candidate]
                cov_count[bp_new] += 1
                cw_set.add(best_candidate)
                codewords.append(best_candidate)
                improved = True

    # Final cleanup: remove any remaining redundant codewords
    if time.time() < deadline - 2:
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline - 2:
                    break
                cw = codewords[i]
                if (cov_count[balls[cw]] > 1).all():
                    cov_count[balls[cw]] -= 1
                    cw_set.discard(cw)
                    codewords.pop(i)
                    changed = True
                    break

    # Convert to word lists
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    result = [to_word(w) for w in codewords]

    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)

    return unique_result
