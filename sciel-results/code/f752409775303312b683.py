import time

def construct():
    deadline = time.time() + 115.0
    target_mask = (1 << 30) - 1  # bits 0..29 for diffs 1..30

    def search(D, max_time):
        start_time = time.time()
        best_result = [None]

        # used_diffs[d] = True if difference d is already used (d in 1..D)
        # We only need to track up to D for Sidon check within [0,D]
        used_diffs = [False] * (D + 1)

        def dfs(S, covered_mask, depth):
            if time.time() > max_time:
                return False

            if (covered_mask & target_mask) == target_mask:
                best_result[0] = list(S)
                return True

            cur_max = S[-1] if S else 0
            if cur_max >= D:
                return False

            missing = target_mask & ~covered_mask
            num_missing = missing.bit_count()

            # Pruning: max elements we can still add
            max_add = min(D - cur_max, 32 - len(S))
            if max_add <= 0:
                return False

            # Tighter potential: each new element x covers at most len(S) new diffs
            # (x - s for s in S), plus diffs between new elements.
            # But we need to cover specific missing values, not just any diffs.
            # Upper bound on distinct new diffs: max_add * len(S) + max_add*(max_add-1)//2
            potential = max_add * len(S) + max_add * (max_add - 1) // 2
            if potential < num_missing:
                return False

            # Additional pruning: if even with perfect coverage we can't reach D
            # (not directly applicable, but if cur_max is too high relative to missing)
            # Skip for now.

            cands = []
            for x in range(cur_max + 1, D + 1):
                ok = True
                new_mask = 0
                for s in S:
                    d = x - s
                    if d <= D and used_diffs[d]:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                if not ok:
                    continue

                new_cover = new_mask & missing
                cnt = new_cover.bit_count()
                if cnt > 0:
                    cands.append((cnt, x, new_mask))

            if not cands:
                return False

            cands.sort(key=lambda t: (-t[0], t[1]))
            # Limit candidates to keep branching manageable
            cands = cands[:30]

            for cnt, x, new_mask in cands:
                changes = []
                for s in S:
                    d = x - s
                    changes.append(d)
                    if d <= D:
                        used_diffs[d] = True

                S.append(x)
                dfs(S, covered_mask | new_mask, depth + 1)
                S.pop()

                for d in changes:
                    if d <= D:
                        used_diffs[d] = False

                if best_result[0] is not None:
                    return True
                if time.time() > max_time:
                    return False

            return False

        dfs([0], 0, 0)
        return best_result[0]

    # Search for minimum D, with increasing time budget for smaller D
    for D in range(30, 80):
        if time.time() > deadline:
            break
        time_left = deadline - time.time()
        # Give more time to smaller D since they are more constrained
        if D <= 35:
            time_per_D = min(8.0, time_left * 0.5)
        elif D <= 45:
            time_per_D = min(5.0, time_left * 0.4)
        else:
            time_per_D = min(3.0, time_left * 0.3)

        if time_per_D < 0.5:
            break

        res = search(D, time.time() + time_per_D)
        if res is not None:
            return res

    # Fallback: greedy construction (should not be reached)
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target and time.time() < deadline:
        best_x = None
        best_cover = -1
        for x in range(max(S) + 1, max(S) + 300):
            ok = True
            cover = 0
            for s in S:
                d = x - s
                if d in used:
                    ok = False
                    break
                if d in target:
                    cover += 1
            if ok and cover > best_cover:
                best_cover = cover
                best_x = x
        if best_x is None:
            break
        S.append(best_x)
        for s in S[:-1]:
            used.add(best_x - s)
        target -= used
    return S
