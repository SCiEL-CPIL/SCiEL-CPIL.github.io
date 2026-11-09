import time
import random

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill

    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    N = 30  # differences 1..30

    # ---------- helpers ----------
    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i + 1:])

    def is_sidon(S):
        n = len(S)
        if n < 2:
            return True
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in seen:
                    return False
                seen.add(d)
        return True

    def covers_all(S):
        diffs = get_diffs(S)
        return target_diffs.issubset(diffs)

    # ---------- beam search (structural core) ----------
    # State: (S_tuple, used_diffs_set, covered_mask, max_val)
    # We build strictly increasing sets (0 first). Sidon enforced via used_diffs.

    def valid_add(S, used_diffs, x):
        """Return (new_diffs_list, new_mask_contrib) if x can be added, else None."""
        if x in S:
            return None
        new_diffs = []
        for s in S:
            d = x - s
            if d <= 0:
                continue
            if d in used_diffs:
                return None
            new_diffs.append(d)
        return new_diffs

    def get_candidates(S, used_diffs, covered_mask, max_val, rng):
        missing = [d for d in range(1, N + 1) if not (covered_mask & (1 << (d - 1)))]
        if not missing:
            return []

        candidates = set()
        # For each missing difference and each existing element, x = s + d
        for d in missing:
            for s in S:
                x = s + d
                if x > max_val or x > 10**6:
                    continue
                candidates.add(x)
        # Also allow a small window of "natural" next values to escape
        if S:
            last = S[-1]
            for k in range(1, 8):
                candidates.add(last + k)

        valid = []
        for x in candidates:
            if x <= max_val or x in S:
                continue
            res = valid_add(S, used_diffs, x)
            if res is None:
                continue
            new_diffs = res
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= N:
                    new_mask |= (1 << (d - 1))
            new_missing = N - bin(new_mask).count('1')
            # Score: prefer smaller x (keeps max low), then fewer missing
            valid.append((x, new_mask, new_diffs, new_missing))

        valid.sort(key=lambda t: (t[0], t[3]))
        return valid[:8]

    beam_width = 2000
    rng = random.Random(12345)

    S0 = (0,)
    used0 = set()
    mask0 = 0
    beam = [(S0, used0, mask0, 0)]

    best_max = float('inf')
    best_set = None

    for step in range(32):
        if time.time() - start_time > deadline:
            break
        new_beam = []
        for S, used, mask, max_val in beam:
            if time.time() - start_time > deadline:
                break
            if mask == full_mask:
                if max_val < best_max:
                    best_max = max_val
                    best_set = list(S)
                continue
            cands = get_candidates(S, used, mask, max_val, rng)
            for x, new_mask, new_diffs, _ in cands:
                new_S = S + (x,)
                new_used = used.union(new_diffs)
                new_max = x
                new_beam.append((new_S, new_used, new_mask, new_max))
                if time.time() - start_time > deadline:
                    break

        if not new_beam:
            break

        new_beam.sort(key=lambda t: (t[3], bin(t[2]).count('0') & 0, len(t[0])))
        # Re-sort: primary max_val, secondary missing count
        new_beam.sort(key=lambda t: (t[3], N - bin(t[2]).count('1')))

        seen = set()
        unique = []
        for st in new_beam:
            key = st[0]
            if key in seen:
                continue
            seen.add(key)
            unique.append(st)
            if len(unique) >= beam_width:
                break
        beam = unique

        # Prune states that can't beat best
        if best_max < float('inf'):
            beam = [st for st in beam if st[3] < best_max]
            if not beam:
                break

    # ---------- local search polish (refinement machinery) ----------
    if best_set is not None:
        S = sorted(best_set)
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            # Try to shrink each element from largest down to smallest
            for i in range(len(S) - 1, 0, -1):
                if (time.time() - start_time) > deadline:
                    break
                s_i = S[i]
                lower = S[i - 1] + 1
                if s_i <= lower:
                    continue
                # Candidate smaller values: just below s_i, and a few randoms
                trial = set()
                for off in range(1, min(12, s_i - lower + 1)):
                    trial.add(s_i - off)
                for _ in range(4):
                    trial.add(rng.randint(lower, s_i - 1))
                # Try from smallest up so we find the tightest valid shrink
                for new_val in sorted(trial):
                    if new_val < lower or new_val >= s_i:
                        continue
                    S_new = S[:i] + [new_val] + S[i + 1:]
                    S_new.sort()
                    if is_sidon(S_new) and covers_all(S_new):
                        S = S_new
                        improved = True
                        break
                if improved:
                    break
        return S

    # ---------- fallback greedy ----------
    S = [0]
    covered = set()
    used_diffs = set()
    for _ in range(31):
        if time.time() - start_time > deadline - 2:
            break
        missing = target_diffs - covered
        if not missing:
            break
        best_c = None
        best_score = None
        for d in missing:
            for s in S:
                x = s + d
                if x <= 0 or x in S:
                    continue
                ok = True
                nd = []
                for s2 in S:
                    diff = x - s2
                    if diff <= 0:
                        continue
                    if diff in used_diffs:
                        ok = False
                        break
                    nd.append(diff)
                if not ok:
                    continue
                nc = sum(1 for diff in nd if diff in missing)
                score = (-nc, x)
                if best_score is None or score < best_score:
                    best_score = score
                    best_c = x
        if best_c is None:
            x = 2 * max(S) + 3
            while True:
                ok = all((x - s) not in used_diffs and (x - s) > 0 for s in S)
                if ok:
                    break
                x += 1
            best_c = x
        x = best_c
        S.append(x)
        S.sort()
        for s in S[:-1]:
            d = x - s
            used_diffs.add(d)
            if 1 <= d <= 30:
                covered.add(d)
    return S
