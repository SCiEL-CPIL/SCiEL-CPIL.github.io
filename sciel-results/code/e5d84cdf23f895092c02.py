import time

def construct():
    start = time.time()
    deadline = start + 140.0
    needed = set(range(1, 31))

    def elapsed():
        return time.time() - start

    def find_set(D, n):
        def backtrack(current, all_diffs):
            if elapsed() > deadline:
                return None
            if len(current) == n:
                if needed.issubset(all_diffs):
                    return list(current)
                return None
            remaining = n - len(current)
            uncovered = needed - all_diffs
            max_new = remaining * len(current) + remaining * (remaining - 1) // 2
            if len(uncovered) > max_new:
                return None
            candidates = []
            for x in range(current[-1] + 1, D + 1):
                new_diffs = set()
                valid = True
                for c in current:
                    d = x - c
                    if d in all_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    score = len(new_diffs & (needed - all_diffs))
                    candidates.append((score, x, new_diffs))
            candidates.sort(key=lambda t: (-t[0], t[1]))
            for score, x, new_diffs in candidates:
                if elapsed() > deadline:
                    return None
                result = backtrack(current + [x], all_diffs | new_diffs)
                if result is not None:
                    return result
            return None
        return backtrack([0], set())

    for D in range(28, 50):
        if elapsed() > deadline - 10:
            break
        for n in range(8, 33):
            if elapsed() > deadline - 10:
                break
            result = find_set(D, n)
            if result is not None:
                s = sorted(result)
                diffs = set()
                ok = True
                for i in range(len(s)):
                    for j in range(i + 1, len(s)):
                        d = s[j] - s[i]
                        if d in diffs:
                            ok = False
                            break
                        diffs.add(d)
                    if not ok:
                        break
                if ok and needed.issubset(diffs):
                    return s

    # Fallback greedy
    a = [0]
    def get_diffs(s):
        return set(y - x for i, x in enumerate(s) for y in s[i + 1:])
    for d in range(1, 31):
        if d in get_diffs(a):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            diffs = get_diffs(trial)
            if len(diffs) == len(trial) * (len(trial) - 1) // 2:
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                diffs = get_diffs(trial)
                if len(diffs) == len(trial) * (len(trial) - 1) // 2:
                    a = trial
                    break
                y += 1
    return a
