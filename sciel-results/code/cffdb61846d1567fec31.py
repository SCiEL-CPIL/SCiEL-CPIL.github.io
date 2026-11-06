"""Search for a small-diameter Sidon set (B_2) containing 0
whose positive differences include 1..30. Minimizes the largest element."""
import time

_WALL_LIMIT = 150.0
_DEADLINE_FRACTION = 0.92


def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT * _DEADLINE_FRACTION

    best_set = None
    best_max = 10**6

    def check_time():
        return time.time() > deadline

    def solve_n(n):
        nonlocal best_set, best_max
        S = [0]
        used_diffs = set()

        def backtrack(k, cur_max):
            nonlocal best_set, best_max
            if check_time():
                return False

            if k == n:
                for d in range(1, 31):
                    if d not in used_diffs:
                        return False
                if cur_max < best_max:
                    best_max = cur_max
                    best_set = list(S)
                return True

            if cur_max >= best_max:
                return False

            remaining = n - (k + 1)
            if remaining < 0:
                return False

            missing = set(range(1, 31)) - used_diffs
            lo = S[-1] + 1
            hi = min(best_max - 1, 10**6)

            candidates = set()
            for s in S:
                for m in missing:
                    x = s + m
                    if lo <= x <= hi:
                        candidates.add(x)

            if not candidates:
                candidates = {lo}

            for x in sorted(candidates):
                if check_time():
                    return False
                if x >= best_max:
                    break

                new_diffs = []
                ok = True
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue

                S.append(x)
                used_diffs.update(new_diffs)

                backtrack(k + 1, x)

                S.pop()
                used_diffs.difference_update(new_diffs)

            return False

        backtrack(0, 0)

    for n in range(9, 17):
        solve_n(n)
        if check_time():
            break

    if best_set is None:
        S = [0]
        used = set()
        for d in range(1, 31):
            if d in used:
                continue
            placed = False
            for v in S:
                for x in (v + d, v - d):
                    if x >= 0 and x not in S:
                        nd = [abs(x - v2) for v2 in S]
                        if len(set(nd)) == len(nd) and all(n2 not in used for n2 in nd):
                            S.append(x)
                            used.update(nd)
                            placed = True
                            break
                if placed:
                    break
            if not placed:
                y = 2 * max(S) + 3 * d + 1
                while True:
                    nd = [abs(y - v) for v in S] + [abs(y + d - v) for v in S] + [d]
                    if len(set(nd)) == len(nd) and all(n2 not in used for n2 in nd):
                        S.extend([y, y + d])
                        used.update(nd)
                        break
                    y += 1
        best_set = sorted(set(S))

    return best_set


if __name__ == "__main__":
    print(construct())
