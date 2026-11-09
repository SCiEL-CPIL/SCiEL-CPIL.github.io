import time

def construct():
    start_time = time.time()
    deadline = 145.0

    def check_time():
        return time.time() - start_time > deadline

    best_solution = None
    best_max = float('inf')

    def solve(S, used_diffs, covered):
        nonlocal best_solution, best_max

        if check_time():
            return

        if covered.issuperset(range(1, 31)):
            if max(S) < best_max:
                best_max = max(S)
                best_solution = S[:]
            return

        if len(S) >= 32 or (S and max(S) >= best_max):
            return

        max_s = max(S) if S else 0
        missing = set(range(1, 31)) - covered

        # Pruning: each missing d must be coverable by adding a future element
        for d in missing:
            possible = False
            for x in S:
                if x + d > max_s:
                    possible = True
                    break
            if not possible:
                return

        # Generate candidates
        candidates = {}
        for d in missing:
            for x in S:
                c = x + d
                if c > max_s:
                    if c not in candidates:
                        candidates[c] = 0
                    candidates[c] += 1

        for c in range(max_s + 1, max_s + 15):
            if c not in candidates:
                candidates[c] = 0

        sorted_cands = sorted(candidates.keys(), key=lambda c: (-candidates[c], c))

        for c in sorted_cands:
            if check_time():
                return
            if c >= best_max:
                break

            new_diffs = [c - x for x in S]
            if any(d in used_diffs for d in new_diffs):
                continue

            new_S = S + [c]
            new_used = used_diffs | set(new_diffs)
            new_covered = covered | (set(new_diffs) & set(range(1, 31)))

            solve(new_S, new_used, new_covered)

            if best_solution is not None and best_max <= max_s:
                return

    solve([0], set(), set())

    if best_solution is not None:
        return best_solution

    # Fallback greedy
    S = [0]
    used = set()
    covered = set()
    target = set(range(1, 31))
    for _ in range(32):
        if covered.issuperset(target) or check_time():
            break
        best_c = None
        best_gain = 0
        for c in range(1, 500):
            if c in S or c <= max(S):
                continue
            nd = [c - x for x in S]
            if any(d in used for d in nd):
                continue
            g = len(set(nd) & target)
            if g > best_gain:
                best_gain = g
                best_c = c
        if best_c is None:
            break
        S.append(best_c)
        used.update(c - x for x in S[:-1])
        covered = used & target
    return S

