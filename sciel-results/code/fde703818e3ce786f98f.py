import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s margin before 90s kill

    # Precompute constraints
    # For each position x (0-based, representing position x+1),
    # list of (i1, i2, i3) such that (i1+1, i2+1, i3+1, x+1) form a 4-AP
    constraints = [[] for _ in range(128)]
    for x in range(128):
        pos = x + 1
        for d in range(1, (pos - 1) // 3 + 1):
            a = pos - 3 * d
            if a < 1:
                break
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            constraints[x].append((i1, i2, i3))

    def try_greedy(max_c, heuristic='simple', seed=0):
        """
        Try to construct a valid coloring using at most max_c colors.
        heuristic: 'simple' = prefer used colors, smallest first
                   'least_constraining' = avoid creating 3-same-in-AP situations
        """
        if seed != 0:
            random.seed(seed)

        colors = [None] * 128
        used = set()

        for x in range(128):
            if time.time() > deadline:
                return None

            valid_colors = []
            for c in range(max_c):
                ok = True
                for i1, i2, i3 in constraints[x]:
                    c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                    if len({c1, c2, c3, c}) < 3:
                        ok = False
                        break
                if ok:
                    valid_colors.append(c)

            if not valid_colors:
                return None

            if heuristic == 'simple':
                # Prefer used colors, then smallest new
                valid_colors.sort(key=lambda c: (0 if c in used else 1, c))
                chosen = valid_colors[0]
            elif heuristic == 'least_constraining':
                # Score each valid color by how many future constraints it might violate
                # or how "dangerous" it is (creates 2-same in APs that could become 3-same)
                best_score = float('inf')
                best_color = valid_colors[0]
                for c in valid_colors:
                    danger = 0
                    # Count how many existing APs this color creates with exactly 2 same colors
                    # (which could become 3 same if a future color matches)
                    for i1, i2, i3 in constraints[x]:
                        c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                        # Already valid (len >= 3), but check if we're creating a "near miss"
                        # If two of c1,c2,c3 equal c, that's a bit risky
                        count_c = sum(1 for v in [c1, c2, c3] if v == c)
                        danger += count_c
                    used_penalty = 0 if c in used else 1
                    score = danger * 2 + used_penalty
                    if score < best_score:
                        best_score = score
                        best_color = c
                chosen = best_color
            else:
                # Random from valid
                chosen = random.choice(valid_colors)

            colors[x] = chosen
            used.add(chosen)

        return colors

    def try_backtracking(max_c, timeout=5.0):
        """
        Backtracking search with at most max_c colors.
        Returns a valid coloring or None.
        """
        start = time.time()
        colors = [None] * 128
        used = set()

        def backtrack(x):
            if time.time() - start > timeout:
                return False
            if x == 128:
                return True

            valid_colors = []
            for c in range(max_c):
                ok = True
                for i1, i2, i3 in constraints[x]:
                    c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                    if len({c1, c2, c3, c}) < 3:
                        ok = False
                        break
                if ok:
                    valid_colors.append(c)

            if not valid_colors:
                return False

            # Prefer used colors first
            valid_colors.sort(key=lambda c: (0 if c in used else 1, c))

            for c in valid_colors:
                colors[x] = c
                old_used = len(used)
                used.add(c)
                if backtrack(x + 1):
                    return True
                colors[x] = None
                if len(used) > old_used:
                    used.remove(c)
            return False

        if backtrack(0):
            return colors
        return None

    # Strategy: Try 3 colors first with various heuristics
    for heuristic in ['simple', 'least_constraining']:
        res = try_greedy(3, heuristic)
        if res is not None:
            return res

    # Try randomized greedy with 3 colors
    for seed in range(20):
        if time.time() > deadline:
            break
        res = try_greedy(3, 'random', seed)
        if res is not None:
            return res

    # Try backtracking with 3 colors (short timeout)
    if time.time() < deadline - 10:
        res = try_backtracking(3, timeout=5.0)
        if res is not None:
            return res

    # Try 4 colors
    for heuristic in ['simple', 'least_constraining']:
        res = try_greedy(4, heuristic)
        if res is not None:
            return res

    for seed in range(20):
        if time.time() > deadline:
            break
        res = try_greedy(4, 'random', seed)
        if res is not None:
            return res

    if time.time() < deadline - 10:
        res = try_backtracking(4, timeout=5.0)
        if res is not None:
            return res

    # Try 5 colors
    for heuristic in ['simple', 'least_constraining']:
        res = try_greedy(5, heuristic)
        if res is not None:
            return res

    for seed in range(20):
        if time.time() > deadline:
            break
        res = try_greedy(5, 'random', seed)
        if res is not None:
            return res

    if time.time() < deadline - 10:
        res = try_backtracking(5, timeout=5.0)
        if res is not None:
            return res

    # Fallback: simple greedy with up to 128 colors
    colors = [None] * 128
    for x in range(128):
        if time.time() > deadline:
            break
        found = False
        for c in range(128):
            ok = True
            for i1, i2, i3 in constraints[x]:
                if len({colors[i1], colors[i2], colors[i3], c}) < 3:
                    ok = False
                    break
            if ok:
                colors[x] = c
                found = True
                break
        if not found:
            # Should not happen with 128 colors
            colors[x] = 0

    return colors
