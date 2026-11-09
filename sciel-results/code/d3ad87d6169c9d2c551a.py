import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 128

    # constraints[idx] = list of (i1,i2,i3) 0-based such that idx is the 4th term
    constraints = [[] for _ in range(N)]
    for x in range(1, N + 1):
        idx = x - 1
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            if a < 1:
                break
            constraints[idx].append((a - 1, a + d - 1, a + 2 * d - 1))

    def validate(colors):
        if colors is None:
            return False
        if len(colors) != N:
            return False
        for i in range(N):
            for (i1, i2, i3) in constraints[i]:
                if len({colors[i], colors[i1], colors[i2], colors[i3]}) < 3:
                    return False
        return True

    # ---- fast greedy (deterministic, tries to keep colors minimal) ----
    def greedy(max_c):
        colors = [-1] * N
        for x in range(N):
            if time.time() > deadline - 1.0:
                return None
            forbidden = [False] * max_c
            for (i1, i2, i3) in constraints[x]:
                c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                if c1 == c2 and c2 == c3:
                    forbidden[c1] = True
                elif c1 == c2:
                    forbidden[c1] = True
                    forbidden[c3] = True
                elif c1 == c3:
                    forbidden[c1] = True
                    forbidden[c2] = True
                elif c2 == c3:
                    forbidden[c2] = True
                    forbidden[c1] = True
            # prefer already-used colors to keep distinct count low
            used = set(colors[:x])
            cands = [c for c in range(max_c) if not forbidden[c]]
            cands.sort(key=lambda c: (0 if c in used else 1, c))
            if not cands:
                return None
            colors[x] = cands[0]
        return colors

    # ---- backtracking with forward checking ----
    def bt_search(max_c, time_limit):
        colors = [-1] * N
        bt_start = time.time()
        node_count = 0

        def backtrack(idx):
            nonlocal node_count
            node_count += 1
            if time.time() > bt_start + time_limit:
                return False
            if idx == N:
                return True
            forbidden = [False] * max_c
            for (i1, i2, i3) in constraints[idx]:
                c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                if c1 == c2 and c2 == c3:
                    forbidden[c1] = True
                elif c1 == c2:
                    forbidden[c1] = True
                    forbidden[c3] = True
                elif c1 == c3:
                    forbidden[c1] = True
                    forbidden[c2] = True
                elif c2 == c3:
                    forbidden[c2] = True
                    forbidden[c1] = True

            # forward checking: ensure every later position still has a color
            for c in range(max_c):
                if forbidden[c]:
                    continue
                colors[idx] = c
                ok = True
                for j in range(idx + 1, N):
                    f = [False] * max_c
                    for (i1, i2, i3) in constraints[j]:
                        c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                        if c1 == c2 and c2 == c3:
                            f[c1] = True
                        elif c1 == c2:
                            f[c1] = True
                            f[c3] = True
                        elif c1 == c3:
                            f[c1] = True
                            f[c2] = True
                        elif c2 == c3:
                            f[c2] = True
                            f[c1] = True
                        if f[0] and f[1] and f[2]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    if backtrack(idx + 1):
                        return True
                colors[idx] = -1
            return False

        if backtrack(0):
            return colors
        return None

    # Try backtracking first (robust, finds valid 3-coloring if exists)
    res = bt_search(3, 55.0)
    if res is not None and validate(res):
        return res

    # Fallback: greedy with 3 colors
    if time.time() < deadline - 1.0:
        res = greedy(3)
        if res is not None and validate(res):
            return res

    # Fallback: greedy with 4 colors (always valid, but higher label count)
    if time.time() < deadline - 0.5:
        res = greedy(4)
        if res is not None and validate(res):
            return res

    # Last-resort: original seed loop
    colors = []
    for x in range(1, N + 1):
        for color in range(N):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2))
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors
