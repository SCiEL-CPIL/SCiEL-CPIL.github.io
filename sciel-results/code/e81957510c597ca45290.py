import random
import time

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute all non-constant 4-term APs (0-indexed positions)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                v += 1
        return v

    def check_valid(colors):
        return count_violations(colors) == 0

    random.seed(42)
    best_colors = None
    best_ncolors = 128

    # Try to find a 3-coloring via local search
    for attempt in range(300):
        if time.time() > deadline - 3.0:
            break

        colors = [random.randint(0, 2) for _ in range(N)]
        v = count_violations(colors)

        for it in range(8000):
            if time.time() > deadline - 1.0:
                break

            if v == 0:
                ncolors = len(set(colors))
                if ncolors < best_ncolors:
                    best_ncolors = ncolors
                    best_colors = colors[:]
                break

            # Try to improve by flipping one position
            improved = False
            # Try all positions in random order
            positions = list(range(N))
            random.shuffle(positions)
            for pos in positions:
                old_c = colors[pos]
                for new_c in range(3):
                    if new_c == old_c:
                        continue
                    colors[pos] = new_c
                    new_v = count_violations(colors)
                    if new_v < v:
                        v = new_v
                        improved = True
                        break
                    else:
                        colors[pos] = old_c
                if improved:
                    break

            if not improved:
                # Perturb: flip up to 5 random positions
                for _ in range(random.randint(3, 6)):
                    pos = random.randint(0, N - 1)
                    colors[pos] = random.randint(0, 2)
                v = count_violations(colors)
                if v == 0:
                    ncolors = len(set(colors))
                    if ncolors < best_ncolors:
                        best_ncolors = ncolors
                        best_colors = colors[:]
                    break

        if best_colors is not None and best_ncolors <= 3:
            break

    if best_colors is not None and check_valid(best_colors):
        return best_colors

    # Fallback: sequential greedy with 3 colors, backtracking if needed
    # If 3 colors fail, try with more colors using the original approach
    colors = [-1] * N
    for x in range(N):
        found = False
        for color in range(3):
            ok = True
            for ap in aps:
                if x in ap:
                    others = [p for p in ap if p != x]
                    if all(p < x for p in others):
                        cset = {colors[p] for p in others}
                        cset.add(color)
                        if len(cset) < 3:
                            ok = False
                            break
            if ok:
                colors[x] = color
                found = True
                break
        if not found:
            # Allow 4th color as fallback
            for color in range(3, 128):
                ok = True
                for ap in aps:
                    if x in ap:
                        others = [p for p in ap if p != x]
                        if all(p < x for p in others):
                            cset = {colors[p] for p in others}
                            cset.add(color)
                            if len(cset) < 3:
                                ok = False
                                break
                if ok:
                    colors[x] = color
                    found = True
                    break
            if not found:
                colors[x] = 3  # Shouldn't happen

    if check_valid(colors):
        return colors

    # Ultimate fallback: original seed logic
    colors2 = []
    for x in range(1, N + 1):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    break
                cset = {color}
                for j in (1, 2, 3):
                    idx = x - 1 - j * d
                    if idx >= 0:
                        cset.add(colors2[idx])
                if len(cset) < 3:
                    valid = False
                    break
            if valid:
                for d in range(1, (x - 1) // 2 + 1):
                    if x - 1 - 2 * d < 0:
                        break
                    ok = False
                    for j in (1, 2):
                        idx = x - 1 - j * d
                        if idx >= 0 and colors2[idx] != color:
                            ok = True
                            break
                    if not ok:
                        valid = False
                        break
            if valid:
                colors2.append(color)
                break

    return colors2
