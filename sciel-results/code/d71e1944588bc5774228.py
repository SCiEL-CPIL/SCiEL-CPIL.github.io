import time
import random
import numpy as np

N = 256

def _build_aps_and_membership():
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    membership = [[] for _ in range(N)]
    for a, b, c, d in aps:
        membership[a].append((b, c, d))
        membership[b].append((a, c, d))
        membership[c].append((a, b, d))
        membership[d].append((a, b, c))
    return aps, membership

_aps, _membership = _build_aps_and_membership()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 256
    membership = _membership
    aps = _aps

    def count_mono(colors):
        cnt = 0
        for a, b, c, d in aps:
            if colors[a] == colors[b] == colors[c] == colors[d]:
                cnt += 1
        return cnt

    def get_delta(pos, colors):
        cp = colors[pos]
        delta = 0
        for o1, o2, o3 in membership[pos]:
            if colors[o1] == cp and colors[o2] == cp and colors[o3] == cp:
                delta -= 1
            elif colors[o1] == (1 - cp) and colors[o2] == (1 - cp) and colors[o3] == (1 - cp):
                delta += 1
        return delta

    # Generate diverse initial patterns
    best_colors = None
    best_count = float('inf')

    patterns = []
    for block in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32, 40, 50]:
        patterns.append([(i // block) % 2 for i in range(N)])
    for mod in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        for thr in range(1, mod):
            patterns.append([(1 if (i % mod) < thr else 0) for i in range(N)])
    rng = random.Random(42)
    for _ in range(30):
        patterns.append([rng.randint(0, 1) for _ in range(N)])

    for pat in patterns:
        cnt = count_mono(pat)
        if cnt < best_count:
            best_count = cnt
            best_colors = list(pat)

    colors = best_colors[:]
    current_count = best_count

    # Simulated annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.005
    no_improve = 0
    max_no_improve = 20000

    while time.time() < deadline:
        if T < min_T:
            T = 0.2
            no_improve = 0

        pos = random.randint(0, N - 1)
        delta = get_delta(pos, colors)

        if delta <= 0:
            colors[pos] = 1 - colors[pos]
            current_count += delta
            no_improve = 0
        else:
            if random.random() < pow(2.718281828, -delta / T):
                colors[pos] = 1 - colors[pos]
                current_count += delta
            no_improve += 1

        T *= cooling

        if no_improve > max_no_improve:
            improved = False
            for p in range(N):
                d = get_delta(p, colors)
                if d < 0:
                    colors[p] = 1 - colors[p]
                    current_count += d
                    improved = True
            if not improved:
                T = 0.5
            no_improve = 0

    return colors
