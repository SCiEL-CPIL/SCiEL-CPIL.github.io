import time
import random
import sys
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def edge_key(i, j):
        if i > j:
            i, j = j, i
        return (i, j)

    def build_all_c6s(n):
        """Build list of all C6s as tuples of 6 edge keys."""
        c6s = []
        verts = list(range(n))
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = list(combo[1:])
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                edges = tuple(edge_key(path[k], path[(k+1) % 6]) for k in range(6))
                c6s.append(edges)
        return c6s

    def edge_to_c6s(c6s):
        """Map each edge to list of C6 indices containing it."""
        e2c = {}
        for ci, c6 in enumerate(c6s):
            for e in c6:
                if e not in e2c:
                    e2c[e] = []
                e2c[e].append(ci)
        return e2c

    def is_mono_c6(colors, c6):
        c0 = colors[c6[0]]
        for e in c6[1:]:
            if colors[e] != c0:
                return False
        return True

    def local_search(colors, n, c6s, e2c, time_budget, rng):
        start = time.time()
        all_edges = [edge_key(i, j) for i in range(n) for j in range(i + 1, n)]

        # Initial bad set
        bad_set = set()
        for ci, c6 in enumerate(c6s):
            if is_mono_c6(colors, c6):
                bad_set.add(ci)

        if not bad_set:
            return colors, 0

        best_bad = len(bad_set)
        best_colors = dict(colors)
        no_improve = 0
        max_no_improve = 50000

        for it in range(10000000):
            if time.time() > start + time_budget:
                break
            if not bad_set:
                break

            # Pick random bad C6
            ci = rng.choice(list(bad_set))
            c6 = c6s[ci]

            # Pick random edge in this C6
            e = rng.choice(c6)
            old_c = colors[e]

            # Try each new color, pick the one that minimizes bad C6s
            best_new_c = None
            best_delta = 999999
            for new_c in range(4):
                if new_c == old_c:
                    continue
                delta = 0
                for ci2 in e2c.get(e, []):
                    c6_2 = c6s[ci2]
                    was_mono = ci2 in bad_set
                    # Check if would be mono with new_c
                    all_match = True
                    for ed in c6_2:
                        if ed == e:
                            continue
                        if colors[ed] != new_c:
                            all_match = False
                            break
                    would_be_mono = all_match
                    if was_mono and not would_be_mono:
                        delta -= 1
                    elif not was_mono and would_be_mono:
                        delta += 1
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c

            if best_new_c is not None and best_delta < 0:
                # Apply change
                for ci2 in e2c.get(e, []):
                    c6_2 = c6s[ci2]
                    was_mono = ci2 in bad_set
                    all_match = True
                    for ed in c6_2:
                        if ed == e:
                            continue
                        if colors[ed] != best_new_c:
                            all_match = False
                            break
                    would_be_mono = all_match
                    if was_mono and not would_be_mono:
                        bad_set.discard(ci2)
                    elif not was_mono and would_be_mono:
                        bad_set.add(ci2)
                colors[e] = best_new_c
                no_improve = 0
            else:
                # Random move (simulated annealing style)
                new_c = rng.randint(0, 3)
                while new_c == old_c:
                    new_c = rng.randint(0, 3)
                for ci2 in e2c.get(e, []):
                    c6_2 = c6s[ci2]
                    was_mono = ci2 in bad_set
                    all_match = True
                    for ed in c6_2:
                        if ed == e:
                            continue
                        if colors[ed] != new_c:
                            all_match = False
                            break
                    would_be_mono = all_match
                    if was_mono and not would_be_mono:
                        bad_set.discard(ci2)
                    elif not was_mono and would_be_mono:
                        bad_set.add(ci2)
                colors[e] = new_c
                no_improve += 1

            if len(bad_set) < best_bad:
                best_bad = len(bad_set)
                best_colors = dict(colors)
                if best_bad == 0:
                    break
            elif no_improve > max_no_improve:
                break

        return best_colors, best_bad

    # Main
    target_n = 17
    c6s = build_all_c6s(target_n)
    e2c = edge_to_c6s(c6s)
    rng = random.Random(42)

    time_per_attempt = 8.0
    max_attempts = int((deadline - start_time - 5) / time_per_attempt)

    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break

        # Different inits
        colors = {}
        if attempt % 4 == 0:
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    colors[edge_key(i, j)] = rng.randint(0, 3)
        elif attempt % 4 == 1:
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    colors[edge_key(i, j)] = (j - i) % 4
        elif attempt % 4 == 2:
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    colors[edge_key(i, j)] = ((i + j) % 4)
        else:
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    colors[edge_key(i, j)] = ((j - i) * 2 + i) % 4

        colors, bad = local_search(colors, target_n, c6s, e2c, time_per_attempt, rng)
        if bad == 0:
            m = target_n * (target_n - 1) // 2
            colors_out = [0] * m
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    idx = i * (target_n - 2) - i * (i - 1) // 2 + j - 1
                    colors_out[idx] = colors[edge_key(i, j)]
            return {'n': target_n, 'colors': colors_out}

    # Try n=16
    if time.time() < deadline - 10:
        target_n = 16
        c6s = build_all_c6s(target_n)
        e2c = edge_to_c6s(c6s)
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            colors = {}
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    colors[edge_key(i, j)] = rng.randint(0, 3)
            colors, bad = local_search(colors, target_n, c6s, e2c, 8.0, rng)
            if bad == 0:
                m = target_n * (target_n - 1) // 2
                colors_out = [0] * m
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        idx = i * (target_n - 2) - i * (i - 1) // 2 + j - 1
                        colors_out[idx] = colors[edge_key(i, j)]
                return {'n': target_n, 'colors': colors_out}

    n = 5
    m = n * (n - 1) // 2
    return {'n': n, 'colors': [0] * m}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
