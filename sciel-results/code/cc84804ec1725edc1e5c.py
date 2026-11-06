import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 150s kill

    # Finite field GF(4) arithmetic for algebraic construction
    # GF(4) = {0, 1, alpha, alpha^2} where alpha^2 = alpha + 1
    # Represent elements as integers 0-3 with 2=alpha, 3=alpha^2
    gf4_add = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
    gf4_mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]

    def gf4_neg(x):
        return x  # characteristic 2, so -x = x

    def gf4_pow(x, n):
        result = 1
        for _ in range(n):
            result = gf4_mul[result][x]
        return result

    # Vertices: elements of GF(4)^3 = 64 points, but we'll use a subset
    # Actually for n=34, we need a specific construction.
    # Known construction: use AG(3,4) minus a plane or similar.
    # Let's try: vertices are all (a,b,c) in GF(4)^3 with a+b+c = 0 (a plane, 16 points)
    # That's only 16. We need 34.
    # Alternative: use all of GF(4)^3 = 64 points and take a subset.
    # Or use a different algebraic structure.

    # Simpler approach: use the known fact that R(4,5;3) > 34 exists.
    # Let me try a quadratic form based coloring on GF(4)^3.
    # Color triple (u,v,w) based on whether some polynomial is a square in GF(4).

    # For now, use a hybrid: try algebraic construction, fall back to search if needed.

    # Algebraic construction attempt:
    # Use GF(4)^2 = 16 points. Not enough.
    # Use a different approach: label vertices 0..n-1 and use modular arithmetic.

    # Let me try: for n=34, use a carefully chosen modular coloring.
    # The key insight: avoid red K4 and blue K5.

    # Try: color(i,j,k) = 0 if (i+j+k) mod 3 == 0, else 1
    # This is the same as the seed's initialization. Let me check if it works for larger n.

    def try_modular(n):
        triples = list(combinations(range(n), 3))
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        # Verify
        for quad in combinations(range(n), 4):
            all_red = all(colors[list(combinations(range(n),3)).index(t)] == 0 for t in combinations(quad,3))
            if all_red:
                return None
        for quint in combinations(range(n), 5):
            all_blue = all(colors[list(combinations(range(n),3)).index(t)] == 1 for t in combinations(quint,3))
            if all_blue:
                return None
        return colors

    # Quick check for small n
    best_n = 1
    best_colors = [0]

    # Try modular for increasing n
    for n in range(1, 35):
        if time.time() > deadline - 5:
            break
        res = try_modular(n)
        if res is not None:
            best_n = n
            best_colors = res
        else:
            break

    # If we got to n >= 34, great. Otherwise, try search for higher n.
    if best_n < 34:
        # Use local search to improve
        def search_for_n(n, time_limit, seed=0):
            M = comb(n, 3)
            triples = list(combinations(range(n), 3))
            tri_to_idx = {tri: i for i, tri in enumerate(triples)}

            quad_triples = []
            for quad in combinations(range(n), 4):
                idxs = [tri_to_idx[tuple(x for j, x in enumerate(quad) if j != i)] for i in range(4)]
                quad_triples.append(idxs)
            quint_triples = []
            for quint in combinations(range(n), 5):
                idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
                quint_triples.append(idxs)

            quad_triples = np.array(quad_triples, dtype=np.int32)
            quint_triples = np.array(quint_triples, dtype=np.int32)

            triple_to_4sets = [[] for _ in range(M)]
            for qidx, tidxs in enumerate(quad_triples):
                for t in tidxs:
                    triple_to_4sets[t].append(qidx)
            triple_to_5sets = [[] for _ in range(M)]
            for qidx, tidxs in enumerate(quint_triples):
                for t in tidxs:
                    triple_to_5sets[t].append(qidx)
            triple_to_4sets = [np.array(x, dtype=np.int32) for x in triple_to_4sets]
            triple_to_5sets = [np.array(x, dtype=np.int32) for x in triple_to_5sets]

            nwords = (M + 63) // 64
            quad_masks = np.zeros((len(quad_triples), nwords), dtype=np.uint64)
            for qidx, tidxs in enumerate(quad_triples):
                for t in tidxs:
                    quad_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
            quint_masks = np.zeros((len(quint_triples), nwords), dtype=np.uint64)
            for qidx, tidxs in enumerate(quint_triples):
                for t in tidxs:
                    quint_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))

            rng = np.random.RandomState(seed)

            colors = np.zeros(M, dtype=np.int8)
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a + b + c) % 3 == 0 else 1

            red_set = np.zeros(nwords, dtype=np.uint64)
            blue_set = np.zeros(nwords, dtype=np.uint64)
            for t in range(M):
                if colors[t] == 0:
                    red_set[t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
                else:
                    blue_set[t >> 6] |= (np.uint64(1) << np.uint64(t & 63))

            def count_violations(red_set, blue_set):
                r4 = np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1).sum()
                b5 = np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1).sum()
                return int(r4) + int(b5)

            def get_violating_quads(red_set):
                return np.where(np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1))[0]

            def get_violating_quints(blue_set):
                return np.where(np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1))[0]

            def flip_score(t, red_set, blue_set):
                old = colors[t]
                if old == 0:
                    qmask = quad_masks[triple_to_4sets[t]]
                    broken = np.all((qmask & red_set[None, :]) == qmask, axis=1).sum()
                    qmask = quint_masks[triple_to_5sets[t]]
                    without_t = qmask & ~(np.uint64(1) << np.uint64(t & 63))
                    created = np.all((without_t & blue_set[None, :]) == without_t, axis=1).sum()
                    return int(broken) - int(created)
                else:
                    qmask = quint_masks[triple_to_5sets[t]]
                    broken = np.all((qmask & blue_set[None, :]) == qmask, axis=1).sum()
                    qmask = quad_masks[triple_to_4sets[t]]
                    without_t = qmask & ~(np.uint64(1) << np.uint64(t & 63))
                    created = np.all((without_t & red_set[None, :]) == without_t, axis=1).sum()
                    return int(broken) - int(created)

            def apply_flip(t, red_set, blue_set):
                old = colors[t]
                bit = np.uint64(1) << np.uint64(t & 63)
                w = t >> 6
                if old == 0:
                    red_set[w] &= ~bit
                    blue_set[w] |= bit
                    colors[t] = 1
                else:
                    blue_set[w] &= ~bit
                    red_set[w] |= bit
                    colors[t] = 0
                return old

            for it in range(50000):
                if time.time() > start_time + time_limit:
                    break
                viol_quads = get_violating_quads(red_set)
                viol_quints = get_violating_quints(blue_set)
                if len(viol_quads) == 0 and len(viol_quints) == 0:
                    return colors.tolist()
                candidate_set = set()
                for q in viol_quads:
                    for t in quad_triples[q]:
                        candidate_set.add(int(t))
                for q in viol_quints:
                    for t in quint_triples[q]:
                        candidate_set.add(int(t))
                if not candidate_set:
                    break
                best_t = -1
                best_s = 0
                for t in candidate_set:
                    s = flip_score(t, red_set, blue_set)
                    if s > best_s:
                        best_s = s
                        best_t = t
                if best_t == -1:
                    t = int(rng.randint(0, M))
                    apply_flip(t, red_set, blue_set)
                    continue
                apply_flip(best_t, red_set, blue_set)

            T = 2.0
            cooling = 0.9999
            min_T = 0.01
            while time.time() < start_time + time_limit:
                t = int(rng.randint(0, M))
                s = flip_score(t, red_set, blue_set)
                apply_flip(t, red_set, blue_set)
                if s < 0 or (T > min_T and rng.random() < np.exp(-s / T)):
                    pass
                else:
                    apply_flip(t, red_set, blue_set)
                if count_violations(red_set, blue_set) == 0:
                    return colors.tolist()
                T *= cooling
            return None

        for n in range(best_n + 1, 35):
            remaining = deadline - time.time()
            if remaining < 5.0:
                break
            per_n_budget = min(remaining - 5.0, 20.0)
            if per_n_budget <= 1:
                break
            found = None
            for r in range(3):
                if time.time() > deadline - 5.0:
                    break
                t_limit = min(per_n_budget / 3, deadline - time.time() - 5.0)
                if t_limit <= 0.5:
                    break
                res = search_for_n(n, t_limit, seed=n * 1000 + r * 13 + 7)
                if res is not None:
                    found = res
                    break
            if found is not None:
                best_n = n
                best_colors = found
            else:
                break

    return {'n': best_n, 'colors': best_colors}
