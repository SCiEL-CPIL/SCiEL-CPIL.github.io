"""Algebraic construction over GF(256) + genetic algorithm for R(4,5;3)."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # --- GF(256) arithmetic with primitive poly x^8+x^4+x^3+x^2+1 (0x11D) ---
    P = 0x11D
    exp_table = [0] * 512
    log_table = [0] * 256
    x = 1
    for i in range(255):
        exp_table[i] = x
        log_table[x] = i
        x <<= 1
        if x & 0x100:
            x ^= P
    for i in range(255, 512):
        exp_table[i] = exp_table[i - 255]

    def gf_mul(a, b):
        if a == 0 or b == 0:
            return 0
        return exp_table[log_table[a] + log_table[b]]

    def gf_add(a, b):
        return a ^ b

    def gf_sub(a, b):
        return a ^ b

    def gf_pow(a, n):
        if a == 0:
            return 0 if n > 0 else 1
        return exp_table[log_table[a] * n % 255]

    # --- Precompute all triples and K4/K5 structures for a given n ---
    def build_structures(n):
        triples = list(combinations(range(n), 3))
        m = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}

        red_k4s = []
        for quad in combinations(range(n), 4):
            red_k4s.append((
                triple_idx[(quad[0], quad[1], quad[2])],
                triple_idx[(quad[0], quad[1], quad[3])],
                triple_idx[(quad[0], quad[2], quad[3])],
                triple_idx[(quad[1], quad[2], quad[3])]
            ))

        blue_k5s = []
        for quint in combinations(range(n), 5):
            blue_k5s.append(tuple(
                triple_idx[tri] for tri in combinations(quint, 3)
            ))

        return triples, m, triple_idx, red_k4s, blue_k5s

    # --- Evaluate a coloring defined by vertex labels and polynomial ---
    def evaluate_coloring(labels, poly_coeffs, n, triples, red_k4s, blue_k5s):
        """
        labels: list of n GF(256) values assigned to vertices 0..n-1
        poly_coeffs: list of (monomial_type, coeff) pairs
          monomial_type: 0=a*b*c, 1=a^2*b, 2=a*b^2, 3=b^2*c, 4=b*c^2, 5=a^2*c, 6=a*c^2, 7=a^3, 8=b^3, 9=c^3, 10=a*b, 11=b*c, 12=a*c, 13=a, 14=b, 15=c, 16=1
        Returns (red_viol, blue_viol, colors)
        """
        m = len(triples)
        colors = [0] * m
        for i, (a, b, c) in enumerate(triples):
            va, vb, vc = labels[a], labels[b], labels[c]
            val = 0
            for mtype, coeff in poly_coeffs:
                if coeff == 0:
                    continue
                if mtype == 0:
                    term = gf_mul(gf_mul(va, vb), vc)
                elif mtype == 1:
                    term = gf_mul(gf_mul(va, va), vb)
                elif mtype == 2:
                    term = gf_mul(va, gf_mul(vb, vb))
                elif mtype == 3:
                    term = gf_mul(gf_mul(vb, vb), vc)
                elif mtype == 4:
                    term = gf_mul(vb, gf_mul(vc, vc))
                elif mtype == 5:
                    term = gf_mul(gf_mul(va, va), vc)
                elif mtype == 6:
                    term = gf_mul(va, gf_mul(vc, vc))
                elif mtype == 7:
                    term = gf_pow(va, 3)
                elif mtype == 8:
                    term = gf_pow(vb, 3)
                elif mtype == 9:
                    term = gf_pow(vc, 3)
                elif mtype == 10:
                    term = gf_mul(va, vb)
                elif mtype == 11:
                    term = gf_mul(vb, vc)
                elif mtype == 12:
                    term = gf_mul(va, vc)
                elif mtype == 13:
                    term = va
                elif mtype == 14:
                    term = vb
                elif mtype == 15:
                    term = vc
                else:  # 16
                    term = coeff  # constant
                val = gf_add(val, term)
            colors[i] = 1 if val != 0 else 0

        red_viol = 0
        for k4 in red_k4s:
            if all(colors[t] == 0 for t in k4):
                red_viol += 1

        blue_viol = 0
        for k5 in blue_k5s:
            all_blue = True
            for t in k5:
                if colors[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_viol += 1

        return red_viol, blue_viol, colors

    # --- Fast violation count using bitmasks for K4s ---
    def count_violations_fast(colors, red_k4s, blue_k5s):
        red_viol = 0
        for k4 in red_k4s:
            if colors[k4[0]] == 0 and colors[k4[1]] == 0 and colors[k4[2]] == 0 and colors[k4[3]] == 0:
                red_viol += 1
        blue_viol = 0
        for k5 in blue_k5s:
            cnt = 0
            for t in k5:
                if colors[t] == 1:
                    cnt += 1
                    if cnt == 10:
                        blue_viol += 1
                        break
        return red_viol, blue_viol

    # --- Simple local search on the color array (fallback) ---
    def local_search(colors, n, triples, red_k4s, blue_k5s, triple_to_k4s, triple_to_k5s, time_limit):
        m = len(triples)
        is_red_viol = [False] * len(red_k4s)
        is_blue_viol = [False] * len(blue_k5s)
        red_viol = 0
        for i, k4 in enumerate(red_k4s):
            if all(colors[t] == 0 for t in k4):
                is_red_viol[i] = True
                red_viol += 1
        blue_viol = 0
        for i, k5 in enumerate(blue_k5s):
            if all(colors[t] == 1 for t in k5):
                is_blue_viol[i] = True
                blue_viol += 1

        best_red, best_blue = red_viol, blue_viol
        best_colors = colors[:]
        rng = random.Random(42)

        for it in range(200000):
            if time.time() > time_limit:
                break
            if red_viol == 0 and blue_viol == 0:
                return colors

            if red_viol > 0 and (blue_viol == 0 or rng.random() < 0.5):
                # Pick random red violation
                while True:
                    vi = rng.randint(0, len(red_k4s) - 1)
                    if is_red_viol[vi]:
                        break
                k4 = red_k4s[vi]
                flip_t = k4[rng.randint(0, 3)]
                colors[flip_t] ^= 1

                for k4_idx in triple_to_k4s[flip_t]:
                    k4_triples = red_k4s[k4_idx]
                    is_all = (colors[k4_triples[0]] == 0 and colors[k4_triples[1]] == 0 and
                              colors[k4_triples[2]] == 0 and colors[k4_triples[3]] == 0)
                    if is_all:
                        if not is_red_viol[k4_idx]:
                            is_red_viol[k4_idx] = True
                            red_viol += 1
                    else:
                        if is_red_viol[k4_idx]:
                            is_red_viol[k4_idx] = False
                            red_viol -= 1

                for k5_idx in triple_to_k5s[flip_t]:
                    k5_triples = blue_k5s[k5_idx]
                    cnt = 0
                    for t in k5_triples:
                        if colors[t] == 1:
                            cnt += 1
                        if cnt < 10:
                            pass
                    is_all_blue = (cnt == 10)
                    if is_all_blue:
                        if not is_blue_viol[k5_idx]:
                            is_blue_viol[k5_idx] = True
                            blue_viol += 1
                    else:
                        if is_blue_viol[k5_idx]:
                            is_blue_viol[k5_idx] = False
                            blue_viol -= 1
            else:
                if blue_viol == 0:
                    continue
                while True:
                    vi = rng.randint(0, len(blue_k5s) - 1)
                    if is_blue_viol[vi]:
                        break
                k5 = blue_k5s[vi]
                flip_t = k5[rng.randint(0, 9)]
                colors[flip_t] ^= 1

                for k4_idx in triple_to_k4s[flip_t]:
                    k4_triples = red_k4s[k4_idx]
                    is_all = (colors[k4_triples[0]] == 0 and colors[k4_triples[1]] == 0 and
                              colors[k4_triples[2]] == 0 and colors[k4_triples[3]] == 0)
                    if is_all:
                        if not is_red_viol[k4_idx]:
                            is_red_viol[k4_idx] = True
                            red_viol += 1
                    else:
                        if is_red_viol[k4_idx]:
                            is_red_viol[k4_idx] = False
                            red_viol -= 1

                for k5_idx in triple_to_k5s[flip_t]:
                    k5_triples = blue_k5s[k5_idx]
                    cnt = 0
                    for t in k5_triples:
                        if colors[t] == 1:
                            cnt += 1
                    is_all_blue = (cnt == 10)
                    if is_all_blue:
                        if not is_blue_viol[k5_idx]:
                            is_blue_viol[k5_idx] = True
                            blue_viol += 1
                    else:
                        if is_blue_viol[k5_idx]:
                            is_blue_viol[k5_idx] = False
                            blue_viol -= 1

            if red_viol < best_red or (red_viol == best_red and blue_viol < best_blue):
                best_red = red_viol
                best_blue = blue_viol
                best_colors = colors[:]

        return best_colors

    # --- Genetic algorithm over label permutations ---
    def genetic_algo(n, triples, red_k4s, blue_k5s, time_limit):
        # Use a subset of GF(256) elements as labels
        # Pick n distinct nonzero elements
        all_elements = list(range(1, 256))
        rng = random.Random(7)

        # Initial population: random permutations of n chosen elements
        pop_size = 30
        # Use first n elements of GF(256) as base, then permute
        base_labels = list(range(1, n + 1))

        # Try several polynomial forms
        poly_forms = [
            # (coeffs for a*b*c + a*b + b*c + a*c)
            [(0, 1), (10, 1), (11, 1), (12, 1)],
            [(0, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1)],
            [(0, 1), (7, 1), (8, 1), (9, 1)],
            [(0, 1), (10, 2), (11, 3), (12, 5)],
            [(0, 1), (10, 1), (11, 1), (12, 1), (16, 1)],
            [(0, 3), (10, 5), (11, 7), (12, 11)],
        ]

        population = []
        for _ in range(pop_size):
            labels = base_labels[:]
            rng.shuffle(labels)
            population.append(labels)

        best_score = float('inf')
        best_labels = None
        best_poly = None
        best_colors = None

        for gen in range(60):
            if time.time() > time_limit:
                break

            for poly_idx, poly in enumerate(poly_forms):
                if time.time() > time_limit:
                    break
                for labels in population:
                    if time.time() > time_limit:
                        break
                    rv, bv, colors = evaluate_coloring(labels, poly, n, triples, red_k4s, blue_k5s)
                    score = rv * 100 + bv
                    if score < best_score:
                        best_score = score
                        best_labels = labels[:]
                        best_poly = poly
                        best_colors = colors[:]
                    if score == 0:
                        return colors

            # Selection and crossover
            scored = []
            for labels in population:
                if best_poly is not None:
                    rv, bv, _ = evaluate_coloring(labels, best_poly, n, triples, red_k4s, blue_k5s)
                    scored.append((rv * 100 + bv, labels))
            scored.sort(key=lambda x: x[0])

            new_pop = []
            # Elitism
            for i in range(min(5, len(scored))):
                new_pop.append(scored[i][1][:])

            while len(new_pop) < pop_size:
                # Tournament selection
                p1 = scored[rng.randint(0, len(scored) - 1)][1]
                p2 = scored[rng.randint(0, len(scored) - 1)][1]
                parent = p1 if rng.random() < 0.5 else p2

                # Mutation: swap two elements
                child = parent[:]
                if rng.random() < 0.7:
                    i, j = rng.sample(range(n), 2)
                    child[i], child[j] = child[j], child[i]
                if rng.random() < 0.3:
                    i, j = rng.sample(range(n), 2)
                    child[i], child[j] = child[j], child[i]
                new_pop.append(child)

            population = new_pop

        return best_colors if best_colors is not None else None

    # --- Main: try n from 34 down ---
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 10:
            break

        triples, m, triple_idx, red_k4s, blue_k5s = build_structures(n)

        # Build reverse maps
        triple_to_k4s = [[] for _ in range(m)]
        for i, k4 in enumerate(red_k4s):
            for t in k4:
                triple_to_k4s[t].append(i)
        triple_to_k5s = [[] for _ in range(m)]
        for i, k5 in enumerate(blue_k5s):
            for t in k5:
                triple_to_k5s[t].append(i)

        # Phase 1: Genetic algorithm
        ga_time = min(60, deadline - time.time() - 10)
        if ga_time > 5:
            ga_colors = genetic_algo(n, triples, red_k4s, blue_k5s, start_time + ga_time)
            if ga_colors is not None:
                rv, bv = count_violations_fast(ga_colors, red_k4s, blue_k5s)
                if rv == 0 and bv == 0:
                    return {'n': n, 'colors': ga_colors}

        # Phase 2: Random init + local search
        remaining = deadline - time.time() - 5
        if remaining < 3:
            break

        rng = random.Random(n * 1000 + 42)
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            colors = [rng.randint(0, 1) for _ in range(m)]
            # Quick check
            rv, bv = count_violations_fast(colors, red_k4s, blue_k5s)
            if rv == 0 and bv == 0:
                return {'n': n, 'colors': colors}
            # Local search
            ls_colors = local_search(colors, n, triples, red_k4s, blue_k5s,
                                     triple_to_k4s, triple_to_k5s,
                                     min(deadline - 5, time.time() + remaining / 5))
            rv, bv = count_violations_fast(ls_colors, red_k4s, blue_k5s)
            if rv == 0 and bv == 0:
                return {'n': n, 'colors': ls_colors}

    # Fallback: small n with simple construction
    n = 9
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
