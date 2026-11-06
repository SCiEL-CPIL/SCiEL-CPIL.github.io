import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0

    # Greedy sequential construction: assign colors to triples in lexicographic order,
    # choosing each color to minimize newly formed partial monochromatic cliques.

    best_n = 9
    best_colors = None

    def greedy_construct(n, seed_offset=0):
        """Build a coloring greedily, triple by triple in lexicographic order."""
        triples = list(combinations(range(n), 3))
        colors = [None] * len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # For efficiency, we track counts of "almost complete" monochromatic cliques.
        # A red K4 is formed when we color the 4th triple of a 4-set red.
        # A blue K5 is formed when we color the 10th triple of a 5-set blue.

        # We'll use a simpler approach: for each new triple, count how many
        # red K4s and blue K5s would be completed by each color choice.

        for idx, (a, b, c) in enumerate(triples):
            if time.time() > deadline - 3.0:
                return None

            # Count red K4s that would be completed if (a,b,c) is red (color 0)
            red_k4_count = 0
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                # Check if (a,b,d), (a,c,d), (b,c,d) are all already red
                # and if they exist (i.e., have been colored)
                t_bd = (a, b, d) if a < b < d else (a, d, b) if a < d < b else (b, a, d) if b < a < d else (b, d, a) if b < d < a else (d, a, b) if d < a < b else (d, b, a)
                t_cd = (a, c, d) if a < c < d else (a, d, c) if a < d < c else (c, a, d) if c < a < d else (c, d, a) if c < d < a else (d, a, c) if d < a < c else (d, c, a)
                t_bd2 = (b, c, d) if b < c < d else (b, d, c) if b < d < c else (c, b, d) if c < b < d else (c, d, b) if c < d < b else (d, b, c) if d < b < c else (d, c, b)

                # Simplify: just use sorted tuples
                t1 = tuple(sorted((a, b, d)))
                t2 = tuple(sorted((a, c, d)))
                t3 = tuple(sorted((b, c, d)))

                i1 = triple_to_idx.get(t1)
                i2 = triple_to_idx.get(t2)
                i3 = triple_to_idx.get(t3)

                if i1 is not None and i2 is not None and i3 is not None:
                    if colors[i1] == 0 and colors[i2] == 0 and colors[i3] == 0:
                        red_k4_count += 1

            # Count blue K5s that would be completed if (a,b,c) is blue (color 1)
            blue_k5_count = 0
            others = [v for v in range(n) if v != a and v != b and v != c]
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    d = others[i]
                    e = others[j]
                    # Check if the other 9 triples of {a,b,c,d,e} are all blue
                    ts = [
                        tuple(sorted((a, b, d))), tuple(sorted((a, b, e))),
                        tuple(sorted((a, c, d))), tuple(sorted((a, c, e))),
                        tuple(sorted((a, d, e))), tuple(sorted((b, c, d))),
                        tuple(sorted((b, c, e))), tuple(sorted((b, d, e))),
                        tuple(sorted((c, d, e)))
                    ]
                    all_blue = True
                    for t in ts:
                        ti = triple_to_idx.get(t)
                        if ti is None or colors[ti] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        blue_k5_count += 1

            # Choose color: prefer the one that creates fewer new violations
            if red_k4_count < blue_k5_count:
                colors[idx] = 0
            elif blue_k5_count < red_k4_count:
                colors[idx] = 1
            else:
                # Tie: use a deterministic but varied choice based on seed
                colors[idx] = (idx + seed_offset) % 2

        return colors

    # Try greedy construction for n from 34 down
    for n in range(34, 9, -1):
        if time.time() > deadline - 3.0:
            break

        # Try a few different seeds for tie-breaking
        for seed_offset in range(5):
            if time.time() > deadline - 3.0:
                break

            colors = greedy_construct(n, seed_offset)
            if colors is None:
                continue

            # Verify the coloring is valid
            triples = list(combinations(range(n), 3))
            triple_to_idx = {t: i for i, t in enumerate(triples)}

            valid = True
            for comb4 in combinations(range(n), 4):
                a, b, c, d = comb4
                t1 = (a, b, c)
                t2 = (a, b, d)
                t3 = (a, c, d)
                t4 = (b, c, d)
                if (colors[triple_to_idx[t1]] == 0 and colors[triple_to_idx[t2]] == 0 and
                        colors[triple_to_idx[t3]] == 0 and colors[triple_to_idx[t4]] == 0):
                    valid = False
                    break

            if valid:
                for comb5 in combinations(range(n), 5):
                    a, b, c, d, e = comb5
                    ts = [
                        (a, b, c), (a, b, d), (a, b, e), (a, c, d), (a, c, e),
                        (a, d, e), (b, c, d), (b, c, e), (b, d, e), (c, d, e)
                    ]
                    all_blue = True
                    for t in ts:
                        if colors[triple_to_idx[t]] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        valid = False
                        break

            if valid:
                return {'n': n, 'colors': colors}

    # Fallback: return a valid small solution
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
