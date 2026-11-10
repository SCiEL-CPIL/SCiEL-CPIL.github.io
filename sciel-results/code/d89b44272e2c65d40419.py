import time
import random
import math
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    n = 14
    m = 35

    # Precompute all pairs (91 total)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    P = len(all_pairs)  # 91

    # Candidate value sets: emphasize multiplicative structure (product collisions)
    # and additive structure (sum collisions). Distinct, positive, <=10000.
    candidate_values = [
        # Powers of 2 -> products are powers of 2 (very few distinct products)
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # Divisor-rich small numbers -> many product collisions
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 18, 20, 24, 30, 36],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 40],
        [3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20],
        [2, 4, 6, 9, 10, 12, 15, 16, 18, 20, 24, 27, 30, 36],
        [1, 2, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30],
        [2, 3, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 27, 30],
        # Geometric-ish with product structure
        [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 9999],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
        [2, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 9000, 9500],
        # Arithmetic progression (sum collisions)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        # Mixed: small factor-rich + a few large
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 144, 192],
        # Powers of 3 (product structure)
        [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 2, 6, 18, 54, 162],
        # Highly composite mix
        [1, 2, 3, 4, 6, 12, 24, 36, 48, 60, 72, 84, 96, 120],
    ]

    # Add a few random factor-rich sets
    random.seed(20240517)
    base_small = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 27, 28, 30, 32, 36, 40, 42, 45, 48, 50, 54, 56, 60, 63, 64, 70, 72, 75, 80, 81, 84, 90, 96, 100]
    for _ in range(40):
        vals = random.sample(base_small, n)
        candidate_values.append(vals)

    def raw_from_edge_set(values, edge_set):
        sums = set()
        prods = set()
        for idx in edge_set:
            i, j = all_pairs[idx]
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def greedy_init(values, seed):
        rng = random.Random(seed)
        ps = [values[i] + values[j] for i, j in all_pairs]
        pp = [values[i] * values[j] for i, j in all_pairs]
        used_s = set()
        used_p = set()
        selected = []
        remaining = list(range(P))
        rng.shuffle(remaining)
        for _ in range(m):
            best_idx = -1
            best_score = float('inf')
            best_tie = -1
            for k in remaining:
                s = ps[k]
                p = pp[k]
                ns = len(used_s) + (1 if s not in used_s else 0)
                np_ = len(used_p) + (1 if p not in used_p else 0)
                score = max(ns, np_)
                tie = (1 if s in used_s else 0) + (1 if p in used_p else 0)
                if score < best_score or (score == best_score and tie > best_tie):
                    best_score = score
                    best_tie = tie
                    best_idx = k
            selected.append(best_idx)
            used_s.add(ps[best_idx])
            used_p.add(pp[best_idx])
            remaining.remove(best_idx)
        return selected

    def simulated_annealing(values, initial_edges, max_iters, seed):
        rng = random.Random(seed)
        ps = [values[i] + values[j] for i, j in all_pairs]
        pp = [values[i] * values[j] for i, j in all_pairs]

        edge_set = set(initial_edges)
        unselected = set(range(P)) - edge_set
        sum_counts = Counter()
        prod_counts = Counter()
        for idx in edge_set:
            sum_counts[ps[idx]] += 1
            prod_counts[pp[idx]] += 1

        def get_raw():
            return max(len(sum_counts), len(prod_counts))

        current_raw = get_raw()
        best_raw = current_raw
        best_edge_set = set(edge_set)

        T = 4.0
        T_min = 0.05
        cooling = 0.99995

        for it in range(max_iters):
            if time.time() > deadline:
                break

            if rng.random() < 0.85:
                # 1-swap
                remove_idx = rng.choice(list(edge_set))
                rs, rp = ps[remove_idx], pp[remove_idx]
                # pick a random add
                add_idx = rng.choice(list(unselected))
                if add_idx == remove_idx:
                    continue
                as_, ap = ps[add_idx], pp[add_idx]

                # apply
                sum_counts[rs] -= 1
                if sum_counts[rs] == 0:
                    del sum_counts[rs]
                sum_counts[as_] += 1
                prod_counts[rp] -= 1
                if prod_counts[rp] == 0:
                    del prod_counts[rp]
                prod_counts[ap] += 1

                new_raw = get_raw()
                delta = new_raw - current_raw
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    edge_set.remove(remove_idx)
                    edge_set.add(add_idx)
                    unselected.add(remove_idx)
                    unselected.remove(add_idx)
                    current_raw = new_raw
                    if current_raw < best_raw:
                        best_raw = current_raw
                        best_edge_set = set(edge_set)
                else:
                    # revert
                    sum_counts[rs] += 1
                    sum_counts[as_] -= 1
                    if sum_counts[as_] == 0:
                        del sum_counts[as_]
                    prod_counts[rp] += 1
                    prod_counts[ap] -= 1
                    if prod_counts[ap] == 0:
                        del prod_counts[ap]
            else:
                # 2-swap
                if len(edge_set) < 2 or len(unselected) < 2:
                    continue
                rem1, rem2 = rng.sample(list(edge_set), 2)
                add1, add2 = rng.sample(list(unselected), 2)
                if rem1 in (add1, add2) or rem2 in (add1, add2):
                    continue
                s1, p1 = ps[rem1], pp[rem1]
                s2, p2 = ps[rem2], pp[rem2]
                sa1, pa1 = ps[add1], pp[add1]
                sa2, pa2 = ps[add2], pp[add2]

                sum_counts[s1] -= 1
                if sum_counts[s1] == 0:
                    del sum_counts[s1]
                prod_counts[p1] -= 1
                if prod_counts[p1] == 0:
                    del prod_counts[p1]
                sum_counts[s2] -= 1
                if sum_counts[s2] == 0:
                    del sum_counts[s2]
                prod_counts[p2] -= 1
                if prod_counts[p2] == 0:
                    del prod_counts[p2]
                sum_counts[sa1] += 1
                prod_counts[pa1] += 1
                sum_counts[sa2] += 1
                prod_counts[pa2] += 1

                new_raw = get_raw()
                delta = new_raw - current_raw
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    edge_set.discard(rem1)
                    edge_set.discard(rem2)
                    edge_set.add(add1)
                    edge_set.add(add2)
                    unselected.add(rem1)
                    unselected.add(rem2)
                    unselected.discard(add1)
                    unselected.discard(add2)
                    current_raw = new_raw
                    if current_raw < best_raw:
                        best_raw = current_raw
                        best_edge_set = set(edge_set)
                else:
                    # revert
                    sum_counts[sa1] -= 1
                    if sum_counts[sa1] == 0:
                        del sum_counts[sa1]
                    prod_counts[pa1] -= 1
                    if prod_counts[pa1] == 0:
                        del prod_counts[pa1]
                    sum_counts[sa2] -= 1
                    if sum_counts[sa2] == 0:
                        del sum_counts[sa2]
                    prod_counts[pa2] -= 1
                    if prod_counts[pa2] == 0:
                        del prod_counts[pa2]
                    sum_counts[s1] += 1
                    prod_counts[p1] += 1
                    sum_counts[s2] += 1
                    prod_counts[p2] += 1

            T *= cooling
            if T < T_min:
                T = T_min

        return list(best_edge_set), best_raw

    best_raw = float('inf')
    best_values = None
    best_edges = None

    # Prioritize candidate sets by an optimistic lower bound:
    # min possible distinct sums/prods over all pairs
    def lower_bound(values):
        ps = set()
        pp = set()
        for i, j in all_pairs:
            ps.add(values[i] + values[j])
            pp.add(values[i] * values[j])
        return max(len(ps), len(pp))

    # Sort candidates by their theoretical lower bound (best first)
    scored = []
    for vi, values in enumerate(candidate_values):
        if len(set(values)) != n or any(v < 1 or v > 10000 for v in values):
            continue
        lb = lower_bound(values)
        scored.append((lb, vi, values))
    scored.sort(key=lambda x: x[0])

    for lb, vi, values in scored:
        if time.time() > deadline:
            break
        # More restarts for promising (low lower-bound) sets
        n_restarts = 4 if lb <= 40 else 2
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            greedy_edges = greedy_init(values, seed=restart * 7919 + vi * 131)
            # Larger SA budget
            max_iters = 60000 if lb <= 40 else 30000
            sa_edges, sa_raw = simulated_annealing(values, greedy_edges, max_iters, seed=restart * 104729 + vi * 17)
            if sa_raw < best_raw:
                best_raw = sa_raw
                best_values = [int(v) for v in values]
                best_edges = [[int(i), int(j)] for i, j in (all_pairs[idx] for idx in sa_edges)]
                if best_raw <= 12:
                    return {'values': best_values, 'edges': best_edges}

    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n)][:m]

    return {'values': best_values, 'edges': best_edges}
