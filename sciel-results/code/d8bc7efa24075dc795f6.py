import numpy as np
import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    # Generate all 4-subsets of [0, 10]
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)  # C(11,4) = 330

    # Precompute support masks for fast intersection
    sup_masks = []
    for s in supports:
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        sup_masks.append(mask)

    # Build conflict graph
    conflict = [set() for _ in range(N_sup)]
    for i in range(N_sup):
        for j in range(i + 1, N_sup):
            common = sup_masks[i] & sup_masks[j]
            if bin(common).count('1') >= 3:
                conflict[i].add(j)
                conflict[j].add(i)

    def local_search(selected, rng, max_iters=30):
        """Improve via 1-2 exchange moves."""
        selected = set(selected)
        improved = True
        iters = 0

        while improved and iters < max_iters and time.time() < deadline - 10.0:
            improved = False
            iters += 1

            for v in list(selected):
                if time.time() > deadline - 10.0:
                    return selected

                sel_minus_v = selected - {v}

                # Find all nodes compatible with sel_minus_v
                potential = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    if not (conflict[x] & sel_minus_v):
                        potential.append(x)

                # Find a compatible pair in potential
                L = len(potential)
                found = False
                for i in range(L):
                    if found:
                        break
                    u1 = potential[i]
                    for j in range(i + 1, L):
                        u2 = potential[j]
                        if u2 not in conflict[u1]:
                            selected.remove(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break

        return selected

    def find_max_iset(rng):
        """Find maximum independent set using multiple greedy starts + local search."""
        best_set = set()
        best_size = 0

        n_starts = 30
        for s in range(n_starts):
            if time.time() > deadline - 15.0:
                break

            # Randomized greedy
            avail = set(range(N_sup))
            selected = set()

            while avail and time.time() < deadline - 15.0:
                candidates = list(avail)
                rng.shuffle(candidates)

                scored = []
                for u in candidates:
                    deg = len(conflict[u] & avail)
                    score = deg + rng.random() * 0.1
                    scored.append((score, u))

                scored.sort()
                best_u = scored[0][1]

                selected.add(best_u)
                to_remove = {best_u} | (conflict[best_u] & avail)
                avail -= to_remove

            selected = local_search(selected, rng, max_iters=20)

            if len(selected) > best_size:
                best_size = len(selected)
                best_set = set(selected)

        return best_set, best_size

    rng = np.random.default_rng(12345)
    best_set, best_size = find_max_iset(rng)

    # Additional local search passes if time permits
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            improved_set = local_search(best_set, rng, max_iters=10)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set

    # Generate vectors
    vectors = []
    for i in sorted(best_set):
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)

    if not vectors:
        sup = supports[0]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)

    return np.array(vectors, dtype=np.int8)
