import numpy as np
import itertools
import time
from collections import defaultdict

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145

    # Type 1: ±2e_i
    type1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            type1.append(v)

    # Type 2: ±e_i ± e_j ± e_k ± e_l
    type2 = []
    type2_info = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            type2.append(v)
            type2_info.append((idxs, signs))

    type1 = np.array(type1, dtype=np.int64)
    type2 = np.array(type2, dtype=np.int64)
    N2 = len(type2)

    # Build conflict graph
    adj = [set() for _ in range(N2)]
    lookup = {}
    for i, (idxs, signs) in enumerate(type2_info):
        key = (idxs, signs)
        lookup[key] = i

    for i, (idxs, signs) in enumerate(type2_info):
        for k in range(4):
            common_idx = idxs[:k] + idxs[k+1:]
            common_signs = signs[:k] + signs[k+1:]
            for j in range(D):
                if j in idxs:
                    continue
                v_idxs = tuple(sorted(common_idx + (j,)))
                sign_map = dict(zip(common_idx, common_signs))
                for s_j in [1, -1]:
                    sign_map[j] = s_j
                    v_signs = tuple(sign_map[idx] for idx in v_idxs)
                    key = (v_idxs, v_signs)
                    if key in lookup:
                        j_idx = lookup[key]
                        if j_idx != i:
                            adj[i].add(j_idx)
                            adj[j_idx].add(i)

    degrees = np.array([len(adj[i]) for i in range(N2)], dtype=np.int64)

    # Least Conflicts heuristic
    rng = np.random.default_rng(42)
    best_set = set()
    best_count = 0

    for trial in range(30):
        if time.time() > deadline - 15:
            break

        # Initialize: randomly select a subset
        if trial == 0:
            # Start with degree-ordered greedy as initial solution
            order = np.argsort(degrees, kind='stable')
            selected = set()
            for i in order:
                if not any(n in selected for n in adj[i]):
                    selected.add(i)
        else:
            # Random initialization with density ~1/avg_degree
            p = 1.0 / (np.mean(degrees) + 1)
            selected = set(i for i in range(N2) if rng.random() < p)

        # Least Conflicts iteration
        max_iters = 500
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Compute conflicts for each selected vertex
            conflicts = {}
            for i in selected:
                c = 0
                for n in adj[i]:
                    if n in selected:
                        c += 1
                conflicts[i] = c

            total_conflicts = sum(conflicts.values())
            if total_conflicts == 0:
                break

            # Find vertices with most conflicts
            max_conf = max(conflicts.values())
            if max_conf == 0:
                break

            # Choose a random vertex among those with max conflicts, or try to fix
            high_conf = [i for i, c in conflicts.items() if c >= max_conf]
            v = rng.choice(high_conf)

            # Strategy: either remove v, or move it (remove and add a non-neighbor)
            r = rng.random()
            if r < 0.4:
                # Remove v
                selected.remove(v)
            elif r < 0.7:
                # Try to add a random non-neighbor, non-selected vertex
                candidates = []
                for j in range(N2):
                    if j not in selected and j not in adj[v]:
                        # Check if j conflicts with current selected (excluding v's removal)
                        ok = True
                        for n in adj[j]:
                            if n in selected:
                                ok = False
                                break
                        if ok:
                            candidates.append(j)
                if candidates:
                    new_v = rng.choice(candidates)
                    selected.remove(v)
                    selected.add(new_v)
            else:
                # Just remove v
                selected.remove(v)

            # After each move, try to greedily add any vertex that fits
            unselected = [i for i in range(N2) if i not in selected]
            rng.shuffle(unselected)
            added = False
            for i in unselected:
                if not any(n in selected for n in adj[i]):
                    selected.add(i)
                    added = True

        if len(selected) > best_count:
            best_count = len(selected)
            best_set = selected.copy()

    # Final greedy fill
    unselected = [i for i in range(N2) if i not in best_set]
    rng.shuffle(unselected)
    for i in unselected:
        if not any(n in best_set for n in adj[i]):
            best_set.add(i)

    final_indices = list(best_set)
    vectors = np.vstack([type1, type2[final_indices]])
    return vectors.astype(np.int64)
