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

    degrees = [len(adj[i]) for i in range(N2)]

    best_set = set()
    best_count = 0
    rng = np.random.default_rng(42)

    for trial in range(50):
        if time.time() > deadline - 15:
            break

        if trial == 0:
            order = np.argsort(degrees, kind='stable')
        else:
            order = rng.permutation(N2)

        selected = set()
        for i in order:
            if not any(n in selected for n in adj[i]):
                selected.add(i)

        # Local search: iterative improvement
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add any unselected vertex
            unselected = [i for i in range(N2) if i not in selected]
            rng.shuffle(unselected)
            for i in unselected:
                if not any(n in selected for n in adj[i]):
                    selected.add(i)
                    improved = True
            # Try swaps: remove one, add two
            if not improved:
                for i in list(selected):
                    if time.time() > deadline - 5:
                        break
                    neighbors_of_i = adj[i]
                    # Candidates to add: not in selected, not neighbor of i
                    candidates = [j for j in range(N2) if j not in selected and j not in neighbors_of_i]
                    # Check which candidates conflict with selected - i
                    valid = []
                    for j in candidates:
                        if not any(n in selected for n in adj[j] if n != i):
                            valid.append(j)
                    if len(valid) >= 2:
                        # Try adding two
                        for a, b in itertools.combinations(valid, 2):
                            if a not in adj[b]:
                                selected.remove(i)
                                selected.add(a)
                                selected.add(b)
                                improved = True
                                break
                        if improved:
                            break

        if len(selected) > best_count:
            best_count = len(selected)
            best_set = selected.copy()

    final_indices = list(best_set)
    vectors = np.vstack([type1, type2[final_indices]])
    return vectors.astype(np.int64)
