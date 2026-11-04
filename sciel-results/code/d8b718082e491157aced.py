"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    all_pos = list(combinations(range(D), 4))
    all_signs = list(product([1, -1], repeat=4))
    n_pos = len(all_pos)
    n_signs = len(all_signs)
    
    inter = [[0] * n_pos for _ in range(n_pos)]
    for i in range(n_pos):
        si = set(all_pos[i])
        for j in range(i + 1, n_pos):
            m = len(si & set(all_pos[j]))
            inter[i][j] = m
            inter[j][i] = m
        inter[i][i] = 4
    
    sign_dot = [[0] * n_signs for _ in range(n_signs)]
    for i in range(n_signs):
        for j in range(n_signs):
            d = 0
            for k in range(4):
                d += all_signs[i][k] * all_signs[j][k]
            sign_dot[i][j] = d
    
    selected = []
    
    for pos_idx in range(n_pos):
        for sign_idx in range(n_signs):
            if time.time() > deadline:
                break
            ok = True
            for (p_idx, s_idx) in selected:
                m = inter[pos_idx][p_idx]
                if m <= 2:
                    continue
                if m == 4:
                    if sign_dot[sign_idx][s_idx] > 2:
                        ok = False
                        break
                else:
                    pos_a = all_pos[pos_idx]
                    pos_b = all_pos[p_idx]
                    signs_a = all_signs[sign_idx]
                    signs_b = all_signs[s_idx]
                    sign_map_b = dict(zip(pos_b, signs_b))
                    dot = 0
                    for k in range(4):
                        if pos_a[k] in sign_map_b:
                            dot += signs_a[k] * sign_map_b[pos_a[k]]
                    if dot > 2:
                        ok = False
                        break
            if ok:
                selected.append((pos_idx, sign_idx))
    
    if selected:
        result = np.zeros((len(selected), D), dtype=np.int64)
        for i, (pos_idx, sign_idx) in enumerate(selected):
            pos = all_pos[pos_idx]
            signs = all_signs[sign_idx]
            for p, s in zip(pos, signs):
                result[i, p] = s
        return result
    else:
        vectors = []
        for i, j in combinations(range(D), 2):
            for si, sj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END

