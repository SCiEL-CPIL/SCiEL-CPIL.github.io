import numpy as np
import time

def construct():
    t0 = time.time()
    N = 12
    M = 40
    total = 1 << N  # 4096

    masks = np.arange(total, dtype=np.int64)
    a_2d = masks[:, None]
    b_2d = masks[None, :]
    is_sup = ((a_2d & b_2d) == a_2d) & (a_2d != b_2d)

    def get_score(s):
        return float(s @ (is_sup @ s))

    def greedy_init(noise=0.0, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        sel = np.zeros(total, dtype=np.float64)
        for step in range(M):
            if time.time() - t0 > 45:
                break
            vec1 = is_sup @ sel
            vec2 = is_sup.T @ sel
            g = vec1 + vec2
            g[sel > 0] = -1
            if noise > 0:
                g = g + rng.normal(0, noise, total)
            c = int(np.argmax(g))
            sel[c] = 1
        return sel

    def local_search(sel, max_iter=30):
        if time.time() - t0 > 55:
            return sel, get_score(sel)
        best_score = get_score(sel)
        for iteration in range(max_iter):
            if time.time() - t0 > 55:
                break
            vec1 = is_sup @ sel
            vec2 = is_sup.T @ sel
            g = vec1 + vec2

            selected_indices = np.where(sel > 0)[0]
            best_delta = -np.inf
            best_a = None
            best_c = None

            for idx_a in selected_indices:
                a = int(idx_a)
                scores_c = g - g[a] - (is_sup[:, a].astype(np.float64) + is_sup[a, :].astype(np.float64))
                scores_c[sel > 0] = -np.inf
                c = int(np.argmax(scores_c))
                delta = float(scores_c[c])
                if delta > best_delta:
                    best_delta = delta
                    best_a = a
                    best_c = c

            if best_a is not None and best_delta > 1e-9:
                sel[best_a] = 0
                sel[best_c] = 1
                best_score = get_score(sel)
            else:
                break

        return sel, best_score

    best_sel = None
    best_score = -1

    # Deterministic greedy
    sel = greedy_init(noise=0.0)
    sel, sc = local_search(sel, max_iter=30)
    if sc > best_score:
        best_score = sc
        best_sel = sel.copy()

    # Randomized greedy + local search
    for trial in range(10):
        if time.time() - t0 > 48:
            break
        sel = greedy_init(noise=0.5, seed=trial)
        sel, sc = local_search(sel, max_iter=20)
        if sc > best_score:
            best_score = sc
            best_sel = sel.copy()

    # Convert to lists
    result = []
    for mask in np.where(best_sel > 0)[0]:
        result.append([b for b in range(N) if (int(mask) >> b) & 1])

    return result
