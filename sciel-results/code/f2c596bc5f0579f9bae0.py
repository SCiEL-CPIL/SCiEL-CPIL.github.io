import numpy as np
import time

def construct():
    t0 = time.time()
    DEADLINE = 135
    n_coeffs = 24
    N = 4096

    k = np.arange(n_coeffs)
    n = np.arange(N)
    Z = np.exp(2j * np.pi * np.outer(k, n) / N)

    def evaluate_fast(c):
        vals = Z @ c
        return np.max(np.abs(vals))

    def evaluate_refined(c, top_k=8):
        vals = Z @ c
        abs_vals = np.abs(vals)
        idx = np.argpartition(abs_vals, -top_k)[-top_k:]
        best = float(abs_vals[idx].max())
        for i in idx:
            lo = (i - 1) * 2.0 * np.pi / N
            hi = (i + 1) * 2.0 * np.pi / N
            gr = (np.sqrt(5) - 1) / 2
            a, b = lo, hi
            c1 = b - gr * (b - a)
            c2 = a + gr * (b - a)
            for _ in range(50):
                f1 = np.abs(np.sum(c * np.exp(1j * k * c1)))
                f2 = np.abs(np.sum(c * np.exp(1j * k * c2)))
                if f1 < f2:
                    b = c2
                    c2 = c1
                    c1 = b - gr * (b - a)
                else:
                    a = c1
                    c1 = c2
                    c2 = a + gr * (b - a)
            mid = (a + b) / 2
            fmid = np.abs(np.sum(c * np.exp(1j * k * mid)))
            if fmid > best:
                best = fmid
        return best

    candidates = []
    p, q = [1], [1]
    while len(p) < n_coeffs:
        p, q = p + q, p + [-c for c in q]
    grs = np.array(p[:n_coeffs], dtype=np.float64)
    candidates.append(grs)
    for phase in range(1, n_coeffs):
        candidates.append(np.roll(grs, phase))
    candidates.append(grs[::-1])
    candidates.append(np.ones(n_coeffs))
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_coeffs)]))

    rng = np.random.RandomState(12345)
    for _ in range(50):
        candidates.append(rng.choice([-1.0, 1.0], size=n_coeffs))

    best_c = None
    best_val = float('inf')
    for c in candidates:
        val = evaluate_fast(c)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    start_candidates = [best_c.copy()]
    for c in candidates[:30]:
        val = evaluate_fast(c)
        if val < best_val * 1.05:
            start_candidates.append(c.copy())

    js, ks = np.where(np.triu(np.ones((n_coeffs, n_coeffs), dtype=bool), k=1))

    for start_c in start_candidates:
        if (time.time() - t0) > DEADLINE - 10:
            break
        current_c = start_c.copy()
        current_val = evaluate_fast(current_c)
        improved = True
        while improved and (time.time() - t0) < DEADLINE - 10:
            improved = False
            current_vals = Z @ current_c
            single_deltas = -2.0 * current_c[:, np.newaxis] * Z
            new_vals = current_vals[np.newaxis, :] + single_deltas
            new_maxes = np.max(np.abs(new_vals), axis=1)
            best_j = np.argmin(new_maxes)
            if new_maxes[best_j] < current_val - 1e-10:
                current_c[best_j] *= -1
                current_val = new_maxes[best_j]
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_c = current_c.copy()
            if (time.time() - t0) > DEADLINE - 10:
                break
            current_vals = Z @ current_c
            single_deltas = -2.0 * current_c[:, np.newaxis] * Z
            pair_deltas = single_deltas[js] + single_deltas[ks]
            pair_new_vals = current_vals[np.newaxis, :] + pair_deltas
            pair_maxes = np.max(np.abs(pair_new_vals), axis=1)
            best_pair_idx = np.argmin(pair_maxes)
            if pair_maxes[best_pair_idx] < current_val - 1e-10:
                j, k_ = js[best_pair_idx], ks[best_pair_idx]
                current_c[j] *= -1
                current_c[k_] *= -1
                current_val = pair_maxes[best_pair_idx]
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_c = current_c.copy()
            if (time.time() - t0) > DEADLINE - 10:
                break
            if rng.random() < 0.1:
                current_vals = Z @ current_c
                single_deltas = -2.0 * current_c[:, np.newaxis] * Z
                for _ in range(50):
                    triplet = rng.choice(n_coeffs, size=3, replace=False)
                    delta = single_deltas[triplet[0]] + single_deltas[triplet[1]] + single_deltas[triplet[2]]
                    new_v = current_vals + delta
                    new_max = np.max(np.abs(new_v))
                    if new_max < current_val - 1e-10:
                        current_c[triplet] *= -1
                        current_val = new_max
                        improved = True
                        if current_val < best_val:
                            best_val = current_val
                            best_c = current_c.copy()
                        break

    if (time.time() - t0) < DEADLINE - 5:
        current_c = best_c.copy()
        current_val = evaluate_fast(current_c)
        for _ in range(20):
            if (time.time() - t0) > DEADLINE - 5:
                break
            improved = False
            current_vals = Z @ current_c
            single_deltas = -2.0 * current_c[:, np.newaxis] * Z
            new_vals = current_vals[np.newaxis, :] + single_deltas
            new_maxes = np.max(np.abs(new_vals), axis=1)
            best_j = np.argmin(new_maxes)
            if new_maxes[best_j] < current_val - 1e-10:
                current_c[best_j] *= -1
                current_val = new_maxes[best_j]
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_c = current_c.copy()
            if not improved:
                break

    return [int(x) for x in best_c]
