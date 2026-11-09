import numpy as np
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    support = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)
    n = len(support)

    si = support[:, None, None, None]
    sj = support[None, :, None, None]
    sk = support[None, None, :, None]
    sl = support[None, None, None, :]
    cond = (si + sj + sk) < (2.0 * sl)
    cond_per_l = [cond[:, :, :, l] for l in range(n)]

    def compute_P(w):
        P = 0.0
        for l in range(n):
            inner_l = np.einsum('ijk,i,j,k->', cond_per_l[l], w, w, w)
            P += w[l] * inner_l
        return P

    def objective(params):
        w = np.zeros(n)
        w[:n-1] = params
        w[n-1] = 1.0 - np.sum(params)
        if np.any(w < -1e-9):
            return 1.0
        w = np.clip(w, 0, None)
        return -compute_P(w)

    best_w = None
    best_score = -np.inf

    starts = [np.ones(n-1) / n]
    rng = np.random.RandomState(42)
    for _ in range(30):
        x = rng.dirichlet(np.ones(n))
        starts.append(x[:n-1])

    for start in starts:
        try:
            res = minimize(objective, start, method='SLSQP',
                          bounds=[(0, 1)] * (n-1),
                          options={'maxiter': 2000, 'ftol': 1e-15})
            if res.success:
                w = np.zeros(n)
                w[:n-1] = res.x
                w[n-1] = 1.0 - np.sum(res.x)
                w = np.clip(w, 0, None)
                total = np.sum(w)
                if total > 1e-9:
                    w = w / total
                    score = compute_P(w)
                    if score > best_score:
                        best_score = score
                        best_w = w
        except Exception:
            continue

    if best_w is None:
        best_w = np.ones(n) / n

    return support, best_w
