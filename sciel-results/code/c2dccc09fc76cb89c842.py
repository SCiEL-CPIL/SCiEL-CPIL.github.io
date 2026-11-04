"""Erdos minimum overlap: choose h_i in [0,1], sum = N/2, minimizing
max_s sum_i h_i (1 - h_{i-s})  (circular shifts).

We minimize a smooth surrogate (log-sum-exp soft-max) of the max-overlap
objective with a hard equality constraint (sum = N/2) enforced by a
reparameterization + projection, then polish with a final exact local
pass. A wall-clock deadline guard returns the best feasible result found.
"""
import time
import numpy as np

N = 100

# ---------- objective ----------
def _overlap_matrix(h):
    """O(s) for all circular shifts s in [0, N).
    O(s) = sum_i h_i * (1 - h_{i-s})  (circular)."""
    h = np.asarray(h, dtype=float)
    # h_{i-s} for all i,s: roll h by s for each s
    # Use FFT-free direct: O[s] = sum_i h_i - sum_i h_i h_{i-s}
    # sum_i h_i h_{i-s} is the circular cross-correlation of h with itself.
    # A(s) = sum_i h_i * h_{i-s}  ->  via ifft(fft(h)*conj(fft(h))) gives
    # A(s) = sum_i h_i h_{i+s}; we want h_{i-s} = h_{i+(-s)}, i.e. A(-s).
    H = np.fft.fft(h)
    A = np.fft.ifft(H * np.conj(H)).real   # A[s] = sum_i h_i h_{i+s}
    A = A % 1.0  # guard tiny negatives
    # O(s) = sum(h) - A(-s) = sum(h) - A[(-s) mod N]
    s = np.arange(N)
    O = h.sum() - A[(-s) % N]
    return O

def _soft_max(O, beta):
    m = O.max()
    return m + np.log(np.sum(np.exp(beta * (O - m)))) / beta

def _objective(h, beta):
    O = _overlap_matrix(h)
    return _soft_max(O, beta)

# ---------- reparameterization to keep h in [0,1] & sum = N/2 ----------
def _project(z):
    """Map unconstrained z to h in [0,1] with sum exactly N/2.
    Use a soft clip then a bisection on a scale to hit the sum."""
    # base: logistic squashing to (0,1)
    t = 1.0 / (1.0 + np.exp(-z))
    # we need sum(t) = N/2. t in (0,1). Use affine: h = a*t + b, choose
    # a,b so that min>=0, max<=1, and sum = N/2. Simpler: h = t + c then
    # clip, but clip breaks smoothness. Use: h = clip(t + c, 0, 1) and
    # solve c by bisection so sum = N/2.
    lo, hi = -1.0, 1.0
    for _ in range(60):
        c = 0.5 * (lo + hi)
        h = np.clip(t + c, 0.0, 1.0)
        s = h.sum()
        if s > N / 2:
            hi = c
        else:
            lo = c
    h = np.clip(t + 0.5 * (lo + hi), 0.0, 1.0)
    # exact fix: distribute residual to keep in [0,1]
    r = N / 2 - h.sum()
    if r > 0:
        room = 1.0 - h
        idx = np.argsort(-room)
        acc = 0.0
        for j in idx:
            give = min(room[j], r - acc)
            h[j] += give
            acc += give
            if acc >= r - 1e-12:
                break
    elif r < 0:
        room = h
        idx = np.argsort(-room)
        acc = 0.0
        for j in idx:
            give = min(room[j], -r - acc)
            h[j] -= give
            acc += give
            if acc >= -r - 1e-12:
                break
    return h

def _grad_soft_max(z, beta):
    """Gradient of soft_max(O(z)) w.r.t. z, via finite-diff-free chain.
    We compute dO/dh analytically, then dh/dz from the projection (approx).
    For simplicity we use a projected gradient on z with a step, relying on
    the projection. Here we just return dO/dh; caller applies to z via
    a finite-difference-free mapping by perturbing z and reprojecting."""
    h = _project(z)
    O = _overlap_matrix(h)
    m = O.max()
    w = np.exp(beta * (O - m))
    w = w / w.sum()
    # dO/dh: O(s) = sum_i h_i - sum_i h_i h_{i-s}
    # dO(s)/dh_k = 1 - sum_s' ... let's derive:
    # O(s) = sum_i h_i (1 - h_{i-s})
    # dO(s)/dh_k: term i=k: (1 - h_{k-s}); term i where i-s = k i.e. i = k+s:
    #   h_i * (-1) => -h_{k+s}
    # So dO(s)/dh_k = (1 - h_{k-s}) - h_{k+s}
    ks = np.arange(N)
    dOdH = (1.0 - h[(-ks) % N]) - h[(ks) % N]   # shape (N,) = dO(s)/dh_k for the s=k? 
    # Actually dO(s)/dh_k depends on s and k. We need full matrix dO(s)/dh_k =
    # (1 - h_{k-s}) - h_{k+s}. Let's build it as a circulant.
    # For each shift s, dO(s)/dh_k = (1 - h_{k-s}) - h_{k+s}.
    # = 1 - h_{k-s} - h_{k+s}.
    # We'll compute the gradient vector g_k = sum_s w_s * dO(s)/dh_k.
    # g_k = sum_s w_s (1 - h_{k-s} - h_{k+s})
    #     = sum_s w_s - sum_s w_s h_{k-s} - sum_s w_s h_{k+s}
    # = 1 - (w * h)[k]_circ - (w * h)[k]_circ  where (w*h) is circular conv.
    # sum_s w_s = 1.
    W = np.fft.fft(w)
    Hh = np.fft.fft(h)
    conv = np.fft.ifft(W * Hh).real   # conv[k] = sum_s w_s h_{k-s}
    g = 1.0 - 2.0 * conv
    return g, h

# ---------- main ----------
def construct():
    deadline = time.time() + 120.0 - 15.0   # 15s safety margin
    rng = np.random.default_rng(0)

    best_h = np.full(N, 0.5)
    best_score = float(_overlap_matrix(best_h).max())

    def consider(h):
        nonlocal best_h, best_score
        h = np.clip(h, 0.0, 1.0)
        # enforce sum
        r = N / 2 - h.sum()
        if abs(r) > 1e-9:
            # distribute
            if r > 0:
                room = 1.0 - h
            else:
                room = h
            idx = np.argsort(-room)
            acc = 0.0
            for j in idx:
                give = min(room[j], abs(r) - acc)
                h[j] += np.sign(r) * give
                acc += give
                if acc >= abs(r) - 1e-12:
                    break
        sc = float(_overlap_matrix(h).max())
        if sc < best_score - 1e-12:
            best_score = sc
            best_h = h.copy()
        return sc

    consider(best_h)

    # Multi-start projected gradient on z (unconstrained), soft-max objective.
    n_starts = 40
    for start in range(n_starts):
        if time.time() > deadline:
            break
        if start == 0:
            z = np.zeros(N)
        else:
            z = rng.normal(0.0, 0.5, N)
        beta = 10.0
        lr = 0.5
        for it in range(400):
            if time.time() > deadline:
                break
            g, h = _grad_soft_max(z, beta)
            # projected gradient step on z (approx: move z opposite to g,
            # then reproject; the projection handles constraints)
            z = z - lr * g
            # occasional anneal
            if it % 50 == 0 and it > 0:
                beta *= 1.5
                lr *= 0.7
            h = _project(z)
            consider(h)
        # final exact polish on the best z region
        h = _project(z)
        consider(h)

    # Final high-precision polish: local coordinate descent on h directly
    h = best_h.copy()
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old = h[i]
            # try small perturbations
            for cand in (old + 0.01, old - 0.01, old + 0.001, old - 0.001):
                if cand < 0 or cand > 1:
                    continue
                # keep sum: adjust a compensating index
                h[i] = cand
                r = N / 2 - h.sum()
                # find j != i with room
                j = (i + 1) % N
                if j == i:
                    j = (i + 2) % N
                room = (1.0 - h[j]) if r > 0 else h[j]
                if room < abs(r) - 1e-12:
                    h[i] = old
                    continue
                h[j] += np.sign(r) * r
                sc = float(_overlap_matrix(h).max())
                if sc < best_score - 1e-12:
                    best_score = sc
                    best_h = h.copy()
                    improved = True
                    break
                else:
                    h[i] = old
                    h[j] -= np.sign(r) * r
        if not improved:
            break
    h = best_h
    consider(h)
    return best_h
