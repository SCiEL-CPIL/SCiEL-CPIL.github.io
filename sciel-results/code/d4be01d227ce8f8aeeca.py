"""Rewritten program: minimize the scale-invariant autocorrelation objective via
projected subgradient (convex minimax) on the simplex sum(h)=1, with FFT autoconv."""
import time
import numpy as np

N = 400


def _eval(h):
    """Return the task objective for a valid h, else None."""
    if h is None:
        return None
    h = np.asarray(h, dtype=float)
    if h.size != N or not np.all(np.isfinite(h)):
        return None
    s = float(np.sum(h))
    if s == 0.0:
        return None
    c = np.fft.fftconvolve(h, h, mode="full")
    return abs(2.0 * N * float(np.max(c)) / (s * s))


def construct():
    t0 = time.time()
    DEADLINE = 120.0
    MARGIN = 8.0
    limit = t0 + DEADLINE - MARGIN

    best_h = np.ones(N, dtype=float)
    best_val = 2.0
    # The value is scale-invariant, so normalizing to sum=1 is lossless.
    rng = np.random.default_rng(12345)

    def project(h):
        # project onto sum(h)=1 (keeps sum exactly 1, preserves finiteness)
        return h - (h - 1.0 / N).mean()

    def run():
        nonlocal best_h, best_val
        h = project(np.ones(N, dtype=float) + rng.standard_normal(N) * 0.02)
        alpha = 0.02
        iters = 4000
        for it in range(iters):
            if time.time() > limit:
                break
            c = np.fft.fftconvolve(h, h, mode="full")
            m = float(c.max())
            # subgradient of max(c) w.r.t. h: 2*(h*h + flip(h)*h)
            g = 2.0 * (h * h + h[::-1] * h)
            g += 2.0 * (c == c.max())
            g -= g.mean()  # keep step on the sum=1 tangent
            h = project(h - alpha * g)
            val = abs(2.0 * N * m)  # sum(h)=1 -> denominator 1
            if val < best_val:
                best_val = val
                best_h = h.copy()
            # adaptive step: shrink if value grew, else grow slightly
            if it > 50:
                if val > best_val * 1.05:
                    alpha *= 0.9
                else:
                    alpha = min(alpha * 1.01, 0.05)

    for _ in range(8):
        if time.time() > limit:
            break
        run()

    # Final rescale (value unchanged; guarantees finite, nonzero sum).
    out = best_h.astype(float)
    s = float(np.sum(out))
    if s == 0.0 or not np.all(np.isfinite(out)):
        out = np.ones(N, dtype=float)
    else:
        out = out / s
    return out


if __name__ == "__main__":
    print(_eval(construct()))
