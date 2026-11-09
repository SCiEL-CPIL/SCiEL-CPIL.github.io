"""Flattest +/-1 polynomial with 24 coefficients (min sup-norm / sqrt(24)).

Combined strategy:
  * Fast sup-norm estimator via FFT dense sampling + parabolic + Newton
    refinement of the top distinct peaks (replaces slow golden-section loop).
  * Smooth Lp surrogate (p=32) to cheaply guide candidate flips.
  * Diverse seeds (Golay-RS + variants, Thue-Morse, blocks, structured, random).
  * Multi-pass local search: single-bit, then pair, then triple flips.
  * Simulated-annealing escape pass to leave local minima.
  * Hard wall-clock deadline with safety margin.
"""

import math
import random
import time

import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_FFT_SIZE = 16384
_K = np.arange(_N)
_TWO_PI = 2.0 * math.pi


# --- Fast sup-norm scoring -------------------------------------------------

def _eval_at_theta(c, theta):
    z = np.exp(1j * theta * _K)
    return abs(np.dot(c, z))


def _find_peak(c):
    """Accurate max_{|z|=1} |P(z)| via FFT grid + parabolic + Newton refine."""
    c_pad = np.zeros(_FFT_SIZE, dtype=np.complex128)
    c_pad[:_N] = c
    F = np.fft.fft(c_pad)
    absF = np.abs(F)
    top = np.argsort(absF)[::-1][:12]

    peaks = []
    for idx in top:
        if all(abs(int(idx) - int(p)) > _FFT_SIZE // 200 for p in peaks):
            peaks.append(int(idx))
        if len(peaks) >= 8:
            break

    best = 0.0
    for idx in peaks:
        dth = _TWO_PI / _FFT_SIZE
        im1 = (idx - 1) % _FFT_SIZE
        ip1 = (idx + 1) % _FFT_SIZE
        f_im1 = _eval_at_theta(c, _TWO_PI * im1 / _FFT_SIZE)
        f_i = _eval_at_theta(c, _TWO_PI * idx / _FFT_SIZE)
        f_ip1 = _eval_at_theta(c, _TWO_PI * ip1 / _FFT_SIZE)

        denom = f_im1 - 2.0 * f_i + f_ip1
        if abs(denom) > 1e-15:
            offset = 0.5 * (f_im1 - f_ip1) / denom
        else:
            offset = 0.0
        offset = max(-0.5, min(0.5, offset))
        theta = _TWO_PI * (idx + offset) / _FFT_SIZE

        # Newton refine on |P|^2
        for _ in range(10):
            z = np.exp(1j * theta)
            P_val = np.dot(c, z ** _K)
            Pp = np.dot(c * _K, 1j * z ** _K)
            grad = 2.0 * np.real(np.conj(P_val) * Pp)
            if abs(grad) < 1e-14:
                break
            h = 1e-5
            zh = np.exp(1j * (theta + h))
            P_h = np.dot(c, zh ** _K)
            Pp_h = np.dot(c * _K, 1j * zh ** _K)
            g_h = 2.0 * np.real(np.conj(P_h) * Pp_h)
            zm = np.exp(1j * (theta - h))
            P_m = np.dot(c, zm ** _K)
            Pp_m = np.dot(c * _K, 1j * zm ** _K)
            g_m = 2.0 * np.real(np.conj(P_m) * Pp_m)
            hess = (g_h - g_m) / (2.0 * h)
            if abs(hess) < 1e-15:
                break
            step = -grad / hess
            if abs(step) > 0.1:
                step = math.copysign(0.1, step)
            theta = (theta + step) % _TWO_PI

        val = _eval_at_theta(c, theta)
        if val > best:
            best = val
    return best


def _score(c):
    """sup-norm / sqrt(N)."""
    return _find_peak(c) / _SQRTN


def _lp(c, p=32):
    """Smooth Lp surrogate of the sup-norm (cheap guide for flips)."""
    c_pad = np.zeros(_FFT_SIZE, dtype=np.complex128)
    c_pad[:_N] = c
    F = np.fft.fft(c_pad)
    return float(np.sum(np.abs(F) ** p) ** (1.0 / p))


# --- Candidate generation --------------------------------------------------

def _golay_rs(n):
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]


def _thue_morse(n):
    out = []
    for k in range(n):
        x = k
        par = 0
        while x:
            par ^= (x & 1)
            x >>= 1
        out.append(1 if par == 0 else -1)
    return out


def _block_pattern(period, signs):
    return [signs[k % period] for k in range(_N)]


def _random_cand(rng):
    return [1 if rng.random() < 0.5 else -1 for _ in range(_N)]


def _initial_candidates(rng):
    cands = []
    g = _golay_rs(_N)
    gr = g[::-1]
    cands.append(g)
    cands.append(gr)
    cands.append([-c for c in g])
    cands.append([-c for c in gr])
    # cyclic shifts of golay and its reverse
    for s in range(1, _N):
        cands.append([g[(i + s) % _N] for i in range(_N)])
        cands.append([gr[(i + s) % _N] for i in range(_N)])
    # all single-bit flips of golay and reverse
    for i in range(_N):
        c = list(g); c[i] = -c[i]; cands.append(c)
        c = list(gr); c[i] = -c[i]; cands.append(c)
    # all double-bit flips of golay
    for i in range(_N):
        for j in range(i + 1, _N):
            c = list(g); c[i] = -c[i]; c[j] = -c[j]; cands.append(c)
    # thue-morse + variants
    tm = _thue_morse(_N)
    cands.append(tm)
    cands.append(tm[::-1])
    cands.append([-c for c in tm])
    # block patterns
    cands.append(_block_pattern(2, [1, -1]))
    cands.append(_block_pattern(3, [1, 1, -1]))
    cands.append(_block_pattern(4, [1, -1, -1, 1]))
    cands.append(_block_pattern(6, [1, 1, -1, -1, 1, -1]))
    cands.append(_block_pattern(8, [1, -1, 1, 1, -1, -1, 1, -1]))
    cands.append(_block_pattern(12, [1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1]))
    # random
    for _ in range(40):
        cands.append(_random_cand(rng))
    return cands


# --- Local search ----------------------------------------------------------

def _local_search(c, rng, deadline):
    """Greedy multi-pass hill-climb: single, then pair, then triple flips."""
    c = list(c)
    cur = _score(c)

    # single-bit flips
    improved = True
    passes = 0
    while improved and passes < 40 and time.time() < deadline:
        improved = False
        passes += 1
        best_c, best_s = None, cur
        for i in range(_N):
            c[i] = -c[i]
            s = _score(c)
            c[i] = -c[i]
            if s < best_s - 1e-12:
                best_s = s
                best_c = list(c); best_c[i] = -best_c[i]
        if best_c is not None:
            c, cur = best_c, best_s
            improved = True

    # pair flips
    improved = True
    passes = 0
    while improved and passes < 12 and time.time() < deadline:
        improved = False
        passes += 1
        best_c, best_s = None, cur
        for i in range(_N):
            for j in range(i + 1, _N):
                c[i] = -c[i]; c[j] = -c[j]
                s = _score(c)
                c[i] = -c[i]; c[j] = -c[j]
                if s < best_s - 1e-12:
                    best_s = s
                    best_c = list(c); best_c[i] = -best_c[i]; best_c[j] = -best_c[j]
        if best_c is not None:
            c, cur = best_c, best_s
            improved = True

    # triple flips
    improved = True
    passes = 0
    while improved and passes < 5 and time.time() < deadline:
        improved = False
        passes += 1
        best_c, best_s = None, cur
        for i in range(_N):
            for j in range(i + 1, _N):
                for k in range(j + 1, _N):
                    c[i] = -c[i]; c[j] = -c[j]; c[k] = -c[k]
                    s = _score(c)
                    c[i] = -c[i]; c[j] = -c[j]; c[k] = -c[k]
                    if s < best_s - 1e-12:
                        best_s = s
                        best_c = list(c)
                        best_c[i] = -best_c[i]; best_c[j] = -best_c[j]; best_c[k] = -best_c[k]
        if best_c is not None:
            c, cur = best_c, best_s
            improved = True

    return c, cur


def _anneal(c, rng, deadline):
    """Simulated-annealing escape from local minima."""
    c = list(c)
    cur = _score(c)
    best = list(c)
    best_s = cur
    T = 0.004
    T_min = 0.0003
    cool = 0.9975
    it = 0
    while time.time() < deadline and T > T_min:
        it += 1
        i = rng.randrange(_N)
        c[i] = -c[i]
        s = _score(c)
        if s <= cur or rng.random() < math.exp(-(s - cur) / T):
            cur = s
            if s < best_s - 1e-12:
                best_s = s
                best = list(c)
        else:
            c[i] = -c[i]
        T *= cool
    return best, best_s


def construct():
    t_start = time.time()
    # Safety margin well under the 120s evaluator timeout / 150s kill.
    deadline = t_start + 108.0

    rng = random.Random(12345)

    # Phase 1: score initial candidates with the fast estimator, keep best few.
    cands = _initial_candidates(rng)
    scored = []
    for c in cands:
        if time.time() > deadline - 25.0:
            break
        scored.append((_score(c), list(c)))
    scored.sort(key=lambda x: x[0])

    if not scored:
        return _golay_rs(_N)

    best_c, best_s = scored[0][1], scored[0][0]

    # Phase 2: local search on the best, plus a few promising random restarts.
    while time.time() < deadline - 3.0:
        c, s = _local_search(best_c, rng, deadline - 3.0)
        if s < best_s - 1e-12:
            best_s, best_c = s, c
        # random restart + local search
        rc = _random_cand(rng)
        rc, rs = _local_search(rc, rng, deadline - 3.0)
        if rs < best_s - 1e-12:
            best_s, best_c = rs, rc
        # simulated-annealing escape on the current best
        ac, as_ = _anneal(best_c, rng, deadline - 3.0)
        if as_ < best_s - 1e-12:
            best_s, best_c = as_, ac

    # Final polish: one more local-search pass on the global best.
    if time.time() < deadline:
        c, s = _local_search(best_c, rng, deadline)
        if s < best_s - 1e-12:
            best_s, best_c = s, c

    return best_c
