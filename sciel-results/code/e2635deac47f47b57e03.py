"""Antireflection coating design: multi-start coordinate descent over sequence + thicknesses."""
import time
import random
import numpy as np

_START = time.time()
_DEADLINE = 140.0  # safety margin well under the 150s wall-clock kill

# 401 equally spaced wavelengths over [400, 800]
_WAVES = np.linspace(400.0, 800.0, 401)


def _reflectance_pct(materials, thicknesses, ambient_index, substrate_index):
    """Mean normal-incidence reflectance (%) over the band via characteristic matrices."""
    nL = materials["L"]
    nH = materials["H"]
    n0 = ambient_index
    ns = substrate_index
    total = 0.0
    for lam in _WAVES:
        k = 2.0 * np.pi / lam
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for m, d in zip(materials, thicknesses):
            n = nL if m == "L" else nH
            delta = k * n * d
            c = np.cos(delta)
            s = np.sin(delta)
            M = M @ np.array([[c, 1j * s], [1j * n * s, n * c]], dtype=complex)
        Y = M[1, 1] / M[0, 1]
        r = (n0 - Y) / (n0 + Y)
        total += (r * np.conj(r)).real
    return 100.0 * total / len(_WAVES)


def _valid(materials, thicknesses, problem):
    if len(materials) > problem["max_layers"]:
        return False
    if len(materials) != len(thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"] + 1e-9:
        return False
    for d in thicknesses:
        if d < -1e-9 or d > problem["max_layer_nm"] + 1e-9:
            return False
    return True


def _clipped(thicknesses, problem):
    lo, hi = 0.0, problem["max_layer_nm"]
    return np.clip(np.asarray(thicknesses, dtype=float), lo, hi)


def _rand_seq(nL, nH, max_layers):
    n = random.randint(nL, nH)
    return [random.choice("LH") for _ in range(n)]


def _rand_thick(n, max_layer_nm):
    return [random.random() * max_layer_nm for _ in range(n)]


def _cd_seq(materials, thicknesses, problem, best_val, max_steps, nL, nH):
    """Coordinate descent over the discrete layer sequence; returns (materials, thicknesses, best)."""
    nL = nL
    nH = nH
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    n = len(materials)
    seq = list(materials)
    th = list(thicknesses)
    best = best_val
    step = 0
    while step < max_steps and time.time() < _DEADLINE:
        step += 1
        op = random.random()
        if op < 0.45 and n < max_layers:
            i = random.randint(0, n)
            cand = seq[:i] + [random.choice("LH")] + seq[i:]
            ct = th[:i] + [random.random() * max_layer_nm] + th[i:]
            v = _reflectance_pct(cand, ct, problem["ambient_index"], problem["substrate_index"])
            if v < best:
                best = v
                seq, th = cand, ct
        elif op < 0.70 and n > 1:
            i = random.randrange(n)
            cand = seq[:i] + seq[i + 1:]
            ct = th[:i] + th[i + 1:]
            v = _reflectance_pct(cand, ct, problem["ambient_index"], problem["substrate_index"])
            if v < best:
                best = v
                seq, th = cand, ct
        else:
            i = random.randrange(n)
            new = "H" if seq[i] == "L" else "L"
            cand = seq[:i] + [new] + seq[i:]
            v = _reflectance_pct(cand, th, problem["ambient_index"], problem["substrate_index"])
            if v < best:
                best = v
                seq, th = cand, th
    return seq, th, best


def _cd_thick(materials, thicknesses, problem, best_val, max_steps, step0, min_step):
    """Coordinate descent over continuous thicknesses; returns (materials, thicknesses, best)."""
    max_layer_nm = problem["max_layer_nm"]
    n = len(materials)
    th = np.array(thicknesses, dtype=float)
    best = best_val
    step = step0
    it = 0
    while it < max_steps and time.time() < _DEADLINE:
        it += 1
        for i in range(n):
            if time.time() > _DEADLINE:
                break
            base = th[i]
            for cand in (base + step, base - step):
                if cand < -1e-9 or cand > max_layer_nm + 1e-9:
                    continue
                th[i] = cand
                v = _reflectance_pct(materials, th, problem["ambient_index"], problem["substrate_index"])
                if v < best:
                    best = v
                else:
                    th[i] = base
                    break
            if time.time() > _DEADLINE:
                break
        step *= 0.85
        if step < min_step:
            break
    return materials, th.tolist(), best


def design(problem):
    """Multi-start search over layer sequence and thicknesses to minimise mean reflectance."""
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    ambient = problem["ambient_index"]
    substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])

    rng = random.Random(12345)
    random.seed(12345)

    # Seed 0: analytical quarter-wave quarter-stack centred in the band
    qL = centre / (4.0 * nL)
    qH = centre / (4.0 * nH)
    best_seq = ["L", "H", "L"]
    best_th = [qL, qH, qL]
    best_val = _reflectance_pct(best_seq, best_th, ambient, substrate)

    # Seed 1: single L quarter-wave (reference)
    s1 = ["L"]
    t1 = [qL]
    v1 = _reflectance_pct(s1, t1, ambient, substrate)
    if v1 < best_val:
        best_val, best_seq, best_th = v1, s1, t1

    # Seed 2: random multi-layer
    for _ in range(3):
        if time.time() > _DEADLINE:
            break
        seq = _rand_seq(3, min(9, max_layers), max_layers)
        th = _rand_thick(len(seq), max_layer_nm)
        if sum(th) > max_total:
            scale = max_total / sum(th)
            th = [x * scale for x in th]
        v = _reflectance_pct(seq, th, ambient, substrate)
        if v < best_val:
            best_val, best_seq, best_th = v, seq, th

    # Coordinate descent on the current best
    best_seq, best_th, best_val = _cd_seq(best_seq, best_th, problem, best_val, 120, 3, min(9, max_layers))
    best_seq, best_th, best_val = _cd_thick(best_seq, best_th, problem, best_val, 60, 40.0, 0.5)

    # Multi-start refinement
    start = time.time()
    while time.time() < _DEADLINE - 20.0:
        # per-start budget scales with remaining time
        rem = _DEADLINE - 20.0 - time.time()
        seq = _rand_seq(3, min(10, max_layers), max_layers)
        th = _rand_thick(len(seq), max_layer_nm)
        if sum(th) > max_total:
            scale = max_total / sum(th)
            th = [x * scale for x in th]
        v = _reflectance_pct(seq, th, ambient, substrate)
        seq, th, v = _cd_seq(seq, th, problem, v, 60, 3, min(10, max_layers))
        seq, th, v = _cd_thick(seq, th, problem, v, 40, 40.0, 0.5)
        if v < best_val:
            best_val, best_seq, best_th = v, seq, th
        if time.time() - start > 25.0:
            # avoid one start dominating; keep iterating
            start = time.time()

    # Final polish on the global best
    best_seq, best_th, best_val = _cd_thick(best_seq, best_th, problem, best_val, 30, 10.0, 0.2)
    best_seq, best_th, best_val = _cd_seq(best_seq, best_th, problem, best_val, 40, 3, min(10, max_layers))
    best_seq, best_th, best_val = _cd_thick(best_seq, best_th, problem, best_val, 30, 10.0, 0.2)

    # Ensure validity (clip and trim)
    best_th = [min(max(d, 0.0), max_layer_nm) for d in best_th]
    while len(best_th) > 1 and sum(best_th) > max_total + 1e-9:
        # drop the thinnest layer to satisfy total
        idx = min(range(len(best_th)), key=lambda i: best_th[i])
        best_th.pop(idx)
        best_seq.pop(idx)
    if sum(best_th) > max_total + 1e-9:
        scale = max_total / sum(best_th)
        best_th = [d * scale for d in best_th]

    return best_seq, [round(d, 6) for d in best_th]
