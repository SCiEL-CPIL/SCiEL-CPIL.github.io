import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s kill

    N = 24
    sqrtN = np.sqrt(float(N))

    # Known good candidates (Golay / Odlyzko / RS families)
    candidates = [
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
        [1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
        [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1],
        [1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1],
    ]

    # Spectral synthesis: build flat-ish target spectrum, IFFT, threshold to +/-1.
    # This creates globally coherent (not random) starting points.
    def synth_candidates():
        out = []
        M = 256
        k = np.arange(M)
        # target: low-ripple flat spectrum (cosine taper near edges to reduce Gibbs)
        for taper in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            t = np.linspace(-1, 1, M)
            # cosine taper factor (1 in center, ->0 at edges)
            if taper > 0:
                w = np.cos(np.pi * t / 2.0) ** taper
            else:
                w = np.ones(M)
            # flat target magnitude with slight low-freq emphasis
            mag = w * (1.0 + 0.2 * np.cos(2 * np.pi * t / M))
            # random phases (deterministic per taper)
            rng = np.random.RandomState(1000 + int(taper * 10))
            phase = rng.uniform(0, 2 * np.pi, M)
            spec = mag * np.exp(1j * phase)
            # ensure Hermitian symmetry so IFFT is real
            spec = spec + np.conj(spec[::-1])
            spec[0] = spec[0].real
            if M % 2 == 0:
                spec[M // 2] = spec[M // 2].real
            sig = np.fft.ifft(spec).real
            # take first N, threshold
            x = sig[:N]
            c = np.where(x >= 0, 1.0, -1.0)
            out.append(c)
        # also a few deterministic phase patterns (no random)
        for ph in [0.0, np.pi / 4, np.pi / 2, np.pi / 3]:
            spec = np.exp(1j * ph * np.arange(M))
            spec = spec + np.conj(spec[::-1])
            spec[0] = spec[0].real
            if M % 2 == 0:
                spec[M // 2] = spec[M // 2].real
            sig = np.fft.ifft(spec).real
            c = np.where(sig[:N] >= 0, 1.0, -1.0)
            out.append(c)
        return out

    candidates += synth_candidates()

    # Evaluation machinery
    M = 2 ** 15  # 32768 points for dense screening
    k = np.arange(M)
    n_idx = np.arange(N)
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    D_cols = D  # (M, N)

    def eval_dft(coeffs):
        F = D @ coeffs
        return np.max(np.abs(F)) / sqrtN

    def golden_refine(coeffs, peak_indices, n_refine=60):
        best = 0.0
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        for idx in peak_indices:
            tc = 2.0 * np.pi * idx / M
            window = 4.0 / M
            a = tc - window
            b = tc + window
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)

            def ev(th):
                z = np.exp(1j * th)
                v = 0.0
                for ck in reversed(coeffs):
                    v = v * z + ck
                return abs(v)

            fc = ev(c_pt)
            fd = ev(d_pt)
            for _ in range(n_refine):
                if fc < fd:
                    b = d_pt
                    d_pt = c_pt
                    fd = fc
                    c_pt = b - gr * (b - a)
                    fc = ev(c_pt)
                else:
                    a = c_pt
                    c_pt = d_pt
                    fc = fd
                    d_pt = a + gr * (b - a)
                    fd = ev(d_pt)
            lm = max(fc, fd)
            if lm > best:
                best = lm
        return best / sqrtN

    def eval_accurate(coeffs):
        F = D @ coeffs
        absF = np.abs(F)
        top = np.argsort(absF)[-12:]
        top = np.unique(top)
        return golden_refine(coeffs, top, n_refine=60)

    # Screen all candidates with DFT, keep best few, refine
    best_score = float('inf')
    best_c = None
    ranked = []
    for c in candidates:
        c_arr = np.array(c, dtype=float)
        s = eval_dft(c_arr)
        ranked.append((s, c_arr))
    ranked.sort(key=lambda x: x[0])
    for s, c_arr in ranked[:8]:
        s_acc = eval_accurate(c_arr)
        if s_acc < best_score:
            best_score = s_acc
            best_c = c_arr.copy()

    current_c = best_c.copy()
    current_f = D @ current_c
    current_max = np.max(np.abs(current_f)) / sqrtN

    np.random.seed(12345)

    temperature = 0.06
    cooling = 0.999
    min_temp = 0.003
    stuck = 0

    while time.time() < deadline:
        if temperature < min_temp:
            temperature = 0.06
            n_flips = np.random.randint(6, 11)
            idx = np.random.choice(N, n_flips, replace=False)
            current_c[idx] *= -1.0
            current_f = D @ current_c
            current_max = np.max(np.abs(current_f)) / sqrtN
            stuck = 0

        # 1-flip
        best1 = -1
        best1v = current_max
        for n in range(N):
            fn = current_f - 2.0 * current_c[n] * D_cols[:, n]
            mv = np.max(np.abs(fn)) / sqrtN
            if mv < best1v:
                best1v = mv
                best1 = n
        if best1 != -1:
            current_c[best1] *= -1.0
            current_f = D @ current_c
            current_max = best1v
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
            temperature *= cooling
            stuck = 0
            continue

        # 2-flip
        best2 = (-1, -1)
        best2v = current_max
        for n1 in range(N):
            for n2 in range(n1 + 1, N):
                fn = current_f - 2.0 * (current_c[n1] * D_cols[:, n1] +
                                        current_c[n2] * D_cols[:, n2])
                mv = np.max(np.abs(fn)) / sqrtN
                if mv < best2v:
                    best2v = mv
                    best2 = (n1, n2)
        if best2 != (-1, -1):
            n1, n2 = best2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = D @ current_c
            current_max = best2v
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
            temperature *= cooling
            stuck = 0
            continue

        # 3-flip (sampled)
        if stuck < 6:
            best3 = (-1, -1, -1)
            best3v = current_max
            for _ in range(300):
                i3 = np.random.choice(N, 3, replace=False)
                fn = current_f - 2.0 * (current_c[i3[0]] * D_cols[:, i3[0]] +
                                        current_c[i3[1]] * D_cols[:, i3[1]] +
                                        current_c[i3[2]] * D_cols[:, i3[2]])
                mv = np.max(np.abs(fn)) / sqrtN
                if mv < best3v:
                    best3v = mv
                    best3 = (i3[0], i3[1], i3[2])
            if best3 != (-1, -1, -1):
                n1, n2, n3 = best3
                current_c[n1] *= -1.0
                current_c[n2] *= -1.0
                current_c[n3] *= -1.0
                current_f = D @ current_c
                current_max = best3v
                if current_max < best_score:
                    best_score = current_max
                    best_c = current_c.copy()
                temperature *= cooling
                stuck = 0
                continue

        # SA move
        n_flips = np.random.randint(2, 5)
        idx = np.random.choice(N, n_flips, replace=False)
        old_c = current_c.copy()
        old_max = current_max
        current_c[idx] *= -1.0
        current_f = D @ current_c
        new_max = np.max(np.abs(current_f)) / sqrtN
        delta = new_max - old_max
        if delta < 0 or np.random.rand() < np.exp(-delta / max(temperature, 1e-10)):
            current_max = new_max
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
        else:
            current_c = old_c
            current_f = D @ old_c
            current_max = old_max
        temperature *= cooling
        stuck += 1

    # Final polish: a few rounds of 1-flip from best_c, then accurate eval
    c_polish = best_c.copy()
    f_polish = D @ c_polish
    max_polish = np.max(np.abs(f_polish)) / sqrtN
    for _ in range(40):
        improved = False
        for n in range(N):
            fn = f_polish - 2.0 * c_polish[n] * D_cols[:, n]
            mv = np.max(np.abs(fn)) / sqrtN
            if mv < max_polish - 1e-12:
                c_polish[n] *= -1.0
                f_polish = D @ c_polish
                max_polish = mv
                improved = True
        if not improved:
            break

    final_score = eval_accurate(c_polish)
    if final_score < best_score:
        best_c = c_polish
        best_score = final_score

    return [int(x) for x in best_c]
