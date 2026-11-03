import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N for fast FFT


def _autocorr_fast(h):
    # linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
    # Use real FFT trick: autocorr of real signal = IFFT(|FFT|^2)
    F = np.fft.rfft(h, n=_NFFT)
    ac = np.fft.irfft(F * np.conj(F), n=_NFFT)[:N]
    return ac


def _autocorr(h):
    return _autocorr_fast(h)


def _hard_score(h):
    # the actual metric: peak autoconvolution concentration (lower is better)
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score(h, beta):
    # smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    # Clip z to avoid overflow in exp
    z_clipped = np.clip(z, -50, 50)
    return m + float(np.log(np.sum(np.exp(z_clipped)))) / beta


class _FastAutocorr:
    """Fast autocorrelation with incremental updates for block flips."""
    
    def __init__(self, h):
        self.h = h.copy()
        self.F = np.fft.rfft(self.h, n=_NFFT)
        self._update_ac()
    
    def _update_ac(self):
        self.ac = np.fft.irfft(self.F * np.conj(self.F), n=_NFFT)[:N]
    
    def flip_block(self, start, size):
        """Flip a block of signs and update autocorrelation."""
        sl = slice(start, start + size)
        self.h[sl] *= -1.0
        # Update FFT incrementally: F_new = F_old + delta_F where delta_F = FFT of (h_new - h_old)
        delta_h = np.zeros(_NFFT)
        delta_h[sl] = self.h[sl] * 2.0  # h_new - h_old = -h_old - h_old = -2*h_old, but h is already flipped
        # Actually: h_new = -h_old, so delta = h_new - h_old = -2*h_old
        # But we already flipped, so delta_h[sl] = -2 * (-self.h[sl]) = 2 * self.h[sl] ... wait
        # Let me redo: before flip, h_old. After flip, h_new = -h_old.
        # delta = h_new - h_old = -2 * h_old
        # But self.h is now h_new, so h_old = -self.h[sl]
        # delta = -2 * (-self.h[sl]) = 2 * self.h[sl]
        # So delta_h[sl] = 2 * self.h[sl] is correct
        delta_F = np.fft.rfft(delta_h, n=_NFFT)
        self.F = self.F + delta_F
        self._update_ac()
    
    def unflip_block(self, start, size):
        """Undo a block flip."""
        self.flip_block(start, size)
    
    def flip_bit(self, i):
        """Flip a single bit."""
        self.flip_block(i, 1)
    
    def get_score(self):
        dc = self.ac[0]
        if dc <= 1e-30:
            return 1e9
        return float(np.max(np.abs(self.ac)) / dc)
    
    def get_smooth_score(self, beta):
        dc = self.ac[0]
        if dc <= 1e-30:
            return 1e9
        y = (self.ac[1:] / dc) ** 2
        m = float(np.max(y))
        z = beta * (y - m)
        z_clipped = np.clip(z, -50, 50)
        return m + float(np.log(np.sum(np.exp(z_clipped)))) / beta


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 115.0          # safety margin inside the 120s limit
        rng = np.random.default_rng(20240517)

        # ---------- diverse initializations ----------
        idx = np.arange(N)
        inits = [
            np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            # Add more diverse inits
            np.sign(np.cos(2 * np.pi * 3 * idx / N)),
            np.sign(np.cos(2 * np.pi * 5 * idx / N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_hard = best_hard
        cur_smooth = _smooth_score(h, 30.0)

        # ---------- simulated annealing with coordinated block sign-flips ----------
        T = 0.05  # Higher initial temperature for more exploration
        T_min = 1e-5
        decay = 0.9998  # Slower decay for longer exploration
        beta = 30.0
        sizes = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256])
        
        # Use FastAutocorr for efficient updates
        fa = _FastAutocorr(h)
        
        while time.time() < deadline - 8.0:
            if rng.random() < 0.55:
                # Block flip
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                fa.flip_block(start, size)
                new_hard = fa.get_score()
                new_smooth = fa.get_smooth_score(beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = fa.h.copy()
                else:
                    fa.unflip_block(start, size)
            elif rng.random() < 0.5:
                # Two-bit flip
                i1 = int(rng.integers(0, N))
                i2 = int(rng.integers(0, N))
                if i1 == i2:
                    i2 = (i2 + 1) % N
                fa.flip_bit(i1)
                fa.flip_bit(i2)
                new_hard = fa.get_score()
                new_smooth = fa.get_smooth_score(beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = fa.h.copy()
                else:
                    fa.flip_bit(i1)
                    fa.flip_bit(i2)
            else:
                # Single bit flip
                i = int(rng.integers(0, N))
                fa.flip_bit(i)
                new_hard = fa.get_score()
                new_smooth = fa.get_smooth_score(beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = fa.h.copy()
                else:
                    fa.flip_bit(i)
            T = max(T_min, T * decay)

        # ---------- greedy multi-scale local polish (optimized) ----------
        h = best_h.copy()
        sizes_polish = [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        fa = _FastAutocorr(h)
        for _pass in range(60):
            if time.time() > deadline - 3.0:
                break
            any_improve = False
            for size in sizes_polish:
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 3.0:
                        break
                    fa.flip_block(start, size)
                    new_hard = fa.get_score()
                    if new_hard < best_hard - 1e-10:
                        best_hard = new_hard
                        best_h = fa.h.copy()
                        any_improve = True
                    else:
                        fa.unflip_block(start, size)
                if time.time() > deadline - 3.0:
                    break
            if not any_improve:
                break
        # Final verification with exact score
        best_hard = _hard_score(best_h)

        # ---------- final single-bit polish (optimized) ----------
        h = best_h.copy()
        fa = _FastAutocorr(h)
        for _pass in range(200):
            if time.time() > deadline - 0.5:
                break
            any_improve = False
            for i in range(N):
                if time.time() > deadline - 0.5:
                    break
                fa.flip_bit(i)
                new_hard = fa.get_score()
                if new_hard < best_hard - 1e-10:
                    best_hard = new_hard
                    best_h = fa.h.copy()
                    any_improve = True
                else:
                    fa.flip_bit(i)
            if not any_improve:
                break
        # Final verification with exact score
        best_hard = _hard_score(best_h)

        # ---------- final two-bit polish ----------
        h = best_h.copy()
        fa = _FastAutocorr(h)
        for _pass in range(50):
            if time.time() > deadline - 0.1:
                break
            any_improve = False
            for i in range(N):
                if time.time() > deadline - 0.1:
                    break
                for j in range(i+1, min(i+20, N)):  # Local pairs only
                    fa.flip_bit(i)
                    fa.flip_bit(j)
                    new_hard = fa.get_score()
                    if new_hard < best_hard - 1e-10:
                        best_hard = new_hard
                        best_h = fa.h.copy()
                        any_improve = True
                        fa.flip_bit(i)  # Undo j flip to continue
                        break
                    else:
                        fa.flip_bit(j)
                        fa.flip_bit(i)
                if time.time() > deadline - 0.1:
                    break
            if not any_improve:
                break
        # Final verification with exact score
        best_hard = _hard_score(best_h)

        return best_h
    except Exception:
        return np.ones(N)
