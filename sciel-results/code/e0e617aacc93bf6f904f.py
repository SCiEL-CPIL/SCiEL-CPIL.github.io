import numpy as np
import time

N = 100
HALF = N / 2.0


def _time_remaining(deadline):
    return max(0.0, deadline - time.time())


def _compute_overlaps(h):
    """Compute all cyclic overlaps of h with 1-h.
    overlap[k] = sum_i h[i] * (1 - h[(i+k) % N])
    = N/2 - autocorr[k]  (for cyclic, with autocorr[0] = sum h^2)
    We want to minimize max over k=1..N-1 of overlap[k],
    equivalently maximize min over k=1..N-1 of autocorr[k].
    """
    H = np.fft.fft(h)
    ac = np.fft.ifft(np.abs(H) ** 2).real
    # ac[0] = sum h^2; ac[k] for k=1..N-1 is the cyclic autocorrelation
    overlaps = HALF - ac[1:]
    return overlaps


def _max_overlap_value(h):
    """Exact max overlap (for final scoring)."""
    ov = _compute_overlaps(h)
    return float(np.max(ov))


def _soft_min_ac(h, tau):
    """Soft approximation of min autocorr (excluding lag 0).
    Returns (value, gradient w.r.t. h).
    We maximize min ac, so we use soft-min of ac[1:].
    """
    H = np.fft.fft(h)
    ac = np.fft.ifft(np.abs(H) ** 2).real
    ac_vec = ac[1:]  # length N-1

    m = np.min(ac_vec)
    # Soft min: -tau * log(sum exp(-ac_vec / tau))
    shifted = ac_vec - m
    weights = np.exp(-shifted / tau)
    w_sum = np.sum(weights)
    p = weights / w_sum  # probability distribution over lags

    # soft_min = m - tau * log(w_sum)
    val = m - tau * np.log(w_sum)

    # Gradient of soft_min w.r.t. ac_vec:
    # d(soft_min)/dac_j = p_j  (since soft_min = min + tau*... actually let's derive)
    # soft_min = -tau log sum exp(-ac_j/tau)
    # d/dac_j = -tau * (1/sum) * exp(-ac_j/tau) * (-1/tau) = exp(-ac_j/tau)/sum = p_j
    # So grad_ac = p (vector)

    # Now grad of ac w.r.t. h:
    # ac[k] = sum_i h[i] h[i+k]  (cyclic)
    # dac[k]/dh[i] = h[i+k] + h[i-k]  (cyclic)
    # So grad_h[i] = sum_{k=1}^{N-1} p[k-1] * (h[i+k] + h[i-k])
    # This can be computed via convolution with a symmetric kernel.
    # kernel[k] = p[k-1] for k=1..N-1, kernel[N-k] = p[k-1] for k=1..N-1, kernel[0]=0
    kernel = np.zeros(N)
    kernel[1:] = p
    kernel[:0:-1] = p  # kernel[-1]=p[0], kernel[-2]=p[1], etc.
    # Actually: kernel[N-k] = p[k-1] for k=1..N-1
    # kernel[1] = p[0], kernel[2]=p[1], ..., kernel[N-1]=p[N-2]
    # kernel[N-1] = p[0]? No: kernel[N-k] for k=1 is kernel[N-1]=p[0]
    # Wait, I set kernel[1:]=p, so kernel[1]=p[0], kernel[2]=p[1],...,kernel[N-1]=p[N-2]
    # And kernel[:0:-1] goes from kernel[-1] down to kernel[1], i.e. kernel[N-1],kernel[N-2],...,kernel[1]
    # That would overwrite! Let me be careful.

    # Let me redo:
    kernel = np.zeros(N)
    # For lag k (1..N-1), weight p[k-1]
    # Contribution to grad_h[i]: p[k-1] * (h[i+k mod N] + h[i-k mod N])
    # This is a circular convolution of h with a symmetric kernel where
    # kernel[k] = p[k-1] for k=1..N-1, and kernel[N-k]=p[k-1] for k=1..N-1
    # So kernel is symmetric: kernel[k] = kernel[N-k]
    for k in range(1, N):
        kernel[k] = p[k - 1]
        kernel[N - k] = p[k - 1]

    # grad_h = circular convolution of h with kernel
    # Use FFT for speed
    grad_h = np.fft.ifft(np.fft.fft(h) * np.fft.fft(kernel)).real

    return val, grad_h


def _project_to_box(h, target_sum):
    """Project h onto {x : 0<=x<=1, sum x = target_sum}."""
    h = np.clip(h, 0.0, 1.0)
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    # Distribute correction to elements with room
    if diff > 0:
        # Need to add; elements with room are those < 1
        room = 1.0 - h
        total_room = np.sum(room)
        if total_room < diff - 1e-12:
            # Can't add enough; set all to 1 and handle
            h = np.clip(h + diff / N, 0, 1)
            return h
        h = h + diff * room / total_room
    else:
        # Need to subtract; elements with room are those > 0
        room = h
        total_room = np.sum(room)
        if total_room < -diff - 1e-12:
            h = np.clip(h + diff / N, 0, 1)
            return h
        h = h + diff * room / total_room

    h = np.clip(h, 0.0, 1.0)
    # Final correction on the element closest to 0.5
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        idx = np.argmin(np.abs(h - 0.5))
        h[idx] += diff
        h[idx] = np.clip(h[idx], 0.0, 1.0)
    return h


def _fourier_to_h(u):
    """Convert zero-mean Fourier coefficients to h in [0,1] with sum N/2.
    u is a vector of length N-1 (excluding DC), representing the non-DC
    part of the spectrum. We construct h = 0.5 + (1/N)*u_real where u_real
    is the inverse FFT of the full spectrum.
    Actually, let's work directly: h = 0.5 + v where v has zero mean.
    We parameterize v via its Fourier coefficients (non-DC).
    """
    # u has length N, u[0]=0 (DC), u[k] for k=1..N-1 are the coefficients
    # v = ifft(u).real, which has zero mean
    v = np.fft.ifft(u).real
    h = 0.5 + v
    return h


def _h_to_fourier(h):
    """Convert h to zero-mean Fourier coefficients (non-DC part)."""
    H = np.fft.fft(h)
    H[0] = 0.0  # Remove DC
    return H


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    best_h = np.full(N, 0.5)
    best_score = _max_overlap_value(best_h)

    def try_update(h_candidate):
        nonlocal best_h, best_score
        h_candidate = np.clip(h_candidate, 0.0, 1.0)
        diff = HALF - np.sum(h_candidate)
        if abs(diff) > 1e-10:
            # Adjust
            h_candidate = h_candidate + diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            diff = HALF - np.sum(h_candidate)
            if abs(diff) > 1e-10:
                idx = np.argmin(np.abs(h_candidate - 0.5))
                h_candidate[idx] += diff
                h_candidate[idx] = np.clip(h_candidate[idx], 0.0, 1.0)
        score = _max_overlap_value(h_candidate)
        if score < best_score:
            best_score = score
            best_h = h_candidate.copy()
        return score

    def projected_gradient_descent(h0, max_iter, tau_schedule, lr_schedule):
        """Projected gradient descent on soft-min of autocorrelation."""
        h = h0.copy()
        h = _project_to_box(h, HALF)
        current_score = _max_overlap_value(h)

        for it in range(max_iter):
            if _time_remaining(deadline) < 5.0:
                break

            # Get tau and learning rate
            frac = it / max(max_iter - 1, 1)
            tau = tau_schedule(frac)
            lr = lr_schedule(frac)

            # Compute soft-min of autocorr and its gradient
            val, grad_h = _soft_min_ac(h, tau)

            # We want to maximize min ac, so ascend in grad_h
            h_new = h + lr * grad_h

            # Project to box + sum constraint
            h_new = _project_to_box(h_new, HALF)

            # Check improvement
            new_score = _max_overlap_value(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
            else:
                # Try smaller step
                h_new2 = h + 0.5 * lr * grad_h
                h_new2 = _project_to_box(h_new2, HALF)
                new_score2 = _max_overlap_value(h_new2)
                if new_score2 < current_score:
                    h = h_new2
                    current_score = new_score2

            # Update best
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()

        return h, current_score

    # --- Phase 1: Multiple random initializations with projected gradient descent ---
    n_restarts = 12
    for r in range(n_restarts):
        if _time_remaining(deadline) < 15.0:
            break

        # Random initialization
        h_init = np.random.uniform(0.3, 0.7, N)
        h_init = _project_to_box(h_init, HALF)

        # Try a few different tau and lr schedules
        for trial in range(2):
            if _time_remaining(deadline) < 10.0:
                break

            # Schedule 1: tau from 1.0 to 0.05, lr from 0.05 to 0.005
            def tau_sched1(frac):
                return 1.0 * (1 - frac) ** 2 + 0.05
            def lr_sched1(frac):
                return 0.05 * (1 - frac) ** 1.5 + 0.005

            h_final, score_final = projected_gradient_descent(
                h_init, max_iter=800,
                tau_schedule=tau_sched1, lr_schedule=lr_sched1
            )
            try_update(h_final)

            # Schedule 2: different parameters
            def tau_sched2(frac):
                return 0.5 * (1 - frac) ** 1.5 + 0.02
            def lr_sched2(frac):
                return 0.03 * (1 - frac) + 0.003

            h_final2, score_final2 = projected_gradient_descent(
                h_final, max_iter=500,
                tau_schedule=tau_sched2, lr_schedule=lr_sched2
            )
            try_update(h_final2)

    # --- Phase 2: Binary-like initializations ---
    for r in range(6):
        if _time_remaining(deadline) < 15.0:
            break

        # Random binary with exactly N/2 ones
        ones_idx = np.random.choice(N, N // 2, replace=False)
        h_bin = np.zeros(N)
        h_bin[ones_idx] = 1.0

        # Perturb slightly
        h_init = h_bin + 0.1 * np.random.randn(N)
        h_init = _project_to_box(h_init, HALF)

        def tau_sched3(frac):
            return 0.3 * (1 - frac) ** 2 + 0.03
        def lr_sched3(frac):
            return 0.04 * (1 - frac) ** 1.2 + 0.004

        h_final, score_final = projected_gradient_descent(
            h_init, max_iter=600,
            tau_schedule=tau_sched3, lr_schedule=lr_sched3
        )
        try_update(h_final)

    # --- Phase 3: Fine-tuning with very small tau ---
    if _time_remaining(deadline) > 20.0:
        h_start = best_h.copy()

        def tau_sched4(frac):
            return 0.05 * (1 - frac) ** 2 + 0.005
        def lr_sched4(frac):
            return 0.01 * (1 - frac) + 0.001

        h_final, score_final = projected_gradient_descent(
            h_start, max_iter=1000,
            tau_schedule=tau_sched4, lr_schedule=lr_sched4
        )
        try_update(h_final)

    # --- Phase 4: Coordinate descent polish ---
    if _time_remaining(deadline) > 10.0:
        h = best_h.copy()
        for _ in range(50):
            if _time_remaining(deadline) < 5.0:
                break
            improved = False
            for i in range(N):
                if _time_remaining(deadline) < 5.0:
                    break
                # Try adjusting h[i] slightly
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    h_new = h.copy()
                    # Compensate sum by adjusting another element
                    j = (i + 1) % N
                    h_new[i] += delta
                    h_new[j] -= delta
                    h_new = np.clip(h_new, 0.0, 1.0)
                    diff = HALF - np.sum(h_new)
                    if abs(diff) > 1e-10:
                        h_new = h_new + diff / N
                        h_new = np.clip(h_new, 0.0, 1.0)
                    score = _max_overlap_value(h_new)
                    if score < best_score - 1e-10:
                        h = h_new
                        improved = True
                        break
            if not improved:
                break
        try_update(h)

    # Final projection and validation
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = HALF - np.sum(best_h)
    if abs(diff) > 1e-10:
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        diff = HALF - np.sum(best_h)
        if abs(diff) > 1e-10:
            idx = np.argmin(np.abs(best_h - 0.5))
            best_h[idx] += diff
            best_h[idx] = np.clip(best_h[idx], 0.0, 1.0)

    # Fallback if something went wrong
    if not np.all((best_h >= -1e-9) & (best_h <= 1 + 1e-9)):
        best_h = np.full(N, 0.5)

    return best_h
