import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _luby_sequence():
    """Infinite Luby sequence for set-cover greedy."""
    m = 0
    while True:
        while m & 1 == 0:
            yield m
            m += 1
        yield m
        m += 1


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561

    # Precompute balls (neighborhoods of radius 1)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Ball as flat set for fast membership
    ball_flat = balls.reshape(-1)

    # Precompute digit representations
    digit_rep = np.zeros((total, n), dtype=np.int8)
    for w in range(total):
        temp = w
        for i in range(n - 1, -1, -1):
            digit_rep[w, i] = temp % 3
            temp //= 3

    # Precompute all-pairs Hamming distance (for diversity in greedy)
    # Use vectorized computation in chunks to save memory
    def compute_hamming_chunk(a_indices, b_indices):
        """Compute Hamming distances between a_indices and b_indices."""
        da = digit_rep[a_indices]  # (len_a, n)
        db = digit_rep[b_indices]  # (len_b, n)
        # (len_a, 1, n) vs (1, len_b, n)
        diff = (da[:, np.newaxis, :] != db[np.newaxis, :, :])
        return diff.sum(axis=2)

    # --- Spectral tools ---
    # The coverage function c(v) = number of selected codewords within dist 1 of v
    # Its DFT: C(k) = sum_v c(v) * omega^{k*v} where omega = exp(2*pi*i/total)
    # For a single codeword w, its contribution to C is: B_w(k) = sum_{u in ball(w)} omega^{k*u}
    # We precompute B_w for all w and all k in a subset of frequencies.

    # Use a smaller frequency set for speed: sample K frequencies
    K_freq = 256  # number of frequency bins to track
    rng_freq = np.random.RandomState(12345)
    freq_indices = rng_freq.choice(total, size=K_freq, replace=False)
    omega = np.exp(2j * np.pi / total)

    # Precompute B_w(k) for all w and sampled frequencies
    # B_w(k) = sum_{u in ball(w)} omega^{k*u}
    # We can compute this as: for each frequency k, B_w(k) = sum over ball(w) of omega^{k*u}
    # Vectorize: for each freq, compute phase for all words, then sum over balls
    phase = np.zeros((K_freq, total), dtype=np.complex64)
    for fi, fi_val in enumerate(freq_indices):
        phase[fi] = omega ** (fi_val * np.arange(total))

    # B_w(fi) = sum_{u in ball(w)} phase[fi, u]
    # balls is (total, 17), so B_w(fi) = sum of phase[fi, balls[w, :]]
    B_all = np.zeros((total, K_freq), dtype=np.complex64)
    for fi in range(K_freq):
        B_all[:, fi] = phase[fi, balls].sum(axis=1)

    # Spectral magnitude for a coverage vector: |sum_{w in S} B_w|^2 at each frequency
    # This is a proxy for how "spread out" the coverage is

    def greedy_construct(seed):
        """Greedy with spectral diversity bias."""
        rng = np.random.RandomState(seed)
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        luby_iter = _luby_sequence()
        luby_val = 0

        # Spectral accumulator: sum of B_w for selected w
        spectral_acc = np.zeros(K_freq, dtype=np.complex64)

        while np.any(uncovered_mask):
            if time.time() > deadline - 15.0:
                return None, None

            # Count coverage for each candidate among uncovered words
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]

            if len(candidates) > 1:
                if luby_val < 2:
                    best_idx = candidates[rng.randint(len(candidates))]
                else:
                    # Spectral diversity: pick candidate that maximizes spectral novelty
                    # i.e., whose B_w adds most new spectral content
                    cand_B = B_all[candidates]  # (len_cand, K_freq)
                    # Current spectral magnitude
                    curr_mag = np.abs(spectral_acc) ** 2  # (K_freq,)
                    # New spectral magnitude if we add candidate i
                    # |spectral_acc + B_i|^2 - |spectral_acc|^2 = 2*Re(spectral_acc * conj(B_i)) + |B_i|^2
                    # = 2*Re(spectral_acc * conj(B_i)) + sum(|B_i|^2)
                    cross = 2.0 * np.real(spectral_acc[np.newaxis, :] * np.conj(cand_B))  # (len_cand, K_freq)
                    self_mag = np.abs(cand_B) ** 2  # (len_cand, K_freq)
                    novelty = (cross + self_mag).sum(axis=1)  # (len_cand,)

                    # Combine with coverage count (prefer high coverage)
                    cov_scores = counts[candidates].astype(np.float64)
                    cov_max = np.max(cov_scores)
                    cov_norm = cov_scores / max(cov_max, 1)

                    # Blend: 70% novelty, 30% coverage
                    combined = 0.7 * novelty / max(np.max(novelty), 1e-10) + 0.3 * cov_norm
                    best_idx = candidates[np.argmax(combined)]
                luby_val += 1
            else:
                best_idx = candidates[0]

            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            spectral_acc += B_all[best_idx]

        if np.any(uncovered_mask):
            return None, None
        return selected, cover_count

    # --- Phase 1: Multiple greedy restarts ---
    best_size = total
    best_set = None
    best_cc = None

    for restart in range(6):
        if time.time() > deadline - 25.0:
            break
        sel, cc = greedy_construct(restart * 7919 + 13)
        if sel is None:
            continue
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = sel
            best_cc = cc.copy()

    if best_set is None:
        # Deterministic fallback
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected
        best_cc = cover_count.copy()
        best_size = len(selected)

    # --- Phase 2: Spectral-guided local search ---
    selected = best_set.copy()
    cover_count = best_cc.copy()

    def try_remove_redundant(sel, cc):
        """Remove all redundant codewords."""
        new_sel = []
        changed = False
        for c in sel:
            if np.min(cc[balls[c]]) > 1:
                cc[balls[c]] -= 1
                changed = True
            else:
                new_sel.append(c)
        return new_sel, cc, changed

    def spectral_redundancy_score(sel, cc):
        """
        For each codeword c in sel, compute how much its removal affects the spectral content.
        A codeword is spectrally redundant if its B_w is well-approximated by the sum of others.
        Score = |B_c - (total_spectral - B_c)|_2^2 / |total_spectral|^2
        Lower score = more redundant.
        """
        if len(sel) == 0:
            return np.array([])

        sel_B = B_all[sel]  # (len_sel, K_freq)
        total_spec = sel_B.sum(axis=0)  # (K_freq,)
        total_mag = np.abs(total_spec) ** 2 + 1e-10

        # For removing codeword i: new_spec = total_spec - B_i
        # Residual = B_i (the part that was unique)
        # Spectral change = |B_i|^2 - 2*Re(B_i * conj(total_spec - B_i))
        # = |B_i|^2 - 2*Re(B_i * conj(total_spec)) + 2*Re(B_i * conj(B_i))
        # = |B_i|^2 - 2*Re(B_i * conj(total_spec)) + 2*|B_i|^2
        # = 3*|B_i|^2 - 2*Re(B_i * conj(total_spec))
        # Simpler: just use |total_spec - B_i|^2 - |total_spec|^2 as "loss"
        # = |B_i|^2 - 2*Re(B_i * conj(total_spec))
        # Lower loss = more redundant

        cross = 2.0 * np.real(sel_B * np.conj(total_spec[np.newaxis, :]))  # (len_sel, K_freq)
        self_mag = np.abs(sel_B) ** 2  # (len_sel, K_freq)
        loss = (self_mag - cross).sum(axis=1) / total_mag.sum()  # (len_sel,)
        return loss

    def try_swap_improve(sel, cc):
        """Try 1-swap improvements."""
        improved = False
        new_sel = []
        for c in sel:
            if np.min(cc[balls[c]]) > 1:
                cc[balls[c]] -= 1
                improved = True
                continue

            critical_mask = (cc[balls[c]] == 1)
            critical_words = balls[c][critical_mask]

            if len(critical_words) == 0:
                cc[balls[c]] -= 1
                improved = True
                continue

            # Find candidates that cover all critical words
            candidate_mask = np.ones(total, dtype=bool)
            for cw in critical_words:
                candidate_mask &= np.isin(ball_flat, balls[cw])

            valid_candidates = np.where(candidate_mask)[0]
            valid_candidates = valid_candidates[valid_candidates != c]

            if len(valid_candidates) > 0:
                # Score by how many additional words they cover
                scores = np.zeros(len(valid_candidates), dtype=np.int32)
                for i, cp in enumerate(valid_candidates):
                    extra_cover = np.sum(cc[balls[cp]] == 1) - len(critical_words)
                    scores[i] = extra_cover

                best_i = np.argmax(scores)
                c_prime = valid_candidates[best_i]

                cc[balls[c]] -= 1
                cc[balls[c_prime]] += 1
                new_sel.append(c_prime)
                improved = True
            else:
                new_sel.append(c)

        return new_sel, cc, improved

    def try_2_exchange(sel, cc, rng, max_trials=400):
        """Try 2-exchange: remove 2 codewords, add 1."""
        if len(sel) < 3:
            return None, None, 0

        for _ in range(max_trials):
            if time.time() > deadline - 5.0:
                return None, None, 0

            idx1 = rng.randint(len(sel))
            idx2 = rng.randint(len(sel))
            if idx1 == idx2:
                continue

            c1 = sel[idx1]
            c2 = sel[idx2]

            cc[balls[c1]] -= 1
            cc[balls[c2]] -= 1

            uncovered = np.where(cc == 0)[0]

            if len(uncovered) == 0:
                new_sel = [c for i, c in enumerate(sel) if i != idx1 and i != idx2]
                return new_sel, cc, -2

            if len(uncovered) <= 10:
                candidate_mask = np.ones(total, dtype=bool)
                for uw in uncovered:
                    candidate_mask &= np.isin(ball_flat, balls[uw])

                valid_candidates = np.where(candidate_mask)[0]
                valid_candidates = valid_candidates[
                    (valid_candidates != c1) & (valid_candidates != c2)]

                if len(valid_candidates) > 0:
                    scores = np.zeros(len(valid_candidates), dtype=np.int32)
                    for i, cp in enumerate(valid_candidates):
                        scores[i] = np.sum(cc[balls[cp]] == 0)
                    best_i = np.argmax(scores)
                    c3 = valid_candidates[best_i]

                    cc[balls[c3]] += 1
                    new_sel = [c for i, c in enumerate(sel) if i != idx1 and i != idx2]
                    new_sel.append(c3)
                    return new_sel, cc, -1

            cc[balls[c1]] += 1
            cc[balls[c2]] += 1

        return None, None, 0

    # Main local search loop with spectral guidance
    T = 5.0
    T_end = 0.1
    cool_rate = 0.97
    rng = np.random.RandomState(42)

    while time.time() < deadline - 5.0:
        # Phase A: Remove redundant
        selected, cover_count, changed = try_remove_redundant(selected, cover_count)
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

        # Phase B: Spectral-guided swap - target the most spectrally redundant codeword
        if time.time() < deadline - 5.0 and len(selected) > 0:
            # Identify spectrally weakest codewords
            if len(selected) > 20:
                spec_scores = spectral_redundancy_score(selected, cover_count)
                # Sort by spectral redundancy (lowest = most redundant)
                spec_order = np.argsort(spec_scores)
                # Try to swap the most redundant ones first
                for trial_idx in spec_order[:min(5, len(spec_order))]:
                    c = selected[trial_idx]
                    if np.min(cover_count[balls[c]]) > 1:
                        continue  # already redundant, handled in Phase A

                    critical_mask = (cover_count[balls[c]] == 1)
                    critical_words = balls[c][critical_mask]
                    if len(critical_words) == 0:
                        continue

                    candidate_mask = np.ones(total, dtype=bool)
                    for cw in critical_words:
                        candidate_mask &= np.isin(ball_flat, balls[cw])

                    valid_candidates = np.where(candidate_mask)[0]
                    valid_candidates = valid_candidates[valid_candidates != c]

                    if len(valid_candidates) > 0:
                        # Pick the one with best spectral complement
                        cand_B = B_all[valid_candidates]
                        curr_spec = B_all[selected].sum(axis=0)
                        cross = 2.0 * np.real(cand_B * np.conj(curr_spec[np.newaxis, :]))
                        self_mag = np.abs(cand_B) ** 2
                        novelty = (cross + self_mag).sum(axis=1)
                        best_i = np.argmax(novelty)
                        c_prime = valid_candidates[best_i]

                        cover_count[balls[c]] -= 1
                        cover_count[balls[c_prime]] += 1
                        selected[trial_idx] = c_prime
                        selected, cover_count, _ = try_remove_redundant(selected, cover_count)
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_set = selected.copy()
                        break
            else:
                # Fallback: standard swap
                new_sel, new_cc, improved = try_swap_improve(selected, cover_count)
                if improved:
                    selected = new_sel
                    cover_count = new_cc
                    selected, cover_count, _ = try_remove_redundant(selected, cover_count)
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_set = selected.copy()

        # Phase C: Try 2-exchange
        if len(selected) > 420 and time.time() < deadline - 5.0:
            new_sel, new_cc, size_delta = try_2_exchange(selected, cover_count, rng,
                                                         max_trials=300)
            if new_sel is not None and size_delta < 0:
                selected = new_sel
                cover_count = new_cc
                selected, cover_count, _ = try_remove_redundant(selected, cover_count)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected.copy()

        # Phase D: SA perturbation
        if len(selected) > 420 and time.time() < deadline - 5.0:
            idx = rng.randint(len(selected))
            c_old = selected[idx]
            c_new = rng.randint(total)

            if c_new != c_old:
                cover_count[balls[c_old]] -= 1
                cover_count[balls[c_new]] += 1

                uncovered = np.sum(cover_count == 0)

                if uncovered == 0:
                    selected[idx] = c_new
                else:
                    delta = float(uncovered)
                    if rng.random() < np.exp(-delta / T):
                        selected[idx] = c_new
                    else:
                        cover_count[balls[c_new]] -= 1
                        cover_count[balls[c_old]] += 1

        T *= cool_rate
        if T < T_end:
            T = T_end

        if best_size <= 403:
            break

    # --- Phase 3: Final polish ---
    if best_set is not None:
        selected = best_set.copy()
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1

        for _ in range(10):
            if time.time() > deadline - 1.0:
                break

            changed = True
            while changed and time.time() < deadline - 1.0:
                changed = False
                new_selected = []
                for c in selected:
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        changed = True
                    else:
                        new_selected.append(c)
                selected = new_selected

            if time.time() < deadline - 1.0:
                new_selected, cover_count, improved = try_swap_improve(selected, cover_count)
                if improved:
                    selected = new_selected

            if len(selected) > 420 and time.time() < deadline - 1.0:
                new_sel, new_cc, size_delta = try_2_exchange(selected, cover_count, rng,
                                                             max_trials=200)
                if new_sel is not None and size_delta < 0:
                    selected = new_sel
                    cover_count = new_cc

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

    # Convert to output format
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result
