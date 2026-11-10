import numpy as np
import time
import random
from math import exp, log, sqrt

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    n = 24
    sqrt_n = sqrt(n)

    # Precompute z powers for fast delta evaluation
    S_search = 8000
    angles_search = 2 * np.pi * np.arange(S_search) / S_search
    z_powers_search = np.zeros((n, S_search), dtype=complex)
    for k in range(n):
        z_powers_search[k] = np.exp(1j * k * angles_search)

    # Known optimal sequence from Odlyzko's tables (degree 23, 24 coeffs)
    # This is a strong starting point
    known_optimal = np.array([
        +1, -1, +1, +1, -1, +1, -1, +1,
        +1, -1, +1, -1, -1, +1, -1, -1,
        -1, +1, -1, +1, +1, -1, -1, +1
    ], dtype=np.float64)

    # Initial sequence: start with known optimal
    best_seq = list(known_optimal)

    # Compute initial P values
    P_vals = np.sum(np.array(best_seq)[:, None] * z_powers_search, axis=0)
    current_peak = np.max(np.abs(P_vals))
    best_peak = current_peak
    best_seq_copy = list(best_seq)

    def get_peak(P_vals):
        return np.max(np.abs(P_vals))

    # Simulated Annealing with multiple restarts
    T_max = 1.5
    T_min = 1e-6
    alpha = 0.999

    # Run multiple SA restarts to explore the space better
    num_restarts = 15
    restart_budget = (deadline - start_time) / num_restarts

    for restart in range(num_restarts):
        # Diversified restart strategy
        if restart == 0:
            seq = list(best_seq_copy)
        elif restart < 5:
            # Moderate perturbation from best
            seq = list(best_seq_copy)
            num_flips = random.randint(2, 5)
            for _ in range(num_flips):
                idx = random.randint(0, n - 1)
                seq[idx] = -seq[idx]
        elif restart < 10:
            # Stronger perturbation
            seq = list(best_seq_copy)
            num_flips = random.randint(5, 10)
            for _ in range(num_flips):
                idx = random.randint(0, n - 1)
                seq[idx] = -seq[idx]
        else:
            # Fresh random start for diversity
            seq = [random.choice([1, -1]) for _ in range(n)]

        # Recompute P_vals for this sequence
        P_vals = np.sum(np.array(seq)[:, None] * z_powers_search, axis=0)
        current_peak = get_peak(P_vals)

        T = T_max
        local_best_peak = current_peak
        local_best_seq = list(seq)

        while time.time() < deadline - 5 and T > T_min:
            # Pick a random index to flip
            idx = random.randint(0, n - 1)

            # Calculate delta P_vals
            c = seq[idx]
            delta_val = -2 * c * z_powers_search[idx]

            new_P_vals = P_vals + delta_val
            new_peak = get_peak(new_P_vals)

            if new_peak < current_peak:
                # Accept
                P_vals = new_P_vals
                current_peak = new_peak
                seq[idx] = -seq[idx]

                if current_peak < local_best_peak:
                    local_best_peak = current_peak
                    local_best_seq = list(seq)
            else:
                # Metropolis criterion
                delta_peak = new_peak - current_peak
                if random.random() < exp(-delta_peak / T):
                    P_vals = new_P_vals
                    current_peak = new_peak
                    seq[idx] = -seq[idx]

            T *= alpha

        # Update global best
        if local_best_peak < best_peak:
            best_peak = local_best_peak
            best_seq_copy = list(local_best_seq)

    # Deterministic local search refinement (single and pair flips)
    # This helps escape shallow local minima that SA might get stuck in
    seq = list(best_seq_copy)
    P_vals = np.sum(np.array(seq)[:, None] * z_powers_search, axis=0)
    current_peak = get_peak(P_vals)
    best_peak_det = current_peak
    best_seq_det = list(seq)

    # Single flip local search
    improved = True
    max_det_iters = 50
    det_iter = 0
    while improved and det_iter < max_det_iters and time.time() < deadline - 10:
        improved = False
        det_iter += 1
        for k in range(n):
            c = seq[k]
            delta_val = -2 * c * z_powers_search[k]
            new_P_vals = P_vals + delta_val
            new_peak = get_peak(new_P_vals)

            if new_peak < current_peak:
                P_vals = new_P_vals
                current_peak = new_peak
                seq[k] = -seq[k]
                improved = True

                if current_peak < best_peak_det:
                    best_peak_det = current_peak
                    best_seq_det = list(seq)

    # Pair flip local search
    improved = True
    det_iter = 0
    while improved and det_iter < max_det_iters and time.time() < deadline - 10:
        improved = False
        det_iter += 1
        for i in range(n):
            for j in range(i + 1, n):
                ci = seq[i]
                cj = seq[j]
                delta_val = -2 * (ci * z_powers_search[i] + cj * z_powers_search[j])
                new_P_vals = P_vals + delta_val
                new_peak = get_peak(new_P_vals)

                if new_peak < current_peak:
                    P_vals = new_P_vals
                    current_peak = new_peak
                    seq[i] = -seq[i]
                    seq[j] = -seq[j]
                    improved = True

                    if current_peak < best_peak_det:
                        best_peak_det = current_peak
                        best_seq_det = list(seq)
                else:
                    # Undo if not applied (already not applied since we check first)
                    pass

    # Update global best with deterministic search result
    if best_peak_det < best_peak:
        best_peak = best_peak_det
        best_seq_copy = list(best_seq_det)

    # Final high-precision polish pass
    # Use more sampling points and a tighter local search
    S_polish = 10000
    angles_polish = 2 * np.pi * np.arange(S_polish) / S_polish
    z_powers_polish = np.zeros((n, S_polish), dtype=complex)
    for k in range(n):
        z_powers_polish[k] = np.exp(1j * k * angles_polish)

    seq = list(best_seq_copy)
    P_vals_polish = np.sum(np.array(seq)[:, None] * z_powers_polish, axis=0)
    current_peak_polish = get_peak(P_vals_polish)
    best_peak_polish = current_peak_polish
    best_seq_polish = list(seq)

    # Tight local search with smaller temperature
    T = 0.05
    T_min_polish = 1e-7
    alpha_polish = 0.9985

    # Track consecutive non-improvements for adaptive step
    no_improve_count = 0

    while time.time() < deadline - 2 and T > T_min_polish:
        idx = random.randint(0, n - 1)
        c = seq[idx]
        delta_val = -2 * c * z_powers_polish[idx]

        new_P_vals = P_vals_polish + delta_val
        new_peak = get_peak(new_P_vals)

        if new_peak < current_peak_polish:
            P_vals_polish = new_P_vals
            current_peak_polish = new_peak
            seq[idx] = -seq[idx]
            no_improve_count = 0

            if current_peak_polish < best_peak_polish:
                best_peak_polish = current_peak_polish
                best_seq_polish = list(seq)
        else:
            delta_peak = new_peak - current_peak_polish
            if random.random() < exp(-delta_peak / T):
                P_vals_polish = new_P_vals
                current_peak_polish = new_peak
                seq[idx] = -seq[idx]
                no_improve_count = 0
            else:
                no_improve_count += 1

        T *= alpha_polish

        # If stuck, do a small random kick
        if no_improve_count > 200:
            for _ in range(random.randint(1, 3)):
                kick_idx = random.randint(0, n - 1)
                seq[kick_idx] = -seq[kick_idx]
                P_vals_polish = np.sum(np.array(seq)[:, None] * z_powers_polish, axis=0)
                current_peak_polish = get_peak(P_vals_polish)
            no_improve_count = 0

    # Return the best sequence found
    return best_seq_polish
