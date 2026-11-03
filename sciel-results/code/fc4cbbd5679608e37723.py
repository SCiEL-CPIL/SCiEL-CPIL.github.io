import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    def max_overlap(h):
        """Compute max overlap of h with shifted 1-h."""
        H = np.fft.fft(h)
        C = np.real(np.fft.ifft(H * np.conj(H)))
        min_C = np.min(C)
        return N/2 - min_C

    def local_search(h, max_iters=5000):
        """Greedy local search: try swapping pairs of values to reduce max_overlap."""
        h = h.copy()
        current_val = max_overlap(h)
        best_val = current_val
        best_h = h.copy()
        no_improve = 0
        rng = np.random.RandomState(42)

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            if no_improve > 100:
                break

            # Pick two random indices
            i, j = rng.randint(N), rng.randint(N)
            if i == j:
                continue

            # Try swapping h[i] and h[j]
            old_i, old_j = h[i], h[j]
            h[i], h[j] = old_j, old_i

            new_val = max_overlap(h)
            if new_val < current_val - 1e-12:
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
                no_improve = 0
            else:
                # Revert
                h[i], h[j] = old_i, old_j
                no_improve += 1

        return best_h, best_val

    def simulated_annealing(h, n_iterations=50000, T_start=0.5, T_end=0.01):
        """Simulated annealing with binary-like swaps."""
        h = h.copy()
        current_val = max_overlap(h)
        best_val = current_val
        best_h = h.copy()
        rng = np.random.RandomState(123)

        for it in range(n_iterations):
            if time.time() > deadline - 10:
                break

            T = T_start * (T_end / T_start) ** (it / max(n_iterations, 1))

            # Pick two random indices
            i, j = rng.randint(N), rng.randint(N)
            if i == j:
                continue

            old_i, old_j = h[i], h[j]
            h[i], h[j] = old_j, old_i

            new_val = max_overlap(h)
            delta = new_val - current_val

            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i], h[j] = old_i, old_j

        return best_h, best_val

    # Initialize with a structured pattern
    def make_initial_pattern(seed=0):
        rng = np.random.RandomState(seed)
        h = np.zeros(N)
        # Create a pattern with some structure
        pattern_len = rng.randint(3, 8)
        for i in range(N):
            if (i // pattern_len) % 2 == 0:
                h[i] = 0.6 + 0.2 * rng.random()
            else:
                h[i] = 0.4 - 0.2 * rng.random()
        h = np.clip(h, 0, 1)
        # Adjust to sum to N/2
        diff = N/2 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0, 1)
        # Fix sum again
        diff = N/2 - np.sum(h)
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
        else:
            h += diff / N
        h = np.clip(h, 0, 1)
        return h

    # Try multiple initial patterns
    best_h = None
    best_val = float('inf')

    for seed in range(10):
        if time.time() > deadline - 20:
            break

        h_init = make_initial_pattern(seed)

        # Phase 1: Simulated annealing
        h_sa, val_sa = simulated_annealing(h_init, n_iterations=30000, T_start=0.3, T_end=0.005)

        # Phase 2: Local search polish
        h_ls, val_ls = local_search(h_sa, max_iters=3000)

        if val_ls < best_val:
            best_val = val_ls
            best_h = h_ls.copy()

    # Additional restarts with different parameters if time allows
    if time.time() < deadline - 15:
        for seed in range(10, 20):
            if time.time() > deadline - 10:
                break

            h_init = make_initial_pattern(seed * 7 + 3)
            h_sa, val_sa = simulated_annealing(h_init, n_iterations=20000, T_start=0.2, T_end=0.003)
            h_ls, val_ls = local_search(h_sa, max_iters=2000)

            if val_ls < best_val:
                best_val = val_ls
                best_h = h_ls.copy()

    # Final local search on best solution
    if time.time() < deadline - 5:
        h_final, val_final = local_search(best_h, max_iters=5000)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final

    # Ensure sum is exactly N/2
    diff = N/2 - np.sum(best_h)
    if abs(diff) > 1e-14:
        mask = (best_h > 1e-8) & (best_h < 1 - 1e-8)
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
        else:
            best_h += diff / N
        best_h = np.clip(best_h, 0, 1)

    return best_h
