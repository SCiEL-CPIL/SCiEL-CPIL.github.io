import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    # Precompute APs: all arithmetic progressions of length >= 2 within [0, N)
    # An AP is defined by start s, step d. We include s, s+d, s+2d, ... < N
    # where len(terms) >= 2.
    ap_terms = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            terms = list(range(s, N, d))
            if len(terms) > 1:
                ap_terms.append(terms)

    num_aps = len(ap_terms)

    # Build incidence matrix M (num_aps x N) as float32
    M = np.zeros((num_aps, N), dtype=np.float32)
    pos_to_aps = [[] for _ in range(N)]
    for j, terms in enumerate(ap_terms):
        for i in terms:
            M[j, i] = 1.0
            pos_to_aps[i].append(j)

    pos_to_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]

    # C = M.T @ M, K = diag(C)
    C = (M.T @ M).astype(np.float32)
    K = np.diag(C).copy()

    def get_hard_max(sums):
        return int(np.max(np.abs(sums)))

    best_signs = None
    best_hard_max = N + 1  # slightly above worst case to ensure first update

    # We'll do multi-restart SA with sum-of-squares objective and soft-max tracking
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000,
             6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]

    for seed in seeds:
        if time.time() > deadline - 6:
            break

        rng = random.Random(seed)

        # Initialize: start with a slightly structured pattern to help convergence
        # Alternate blocks of +1/-1 to give some initial structure
        signs = np.zeros(N, dtype=np.float32)
        for i in range(N):
            # Use a pseudo-random but deterministic initialization
            signs[i] = 1.0 if rng.random() < 0.5 else -1.0

        sums = M @ signs
        G = C @ signs  # G[k] = sum_{P containing k} S_P

        current_hard_max = get_hard_max(sums)
        if current_hard_max < best_hard_max:
            best_hard_max = current_hard_max
            best_signs = signs.copy()

        # SA parameters
        T = 8.0
        T_min = 0.3
        T_decay = 0.995  # slower cooling for more thorough exploration

        steps = 0
        no_improve = 0
        last_improve_hard = current_hard_max

        # Main SA loop
        while time.time() < deadline - 3:
            k = rng.randint(0, N - 1)
            delta = -2.0 * signs[k]

            # delta_J = 2*delta*G[k] + delta^2*K[k]
            delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]

            accept = False
            if delta_J < 0:
                accept = True
            else:
                if delta_J / T > 80.0:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_J / T)

            if accept:
                signs[k] = -signs[k]
                idxs = pos_to_aps_arr[k]
                sums[idxs] += delta
                G += delta * C[:, k]
                steps += 1

                # Periodically check hard max
                if steps % 40 == 0:
                    current_hard_max = get_hard_max(sums)
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                        no_improve = 0
                        last_improve_hard = current_hard_max
                    else:
                        no_improve += 1

                    # If no improvement for a while, perturb slightly
                    if no_improve > 200:
                        # Flip a few random signs to escape local minimum
                        for _ in range(rng.randint(2, 5)):
                            kk = rng.randint(0, N - 1)
                            dd = -2.0 * signs[kk]
                            signs[kk] = -signs[kk]
                            sums[pos_to_aps_arr[kk]] += dd
                            G += dd * C[:, kk]
                        no_improve = 0

            # Annealing
            if steps % 800 == 0:
                T *= T_decay
                if T < T_min:
                    T = T_min

            # Early exit if we hit the theoretical minimum
            if best_hard_max <= 1:
                break

    # Final polish: if we have a decent solution, do a local search pass
    if best_signs is not None and time.time() < deadline - 2:
        signs = best_signs.copy()
        sums = M @ signs
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for k in range(N):
                delta = -2.0 * signs[k]
                delta_J = 2.0 * delta * np.sum(sums[pos_to_aps_arr[k]]) + (delta * delta) * K[k]
                if delta_J < -1e-9:
                    signs[k] = -signs[k]
                    sums[pos_to_aps_arr[k]] += delta
                    improved = True
            if not improved:
                # Check if hard max improved
                current_hm = get_hard_max(sums)
                if current_hm < best_hard_max:
                    best_hard_max = current_hm
                    best_signs = signs.copy()

    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]

    return [int(s) for s in best_signs]
