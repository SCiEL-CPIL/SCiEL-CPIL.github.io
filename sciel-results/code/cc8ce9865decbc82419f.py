import numpy as np
import time

def evaluate_measure(s, w):
    """
    Fast evaluation of P[X1 + X2 + X3 < 2*X4].
    P = sum_{k,l} p_k * p_l * sum_{i,j: s_i+s_j < 2*s_l - s_k} p_i * p_j
    """
    n = len(s)
    if n == 0:
        return 0.0

    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    p = w / w_sum
    p = np.maximum(p, 0.0)
    s = np.maximum(s, 0.0)

    # Re-normalize after clipping
    p_sum = np.sum(p)
    if p_sum <= 1e-9:
        return 0.0
    p = p / p_sum

    # Build pair sums and probs
    ss = s[:, None] + s[None, :]
    pp = p[:, None] * p[None, :]

    ss_flat = ss.flatten()
    pp_flat = pp.flatten()

    # Sort by sum
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    pp_sorted = pp_flat[sort_idx]

    # Cumulative sum of probs
    C = np.cumsum(pp_sorted)

    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            idx = np.searchsorted(ss_sorted, threshold, side='left')
            if idx > 0:
                prob_inner = C[idx - 1]
            else:
                prob_inner = 0.0
            P += p[k] * p[l] * prob_inner

    return P

def optimize_weights_for_fixed_support(s, time_limit):
    """
    Optimizes weights for a fixed support using coordinate descent + random restarts.
    """
    n = len(s)
    best_w = np.ones(n) / n
    best_score = evaluate_measure(s, best_w)

    start_time = time.time()
    rng = np.random.default_rng(42)

    max_restarts = 30

    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break

        # Random initialization
        w = rng.dirichlet(np.ones(n))
        current_score = evaluate_measure(s, w)

        if current_score > best_score:
            best_score = current_score
            best_w = w.copy()

        # Coordinate descent
        no_improve_count = 0
        max_no_improve = 100

        for _ in range(5000):
            if time.time() - start_time > time_limit:
                break
            if no_improve_count > max_no_improve:
                break

            improved = False
            # Random order for coordinates
            order = rng.permutation(n)
            for idx in order:
                if w[idx] < 1e-10:
                    continue

                # Try to move mass from idx to other indices
                # Find the best target
                best_target = -1
                best_new_score = current_score

                # Sample a few targets to save time
                num_targets = min(n - 1, 20)
                if n > 1:
                    targets = rng.choice(n, size=num_targets, replace=False)
                    for t in targets:
                        if t == idx:
                            continue
                        # Move a small fraction of mass
                        for frac in [0.01, 0.05, 0.1, 0.2]:
                            move = w[idx] * frac
                            w_trial = w.copy()
                            w_trial[idx] -= move
                            w_trial[t] += move

                            if np.any(w_trial < -1e-9):
                                continue

                            score_trial = evaluate_measure(s, w_trial)
                            if score_trial > best_new_score:
                                best_new_score = score_trial
                                best_target = t
                                best_move = move

                if best_target >= 0:
                    w[idx] -= best_move
                    w[best_target] += best_move
                    current_score = best_new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_w = w.copy()

            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0

    return best_w, best_score

def generate_support_candidates(deadline, safety_margin):
    """
    Generate diverse support candidates and evaluate them with uniform weights.
    """
    candidates = []
    rng = np.random.default_rng(123)

    # 1. Geometric sequences with various bases
    for base in [1.2, 1.3, 1.5, 2.0, 2.5, 3.0]:
        for n in [10, 15, 20]:
            s = base ** np.arange(n)
            s = s / s[-1]  # Normalize to [0,1]
            candidates.append(s)

    # 2. Two-cluster: many points near 0 and 1
    for n1 in [5, 10]:
        for n2 in [5, 10]:
            s1 = np.linspace(0, 0.1, n1)
            s2 = np.linspace(0.9, 1.0, n2)
            s = np.concatenate([s1, s2])
            candidates.append(s)

    # 3. Exponential with noise
    for n in [15, 20]:
        s = np.exp(rng.uniform(0, 2, n))
        s = (s - s.min()) / (s.max() - s.min())
        candidates.append(s)

    # 4. Power laws
    for alpha in [0.5, 1.0, 2.0]:
        for n in [15, 20]:
            s = np.arange(1, n + 1) ** alpha
            s = s / s[-1]
            candidates.append(s)

    # 5. Random sorted
    for n in [15, 20]:
        s = np.sort(rng.uniform(0, 1, n))
        candidates.append(s)

    # Evaluate all candidates with uniform weights and keep the best
    scored = []
    for s in candidates:
        n = len(s)
        w = np.ones(n) / n
        score = evaluate_measure(s, w)
        scored.append((score, s))

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])

    return scored[:10]  # Keep top 10

def beat_the_average_measure() -> tuple:
    deadline = time.time() + 600.0
    safety_margin = 10.0

    # Generate candidate supports
    scored_supports = generate_support_candidates(deadline, safety_margin)

    best_support = None
    best_weights = None
    best_score = -1.0

    if not scored_supports:
        # Fallback
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n) / n
        return best_support, best_weights

    # Process top candidates
    num_to_optimize = min(len(scored_supports), 8)

    for i in range(num_to_optimize):
        if time.time() + 5.0 > deadline - safety_margin:
            break

        seed_score, s = scored_supports[i]

        # Update best if seed is better
        if seed_score > best_score:
            best_score = seed_score
            best_support = s.copy()
            best_weights = np.ones(len(s)) / len(s)

        # Allocate time for optimization
        remaining = deadline - safety_margin - time.time()
        if remaining < 5.0:
            break

        # Give more time to earlier (better) supports
        time_alloc = remaining * (0.5 ** i)
        time_alloc = max(2.0, min(80.0, time_alloc))

        try:
            w_opt, score_opt = optimize_weights_for_fixed_support(s, time_alloc)

            if score_opt > best_score:
                best_score = score_opt
                best_support = s.copy()
                best_weights = w_opt.copy()

        except Exception:
            pass

    # Fallback
    if best_support is None:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n) / n

    # Final validation and cleanup
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()

    # Ensure non-negative
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)

    # Normalize weights
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum
    else:
        n = len(best_support)
        best_weights = np.ones(n) / n

    return best_support, best_weights

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(f"Score: {evaluate_measure(s, w)}")
    print(f"n = {len(s)}")
