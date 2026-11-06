import time
import random
import numpy as np

N = 16
NUM_EDGES = 120
NUM_K4 = 1820

def get_edge_idx(i, j):
    if i > j:
        i, j = j, i
    return 15 * i - i * (i - 1) // 2 + (j - i - 1)

# Precompute K4s and edge mappings
K4_EDGE_INDICES = []
EDGE_TO_K4S = [[] for _ in range(NUM_EDGES)]

k4_idx = 0
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
                e_idxs = [get_edge_idx(u, v) for u, v in edges]
                K4_EDGE_INDICES.append(e_idxs)
                for e in e_idxs:
                    EDGE_TO_K4S[e].append(k4_idx)
                k4_idx += 1

def calculate_initial_penalty(colors, K):
    """
    Calculate initial penalty and setup counts for incremental updates.
    Returns (penalty, k4_counts, k4_distinct)
    k4_counts: list of lists, k4_counts[k][c] = count of color c in k4 k
    k4_distinct: list of ints, k4_distinct[k] = number of distinct colors in k4 k
    """
    k4_counts = [[0] * K for _ in range(NUM_K4)]
    k4_distinct = [0] * NUM_K4
    penalty = 0

    for k in range(NUM_K4):
        e_idxs = K4_EDGE_INDICES[k]
        for e in e_idxs:
            c = colors[e]
            k4_counts[k][c] += 1

        distinct = sum(1 for count in k4_counts[k] if count > 0)
        k4_distinct[k] = distinct
        if distinct < 5:
            penalty += (5 - distinct)

    return penalty, k4_counts, k4_distinct

def cyclic_seed(K):
    """
    Cyclic 1-factorization based seed (inspiration 1).
    Vertices 0..14 in Z_15, vertex 15 as infinity.
    This gives a structured low-penalty starting point.
    Colors are reduced mod K to fit the palette.
    """
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            if i == 15 or j == 15:
                base = j if i == 15 else i
            else:
                base = (8 * (i + j)) % 15
            colors[get_edge_idx(i, j)] = base % K
    return colors

def random_seed(K):
    return np.random.randint(0, K, size=NUM_EDGES).tolist()

def verify_valid(colors):
    """Exact verification: every K4 has >=5 distinct colors."""
    for e_idxs in K4_EDGE_INDICES:
        c0 = colors[e_idxs[0]]
        c1 = colors[e_idxs[1]]
        c2 = colors[e_idxs[2]]
        c3 = colors[e_idxs[3]]
        c4 = colors[e_idxs[4]]
        c5 = colors[e_idxs[5]]
        mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << c3) | (1 << c4) | (1 << c5)
        distinct = bin(mask).count('1')
        if distinct < 5:
            return False
    return True

def sa_run(colors, K, k4_counts, k4_distinct, current_penalty, time_budget, T0, T_min, alpha, rng):
    """
    Simulated annealing with incremental penalty updates.
    Returns (colors, current_penalty, best_colors, best_penalty, best_K).
    """
    best_colors = colors.copy()
    best_penalty = current_penalty
    T = T0
    t_start = time.time()

    while time.time() - t_start < time_budget:
        # Pick random edge
        e_idx = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e_idx]

        # Pick new color
        new_color = rng.randint(0, K - 1)
        if new_color == old_color:
            continue

        # Calculate delta penalty
        delta = 0
        affected_k4s = EDGE_TO_K4S[e_idx]

        # Store old distinct values for rollback
        old_distincts = [0] * len(affected_k4s)

        for i, k in enumerate(affected_k4s):
            old_distincts[i] = k4_distinct[k]

            cnt = k4_counts[k]

            # Remove old color
            cnt[old_color] -= 1
            if cnt[old_color] == 0:
                k4_distinct[k] -= 1

            # Add new color
            cnt[new_color] += 1
            if cnt[new_color] == 1:
                k4_distinct[k] += 1

            # Recalculate penalty contribution for this K4
            old_pen = 5 - old_distincts[i] if old_distincts[i] < 5 else 0
            new_pen = 5 - k4_distinct[k] if k4_distinct[k] < 5 else 0
            delta += new_pen - old_pen

        # Acceptance
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept
            colors[e_idx] = new_color
            current_penalty += delta

            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_colors = colors.copy()
                if best_penalty == 0:
                    return colors, current_penalty, best_colors, best_penalty, K
        else:
            # Reject: Revert changes
            for i, k in enumerate(affected_k4s):
                cnt = k4_counts[k]

                # Revert new color
                cnt[new_color] -= 1
                if cnt[new_color] == 0:
                    k4_distinct[k] -= 1

                # Revert old color
                cnt[old_color] += 1
                if cnt[old_color] == 1:
                    k4_distinct[k] += 1

        T *= alpha
        if T < T_min:
            T = T_min

    return colors, current_penalty, best_colors, best_penalty, K

def polish(colors, K, time_budget, rng):
    """
    Final high-precision polish: strict local search (only accept improvements)
    to remove any residual violations.
    """
    current_penalty, k4_counts, k4_distinct = calculate_initial_penalty(colors, K)
    if current_penalty == 0:
        return colors

    best_colors = colors.copy()
    best_penalty = current_penalty
    t_start = time.time()
    no_improve = 0

    while time.time() - t_start < time_budget:
        e_idx = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e_idx]
        new_color = rng.randint(0, K - 1)
        if new_color == old_color:
            continue

        delta = 0
        affected_k4s = EDGE_TO_K4S[e_idx]
        old_distincts = [0] * len(affected_k4s)

        for i, k in enumerate(affected_k4s):
            old_distincts[i] = k4_distinct[k]
            cnt = k4_counts[k]
            cnt[old_color] -= 1
            if cnt[old_color] == 0:
                k4_distinct[k] -= 1
            cnt[new_color] += 1
            if cnt[new_color] == 1:
                k4_distinct[k] += 1
            old_pen = 5 - old_distincts[i] if old_distincts[i] < 5 else 0
            new_pen = 5 - k4_distinct[k] if k4_distinct[k] < 5 else 0
            delta += new_pen - old_pen

        if delta < 0:
            colors[e_idx] = new_color
            current_penalty += delta
            no_improve = 0
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_colors = colors.copy()
                if best_penalty == 0:
                    return colors
        else:
            # Revert
            for i, k in enumerate(affected_k4s):
                cnt = k4_counts[k]
                cnt[new_color] -= 1
                if cnt[new_color] == 0:
                    k4_distinct[k] -= 1
                cnt[old_color] += 1
                if cnt[old_color] == 1:
                    k4_distinct[k] += 1
            no_improve += 1

    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for final checks and return

    best_colors = None
    best_penalty = float('inf')
    best_K = 17  # Fallback K

    # Fallback: The seed solution
    fallback_colors = [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]

    # Try K = 13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 10.0:
            break

        # Time budget per K
        time_budget_for_K = 35.0
        k_start = time.time()

        # Multiple restarts
        while time.time() - k_start < time_budget_for_K:
            if time.time() > deadline - 5.0:
                break

            # Alternate between cyclic seed and random seed for diversity
            use_cyclic = (random.random() < 0.5)
            if use_cyclic:
                colors = cyclic_seed(K)
            else:
                colors = random_seed(K)

            # Calculate initial penalty and state
            current_penalty, k4_counts, k4_distinct = calculate_initial_penalty(colors, K)

            if current_penalty == 0:
                return colors

            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_colors = colors.copy()
                best_K = K

            # Simulated Annealing with varied parameters
            T = random.uniform(0.5, 2.0)
            T_min = 0.01
            alpha = random.uniform(0.998, 0.9995)
            time_per_run = min(5.0, max(1.0, deadline - 5.0 - time.time()))

            colors, current_penalty, run_best_colors, run_best_penalty, _ = sa_run(
                colors, K, k4_counts, k4_distinct, current_penalty,
                time_per_run, T, T_min, alpha, random
            )

            if run_best_penalty < best_penalty:
                best_penalty = run_best_penalty
                best_colors = run_best_colors
                if best_penalty == 0:
                    return best_colors

        if best_penalty == 0:
            return best_colors

    # Final polish pass on the best solution found
    if best_colors is not None and best_penalty > 0:
        if time.time() < deadline - 5.0:
            polished = polish(best_colors, best_K, min(10.0, deadline - 5.0 - time.time()), random)
            if verify_valid(polished):
                best_colors = polished
                best_penalty = 0
            else:
                # Recompute penalty for polished
                pen, _, _ = calculate_initial_penalty(polished, best_K)
                if pen < best_penalty:
                    best_colors = polished
                    best_penalty = pen

    # Check if best_colors is valid (penalty 0)
    if best_colors is not None:
        # Re-verify penalty for best_colors to be safe
        if verify_valid(best_colors):
            return best_colors

    # Fallback to seed solution (17 colors)
    return fallback_colors
