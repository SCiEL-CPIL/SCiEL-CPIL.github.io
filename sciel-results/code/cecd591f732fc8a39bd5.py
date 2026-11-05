import math
import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 135  # safety margin before 150s kill

    # Precompute all 4-APs as numpy arrays
    aps_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            aps_list.append((a, a+d, a+2*d, a+3*d))

    aps_np = np.array(aps_list, dtype=np.int32)  # shape (M, 4)
    M = len(aps_np)

    # For each position, precompute which APs it belongs to and the other 3 positions
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d_pos) in enumerate(aps_list):
        pos_aps[a].append((idx, b, c, d_pos))
        pos_aps[b].append((idx, a, c, d_pos))
        pos_aps[c].append((idx, a, b, d_pos))
        pos_aps[d_pos].append((idx, a, b, c))

    # Convert to numpy arrays for fast lookup
    pos_aps_arr = []
    for i in range(N):
        if pos_aps[i]:
            arr = np.array(pos_aps[i], dtype=np.int32)
            pos_aps_arr.append(arr)
        else:
            pos_aps_arr.append(np.empty((0, 4), dtype=np.int32))

    # Precompute the count of APs per position for weighted random selection
    ap_counts = np.array([len(pos_aps[i]) for i in range(N)], dtype=np.float64)
    ap_probs = ap_counts / ap_counts.sum()

    def count_mono(cols_np):
        """Vectorized count of monochromatic 4-APs."""
        c0 = cols_np[aps_np[:, 0]]
        c1 = cols_np[aps_np[:, 1]]
        c2 = cols_np[aps_np[:, 2]]
        c3 = cols_np[aps_np[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))

    def delta_for_flip(cols_np, i):
        """Calculate cost delta if we flip position i."""
        c_i = cols_np[i]
        arr = pos_aps_arr[i]
        if len(arr) == 0:
            return 0
        others = arr[:, 1:]  # shape (k, 3)
        c_others = cols_np[others]  # shape (k, 3)
        all_same = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
        n_same = int(np.sum(all_same))
        if n_same == 0:
            return 0
        matching = int(np.sum(c_others[all_same, 0] == c_i))
        return -(matching) + (n_same - matching)

    def greedy_polish(cols, current_cost, max_iters=50):
        """Try all single flips and apply any that reduce cost. Iterate until no improvement."""
        cols_np = np.array(cols, dtype=np.int8)
        for _ in range(max_iters):
            improved = False
            for i in range(N):
                d = delta_for_flip(cols_np, i)
                if d < 0:
                    cols[i] = 1 - cols[i]
                    cols_np[i] = 1 - cols_np[i]
                    current_cost += d
                    improved = True
            if not improved:
                break
        return cols, current_cost

    def thue_morse_coloring():
        """Generate Thue-Morse coloring (1-indexed)."""
        colors = [0] * N
        for i in range(N):
            n = i + 1
            colors[i] = bin(n).count('1') % 2
        return colors

    def thue_morse_shifted(shift):
        """Thue-Morse with a shift."""
        colors = [0] * N
        for i in range(N):
            n = (i + 1 + shift) % (N + 1)
            if n == 0:
                n = N + 1
            colors[i] = bin(n).count('1') % 2
        return colors

    def generate_structured_patterns():
        """Generate diverse structured initial colorings."""
        patterns = []

        # Thue-Morse
        patterns.append(thue_morse_coloring())

        # Thue-Morse shifted
        for shift in range(1, 10):
            patterns.append(thue_morse_shifted(shift))

        # Inverted Thue-Morse
        tm = thue_morse_coloring()
        patterns.append([1 - c for c in tm])

        # Block patterns
        for block in [2, 3, 4, 5, 6, 7, 8, 12, 16]:
            p = [(i // block) % 2 for i in range(N)]
            patterns.append(p)

        # Alternating
        patterns.append([i % 2 for i in range(N)])

        # XOR patterns
        patterns.append([(i & 1) ^ ((i >> 1) & 1) for i in range(N)])
        patterns.append([(i & 1) ^ ((i >> 2) & 1) for i in range(N)])
        patterns.append([(i >> 1) ^ ((i >> 3) & 1) for i in range(N)])

        # Random with different seeds
        for seed in [12345, 67890, 11111, 22222]:
            rng_init = random.Random(seed)
            patterns.append([rng_init.randint(0, 1) for _ in range(N)])

        # Evaluate all and return top 6
        scored = []
        for p in patterns:
            cost = count_mono(np.array(p, dtype=np.int8))
            scored.append((cost, p))
        scored.sort(key=lambda x: x[0])
        return [p for _, p in scored[:6]]

    # Get initial patterns
    initial_patterns = generate_structured_patterns()

    global_best_colors = initial_patterns[0][:]
    global_best_cost = count_mono(np.array(global_best_colors, dtype=np.int8))

    # Simulated annealing with hybrid selection
    def simulated_annealing(initial_colors, T_start, cooling, min_T, time_limit, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        colors = initial_colors[:]
        cols_np = np.array(colors, dtype=np.int8)
        current_cost = count_mono(cols_np)
        best_cost = current_cost
        best_colors = colors[:]

        T = T_start
        end_time = time.time() + time_limit
        steps = 0

        while time.time() < end_time:
            steps += 1

            # Hybrid selection: 70% weighted, 30% uniform
            if rng.random() < 0.7:
                i = rng.choices(range(N), weights=ap_probs, k=1)[0]
            else:
                i = rng.randint(0, N - 1)

            delta = delta_for_flip(cols_np, i)

            if delta < 0:
                colors[i] = 1 - colors[i]
                cols_np[i] = 1 - cols_np[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            elif delta == 0:
                if rng.random() < 0.4:
                    colors[i] = 1 - colors[i]
                    cols_np[i] = 1 - cols_np[i]
            else:
                if T > min_T:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        colors[i] = 1 - colors[i]
                        cols_np[i] = 1 - cols_np[i]
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colors = colors[:]

            # Periodic kick: every 2000 steps, do a burst of random flips
            if steps % 2000 == 0 and rng.random() < 0.3:
                n_kicks = rng.randint(3, 8)
                for _ in range(n_kicks):
                    ki = rng.randint(0, N - 1)
                    colors[ki] = 1 - colors[ki]
                    cols_np[ki] = 1 - cols_np[ki]
                current_cost = count_mono(cols_np)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]

            T *= cooling
            if T < min_T:
                T = min_T

        return best_colors, best_cost

    # Multiple restarts
    num_restarts = 14
    time_per_restart = 120.0 / num_restarts

    for restart in range(num_restarts):
        if time.time() > deadline - 12:
            break

        # Choose initial
        if restart < len(initial_patterns):
            initial = initial_patterns[restart % len(initial_patterns)][:]
        elif restart < 8:
            # Perturb best significantly
            initial = global_best_colors[:]
            n_flips = random.randint(N // 3, N // 2)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        else:
            # Moderate perturbation of best
            initial = global_best_colors[:]
            n_flips = random.randint(N // 8, N // 4)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]

        # SA with varying parameters
        T_start = 1.5 + 0.4 * (restart % 5)
        cooling = 0.99997 - 0.000001 * (restart % 7)
        min_T = 0.004
        seed = 42 + restart * 137

        sa_colors, sa_cost = simulated_annealing(
            initial, T_start, cooling, min_T, time_per_restart * 0.85, seed=seed
        )

        # Greedy polish
        polished_colors, polished_cost = greedy_polish(sa_colors, sa_cost)

        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_colors = polished_colors[:]

    # Final polish with extra iterations
    if time.time() < deadline - 5:
        final_colors, final_cost = greedy_polish(global_best_colors, global_best_cost, max_iters=100)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors

    return global_best_colors
