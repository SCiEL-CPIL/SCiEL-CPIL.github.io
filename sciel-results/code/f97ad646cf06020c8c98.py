import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # Store as 0-indexed
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    total_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)  # shape (total_aps, 4)

    # For each position, precompute which APs contain it and the other 3 indices
    pos_ap_indices = []
    pos_ap_others = []
    for p in range(N):
        indices = []
        others = []
        for i in range(total_aps):
            row = aps[i]
            for j in range(4):
                if row[j] == p:
                    indices.append(i)
                    others.append([row[k] for k in range(4) if k != j])
                    break
        pos_ap_indices.append(np.array(indices, dtype=np.int32))
        pos_ap_others.append(np.array(others, dtype=np.int32))  # shape (n_aps, 3)

    def count_mono(colors):
        c = colors[aps]
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())

    def fast_delta(colors, p):
        """Compute delta in mono count if we flip position p."""
        old_color = colors[p]
        ap_idxs = pos_ap_indices[p]
        n_aps = len(ap_idxs)
        if n_aps == 0:
            return 0

        others = pos_ap_others[p]  # shape (n_aps, 3)
        other_colors = colors[others]  # shape (n_aps, 3)

        # Before flip: mono if all 3 others have same color AND that color == old_color
        all_same = (other_colors[:, 0] == other_colors[:, 1]) & (other_colors[:, 1] == other_colors[:, 2])
        mono_before = all_same & (other_colors[:, 0] == old_color)

        # After flip: new_color = 1 - old_color
        new_color = 1 - old_color
        mono_after = all_same & (other_colors[:, 0] == new_color)

        return int(mono_after.sum()) - int(mono_before.sum())

    def multi_bit_delta(colors, positions):
        """Compute delta for flipping multiple positions simultaneously."""
        old_count = count_mono(colors)
        for p in positions:
            colors[p] ^= 1
        new_count = count_mono(colors)
        for p in positions:
            colors[p] ^= 1  # restore
        return new_count - old_count

    def local_search(colors, max_iter=50):
        """Hill climbing with fast delta."""
        current_count = count_mono(colors)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            for p in range(N):
                delta = fast_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
            iters += 1
        return current_count

    # Generate diverse initial configurations
    candidates = []

    # Block patterns: (i // bs) % 2
    for bs in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20, 32]:
        candidates.append(np.array([(i // bs) % 2 for i in range(N)], dtype=np.int8))

    # Stripe patterns: (i % period) < period//2
    for p in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        candidates.append(np.array([(i % p) < p // 2 for i in range(N)], dtype=np.int8))

    # XOR of two periodic stripe patterns
    for p1 in [3, 4, 5, 6, 7, 8, 9, 10]:
        for p2 in [5, 7, 9, 11, 13, 15]:
            if p1 != p2:
                a = (np.arange(N) % p1) < p1 // 2
                b = (np.arange(N) % p2) < p2 // 2
                candidates.append((a ^ b).astype(np.int8))

    # Random inits
    for _ in range(40):
        candidates.append(np.random.randint(0, 2, size=N).astype(np.int8))

    # Evaluate and refine all candidates with quick local search
    scored = []
    for c in candidates:
        c = c.copy()
        sc = local_search(c, max_iter=30)
        scored.append((sc, c))

    scored.sort(key=lambda x: x[0])

    # Track best overall
    best_score = scored[0][0]
    best_color = scored[0][1].copy()

    # Genetic Algorithm
    rng = random.Random(12345)
    pop_size = 20
    elite_count = 4
    mutation_rate = 0.04

    # Initialize population from top candidates
    population = [scored[i][1].copy() for i in range(min(pop_size, len(scored)))]
    while len(population) < pop_size:
        population.append(np.random.randint(0, 2, size=N).astype(np.int8))

    ga_time_budget = deadline - 20.0

    while time.time() < ga_time_budget:
        # Evaluate population
        scored_pop = []
        for ind in population:
            sc = count_mono(ind)
            scored_pop.append((sc, ind))

        scored_pop.sort(key=lambda x: x[0])

        # Update best
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_color = scored_pop[0][1].copy()

        # Keep elites
        new_population = [ind.copy() for _, ind in scored_pop[:elite_count]]

        # Generate offspring
        scores_list = [s for s, _ in scored_pop]

        while len(new_population) < pop_size:
            if time.time() > ga_time_budget - 5:
                break

            # Tournament selection
            def tournament_select():
                cands = rng.sample(range(len(population)), min(3, len(population)))
                best_idx = min(cands, key=lambda i: scores_list[i])
                return population[best_idx].copy()

            parent1 = tournament_select()
            parent2 = tournament_select()

            # Uniform crossover
            child = np.array([
                parent1[i] if rng.random() < 0.5 else parent2[i]
                for i in range(N)
            ], dtype=np.int8)

            # Mutation
            for i in range(N):
                if rng.random() < mutation_rate:
                    child[i] ^= 1

            # Quick local search on child
            local_search(child, max_iter=20)

            new_population.append(child)

        population = new_population

    # Final SA refinement on best individual
    if time.time() < deadline - 5:
        sa_colors = best_color.copy()
        current_count = count_mono(sa_colors)

        T = 2.0
        T_min = 0.02
        cooling = 0.9998

        sa_deadline = deadline - 3
        step = 0
        while time.time() < sa_deadline:
            step += 1

            r = rng.random()
            if r < 0.65:
                # 1-bit flip
                p = rng.randint(0, N - 1)
                delta = fast_delta(sa_colors, p)

                if delta <= 0:
                    sa_colors[p] ^= 1
                    current_count += delta
                    if current_count < best_score:
                        best_score = current_count
                        best_color = sa_colors.copy()
                elif T > T_min:
                    if rng.random() < np.exp(-delta / T):
                        sa_colors[p] ^= 1
                        current_count += delta
            elif r < 0.85:
                # 2-bit flip
                p1, p2 = rng.sample(range(N), 2)
                delta = multi_bit_delta(sa_colors, [p1, p2])

                if delta <= 0:
                    sa_colors[p1] ^= 1
                    sa_colors[p2] ^= 1
                    current_count += delta
                    if current_count < best_score:
                        best_score = current_count
                        best_color = sa_colors.copy()
                elif T > T_min:
                    if rng.random() < np.exp(-delta / T):
                        sa_colors[p1] ^= 1
                        sa_colors[p2] ^= 1
                        current_count += delta
            else:
                # 3-bit flip
                p1, p2, p3 = rng.sample(range(N), 3)
                delta = multi_bit_delta(sa_colors, [p1, p2, p3])

                if delta <= 0:
                    sa_colors[p1] ^= 1
                    sa_colors[p2] ^= 1
                    sa_colors[p3] ^= 1
                    current_count += delta
                    if current_count < best_score:
                        best_score = current_count
                        best_color = sa_colors.copy()
                elif T > T_min:
                    if rng.random() < np.exp(-delta / T):
                        sa_colors[p1] ^= 1
                        sa_colors[p2] ^= 1
                        sa_colors[p3] ^= 1
                        current_count += delta

            T = max(T_min, T * cooling)

    # Final local search polish
    if time.time() < deadline - 2:
        final = best_color.copy()
        local_search(final, max_iter=100)
        final_count = count_mono(final)
        if final_count < best_score:
            best_score = final_count
            best_color = final

    return [int(c) for c in best_color]
