"""
Two-colourings of [1,256] with few monochromatic four-term progressions.
Strategy: diverse structured initialization, fast greedy local search,
simulated annealing with vectorized delta computation, and a genetic algorithm
with segment crossover and targeted mutation.
"""
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115

    # Precompute all 4-term arithmetic progressions (0-indexed: positions 0..255)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # For each position, which APs does it participate in?
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps_arr[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))

    pos_ap_indices = []
    pos_ap_offsets = []
    for i in range(n):
        if pos_aps[i]:
            indices, offsets = zip(*pos_aps[i])
            pos_ap_indices.append(np.array(indices, dtype=np.int32))
            pos_ap_offsets.append(np.array(offsets, dtype=np.int8))
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_ap_offsets.append(np.array([], dtype=np.int8))

    def count_mono_4ap(color):
        c = color[aps_arr]
        mono = (c[:, 1] == c[:, 0]) & (c[:, 2] == c[:, 0]) & (c[:, 3] == c[:, 0])
        return int(np.sum(mono))

    def delta_flip(color, pos):
        """Compute the change in mono-AP count if we flip position pos."""
        ap_idx = pos_ap_indices[pos]
        if len(ap_idx) == 0:
            return 0
        sub = color[aps_arr[ap_idx]]
        off = pos_ap_offsets[pos]
        curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
        curr_count = int(np.sum(curr_mono))
        new_sub = sub.copy()
        new_sub[np.arange(len(ap_idx)), off] = 1 - new_sub[np.arange(len(ap_idx)), off]
        new_mono = (new_sub[:, 1] == new_sub[:, 0]) & (new_sub[:, 2] == new_sub[:, 0]) & (new_sub[:, 3] == new_sub[:, 0])
        new_count = int(np.sum(new_mono))
        return new_count - curr_count

    def greedy_local_search(color, score, time_budget=3.0):
        """Greedy local search: keep flipping if it improves."""
        improved = True
        start = time.time()
        while improved and time.time() - start < time_budget:
            improved = False
            for pos in range(n):
                delta = delta_flip(color, pos)
                if delta < 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                    improved = True
        return color, score

    def simulated_annealing(color, score, time_budget=15.0):
        """Simulated annealing with single-flip moves and dynamic temperature."""
        start = time.time()
        T_start = 2.5
        T_end = 0.01
        best_score = score
        best_color = color.copy()

        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0.0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * remaining_frac ** 1.3

            pos = np.random.randint(0, n)
            delta = delta_flip(color, pos)

            if delta <= 0:
                color[pos] = 1 - color[pos]
                score += delta
            else:
                accept_prob = np.exp(-delta / max(T, 0.001))
                if np.random.random() < accept_prob:
                    color[pos] = 1 - color[pos]
                    score += delta

            if score < best_score:
                best_score = score
                best_color = color.copy()

        return best_color, best_score

    def multi_flip_sa(color, score, time_budget=12.0):
        """Simulated annealing with multi-flip moves for escaping local minima."""
        start = time.time()
        T_start = 3.5
        T_end = 0.005
        best_score = score
        best_color = color.copy()

        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0.0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * remaining_frac ** 1.5

            if np.random.random() < 0.55:
                pos = np.random.randint(0, n)
                delta = delta_flip(color, pos)
                if delta <= 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                else:
                    if np.random.random() < np.exp(-delta / max(T, 0.001)):
                        color[pos] = 1 - color[pos]
                        score += delta
            else:
                # Multi-flip: pick 2-4 random distinct positions
                k = np.random.randint(2, 5)
                positions = np.random.choice(n, size=k, replace=False)
                # Compute delta for multi-flip
                affected_aps = set()
                for p in positions:
                    if len(pos_ap_indices[p]) > 0:
                        affected_aps.update(pos_ap_indices[p].tolist())
                if affected_aps:
                    ap_list = np.array(sorted(affected_aps), dtype=np.int32)
                    sub = color[aps_arr[ap_list]]
                    curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
                    curr_count = int(np.sum(curr_mono))
                    new_sub = sub.copy()
                    for p in positions:
                        off_arr = pos_ap_offsets[p]
                        idx_arr = pos_ap_indices[p]
                        for j, ap_i in enumerate(idx_arr):
                            row_in_new = np.where(ap_list == ap_i)[0]
                            if len(row_in_new) > 0:
                                new_sub[row_in_new[0], off_arr[j]] = 1 - new_sub[row_in_new[0], off_arr[j]]
                    new_mono = (new_sub[:, 1] == new_sub[:, 0]) & (new_sub[:, 2] == new_sub[:, 0]) & (new_sub[:, 3] == new_sub[:, 0])
                    new_count = int(np.sum(new_mono))
                    delta = new_count - curr_count
                    if delta <= 0:
                        for p in positions:
                            color[p] = 1 - color[p]
                        score += delta
                    else:
                        if np.random.random() < np.exp(-delta / max(T, 0.001)):
                            for p in positions:
                                color[p] = 1 - color[p]
                            score += delta

            if score < best_score:
                best_score = score
                best_color = color.copy()

        return best_color, best_score

    def genetic_algorithm(population, scores, time_budget=30.0):
        """
        Genetic Algorithm with tournament selection, segment crossover, and mutation.
        """
        start = time.time()
        pop_size = len(population)
        best_score = min(scores)
        best_idx = int(np.argmin(scores))
        best_color = population[best_idx].copy()

        pop = [p.copy() for p in population]
        sc = list(scores)

        while time.time() - start < time_budget:
            remaining = time_budget - (time.time() - start)
            if remaining < 0.1:
                break

            new_pop = [pop[best_idx].copy()]
            new_sc = [sc[best_idx]]

            while len(new_pop) < pop_size:
                # Tournament selection
                k = 3
                idxs = np.random.choice(pop_size, k, replace=False)
                parent1_idx = int(idxs[np.argmin([sc[i] for i in idxs])])
                idxs2 = np.random.choice(pop_size, k, replace=False)
                parent2_idx = int(idxs2[np.argmin([sc[i] for i in idxs2])])

                p1 = pop[parent1_idx]
                p2 = pop[parent2_idx]

                # Segment crossover: pick a random contiguous segment from one parent
                if np.random.random() < 0.5:
                    seg_start = np.random.randint(0, n - 16)
                    seg_len = np.random.randint(16, 64)
                    child = p1.copy()
                    child[seg_start:seg_start + seg_len] = p2[seg_start:seg_start + seg_len]
                else:
                    # Uniform crossover
                    mask = np.random.randint(0, 2, n)
                    child = np.where(mask == 1, p1, p2).astype(np.int8)

                # Mutation: flip a few bits
                num_mutations = np.random.randint(1, 6)
                mut_positions = np.random.choice(n, num_mutations, replace=False)
                child[mut_positions] = 1 - child[mut_positions]

                # Refine with short local search
                child_score = count_mono_4ap(child)
                ls_time = max(0.02, remaining * 0.008)
                if ls_time > 0.01:
                    child, child_score = greedy_local_search(child, child_score, time_budget=ls_time)

                new_pop.append(child)
                new_sc.append(child_score)

            pop = new_pop
            sc = new_sc

            current_best_idx = int(np.argmin(sc))
            if sc[current_best_idx] < best_score:
                best_score = sc[current_best_idx]
                best_color = pop[current_best_idx].copy()
                best_idx = current_best_idx

        return best_color, best_score

    def generate_structured_colorings():
        """Generate a diverse set of structured candidate colorings."""
        candidates = []

        # Thue-Morse
        tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(tm.copy())

        # Bit-reversal Thue-Morse
        br = np.array([bin(i)[::-1].count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(br.copy())

        # Modular patterns
        for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            for q in range(1, p):
                c = np.array([(i % p) < q for i in range(n)], dtype=np.int8)
                candidates.append(c)

        # Walsh functions
        for k in range(8):
            c = np.array([(i >> k) & 1 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # XOR combinations of bits (all non-zero masks up to 255)
        for mask in range(1, 256):
            c = np.array([bin(i & mask).count('1') % 2 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # Random starting points with fixed seed for reproducibility
        rng = np.random.default_rng(42)
        for _ in range(40):
            c = rng.integers(0, 2, n).astype(np.int8)
            candidates.append(c)

        return candidates

    best_color = None
    best_score = float('inf')

    # Evaluate all structured candidates
    candidates = generate_structured_colorings()
    scored = []
    for c in candidates:
        s = count_mono_4ap(c)
        scored.append((s, c.copy()))

    scored.sort(key=lambda x: x[0])

    # Take top candidates for refinement
    top_k = min(20, len(scored))

    # Phase 1: Greedy local search on top candidates
    refined_candidates = []
    for rank in range(top_k):
        if time.time() > deadline - 55:
            break

        score, color = scored[rank]
        if score < best_score:
            best_score = score
            best_color = color.copy()

        ls_budget = min(1.5, (deadline - time.time() - 55) * 0.04)
        if ls_budget > 0.1:
            color_ls, score_ls = greedy_local_search(color.copy(), score, time_budget=ls_budget)
            refined_candidates.append((score_ls, color_ls))
            if score_ls < best_score:
                best_score = score_ls
                best_color = color_ls.copy()
        else:
            refined_candidates.append((score, color.copy()))

    refined_candidates.sort(key=lambda x: x[0])

    # Phase 2: Simulated annealing on top refined candidates
    sa_refined = []
    for rank in range(min(10, len(refined_candidates))):
        if time.time() > deadline - 40:
            break

        score, color = refined_candidates[rank]
        sa_budget = min(5.0, (deadline - time.time() - 40) * 0.1)
        if sa_budget > 0.5:
            color_sa, score_sa = simulated_annealing(color.copy(), score, time_budget=sa_budget)
            sa_refined.append((score_sa, color_sa))
            if score_sa < best_score:
                best_score = score_sa
                best_color = color_sa.copy()

    sa_refined.sort(key=lambda x: x[0])

    # Phase 3: Genetic Algorithm
    pop_size = 24
    ga_population = []
    ga_scores = []

    # Add top refined and SA-refined candidates
    for s, c in refined_candidates[:8]:
        ga_population.append(c)
        ga_scores.append(s)
    for s, c in sa_refined[:8]:
        ga_population.append(c)
        ga_scores.append(s)

    # Add random ones to ensure diversity
    rng = np.random.default_rng(123)
    while len(ga_population) < pop_size:
        c = rng.integers(0, 2, n).astype(np.int8)
        s = count_mono_4ap(c)
        ga_population.append(c)
        ga_scores.append(s)

    if time.time() < deadline - 25:
        ga_budget = deadline - time.time() - 25
        ga_color, ga_score = genetic_algorithm(ga_population, ga_scores, time_budget=ga_budget)
        if ga_score < best_score:
            best_score = ga_score
            best_color = ga_color.copy()

    # Phase 4: Final polish with multi-flip SA and greedy LS
    if best_color is not None and time.time() < deadline - 10:
        final_budget = deadline - time.time() - 8
        color_final, score_final = greedy_local_search(best_color.copy(), best_score, time_budget=final_budget * 0.4)
        if score_final < best_score:
            best_score = score_final
            best_color = color_final.copy()

        if time.time() < deadline - 12:
            mf_budget = min(8.0, deadline - time.time() - 12)
            color_mf, score_mf = multi_flip_sa(best_color.copy(), best_score, time_budget=mf_budget)
            if score_mf < best_score:
                best_score = score_mf
                best_color = color_mf.copy()

        if time.time() < deadline - 5:
            color_final2, score_final2 = greedy_local_search(best_color.copy(), best_score, time_budget=deadline - time.time() - 5)
            if score_final2 < best_score:
                best_score = score_final2
                best_color = color_final2.copy()

    if best_color is None:
        best_color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    return [int(c) for c in best_color]
