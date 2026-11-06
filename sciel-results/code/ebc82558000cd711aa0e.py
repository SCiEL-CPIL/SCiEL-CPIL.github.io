"""Model-and-optimize approach: structured parameterization + local search refinement."""
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110  # safety margin before 150s kill

    # Precompute all 4-term arithmetic progressions
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

    def local_search(color, score, time_budget=5.0):
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

    def simulated_annealing(color, score, time_budget=10.0):
        """Simulated annealing with single and multi-flip moves."""
        start = time.time()
        T_start = 4.0
        T_end = 0.005
        it = 0
        # Track best seen during SA
        sa_best_score = score
        sa_best_color = color.copy()

        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * remaining_frac ** 1.5

            # Choose move type: 70% single flip, 30% multi-flip (2-4 positions)
            if np.random.random() < 0.7:
                # Single flip
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
                # Compute total delta by flipping all at once
                # We need to compute the score change for flipping multiple positions simultaneously
                # Recompute affected APs
                affected_aps = set()
                for p in positions:
                    if len(pos_ap_indices[p]) > 0:
                        affected_aps.update(pos_ap_indices[p].tolist())
                if affected_aps:
                    ap_list = np.array(list(affected_aps), dtype=np.int32)
                    sub = color[aps_arr[ap_list]]
                    # Count current mono
                    curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
                    curr_count = int(np.sum(curr_mono))
                    # Apply flips
                    new_sub = sub.copy()
                    for p in positions:
                        off_arr = pos_ap_offsets[p]
                        idx_arr = pos_ap_indices[p]
                        # Find which rows in new_sub correspond to these AP indices
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

            # Track best
            if score < sa_best_score:
                sa_best_score = score
                sa_best_color = color.copy()
            it += 1

        # Return best seen during SA
        return sa_best_color, sa_best_score

    def generate_structured_colorings():
        """Generate a set of structured candidate colorings (model-based)."""
        candidates = []

        # Thue-Morse
        tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(tm.copy())

        # Bit-reversal Thue-Morse
        br = np.array([bin(i)[::-1].count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(br.copy())

        # Modular patterns: color[i] = (i mod p) < q for various p, q
        for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            for q in range(1, p):
                c = np.array([(i % p) < q for i in range(n)], dtype=np.int8)
                candidates.append(c)
            # Also flipped
            for q in range(1, p):
                c = np.array([(i % p) >= q for i in range(n)], dtype=np.int8)
                candidates.append(c)

        # Walsh functions: color[i] = bit k of i
        for k in range(8):
            c = np.array([(i >> k) & 1 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # XOR combinations of bits (Walsh-Hadamard basis)
        for mask in range(1, 256):
            c = np.array([bin(i & mask).count('1') % 2 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # Linear phase: color[i] = (a*i + b) mod 2
        for a in range(1, 8):
            for b in range(2):
                c = np.array([(a * i + b) % 2 for i in range(n)], dtype=np.int8)
                candidates.append(c)

        # Quadratic: color[i] = (a*i*i + b*i + c) mod 2
        for a in range(1, 4):
            for b in range(2):
                for c in range(2):
                    col = np.array([(a * i * i + b * i + c) % 2 for i in range(n)], dtype=np.int8)
                    candidates.append(col)

        # Sine-wave quantization
        for freq in [1, 2, 3, 4, 5, 7, 8, 11]:
            for phase in range(4):
                vals = np.sin(2 * np.pi * freq * np.arange(n) / n + phase * np.pi / 2)
                c = (vals > 0).astype(np.int8)
                candidates.append(c)

        # Random starting points
        rng = np.random.default_rng(12345)
        for _ in range(20):
            c = rng.integers(0, 2, n).astype(np.int8)
            candidates.append(c)

        return candidates

    best_color = None
    best_score = float('inf')

    # Evaluate all structured candidates, keep the best few for refinement
    candidates = generate_structured_colorings()

    # Score all candidates
    scored = []
    for c in candidates:
        s = count_mono_4ap(c)
        scored.append((s, c.copy()))

    scored.sort(key=lambda x: x[0])

    # Take top candidates for refinement
    top_k = min(20, len(scored))
    remaining_time = deadline - time.time()
    time_per_candidate = max(1.0, remaining_time / (top_k + 3))

    for rank in range(top_k):
        if time.time() > deadline - 20:
            break
        score, color = scored[rank]
        if score < best_score:
            best_score = score
            best_color = color.copy()

        # Local search refinement
        ls_budget = min(4.0, time_per_candidate * 0.35)
        color_ls, score_ls = local_search(color.copy(), score, time_budget=ls_budget)
        if score_ls < best_score:
            best_score = score_ls
            best_color = color_ls.copy()

        # Simulated annealing from best local search result
        time_remaining = deadline - time.time() - 20
        if time_remaining > 3:
            sa_budget = min(12.0, time_remaining * 0.55)
            color_sa, score_sa = simulated_annealing(color_ls.copy(), score_ls, time_budget=sa_budget)
            if score_sa < best_score:
                best_score = score_sa
                best_color = color_sa.copy()

            # Final local search after SA
            ls2_budget = min(4.0, time_remaining * 0.3)
            color_final, score_final = local_search(color_sa.copy(), score_sa, time_budget=ls2_budget)
            if score_final < best_score:
                best_score = score_final
                best_color = color_final.copy()

    # Final polish on the overall best: local search + short SA + local search
    if best_color is not None and time.time() < deadline - 10:
        final_budget = deadline - time.time() - 8
        color_final, score_final = local_search(best_color.copy(), best_score, time_budget=final_budget * 0.4)
        if score_final < best_score:
            best_score = score_final
            best_color = color_final.copy()

        # Short SA from the polished result
        if time.time() < deadline - 12:
            sa_final_budget = min(10.0, deadline - time.time() - 12)
            color_sa_f, score_sa_f = simulated_annealing(best_color.copy(), best_score, time_budget=sa_final_budget)
            if score_sa_f < best_score:
                best_score = score_sa_f
                best_color = color_sa_f.copy()

            # Final local search
            if time.time() < deadline - 6:
                color_final2, score_final2 = local_search(best_color.copy(), best_score, time_budget=deadline - time.time() - 6)
                if score_final2 < best_score:
                    best_score = score_final2
                    best_color = color_final2.copy()

    if best_color is None:
        best_color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    return [int(c) for c in best_color]
