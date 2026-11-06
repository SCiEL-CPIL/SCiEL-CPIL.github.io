"""Combined: fast vectorized SA (current) + GA diversity (inspiration) + multi-restart local search."""
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115  # safety margin before 150s kill

    # Precompute all 4-term arithmetic progressions as index tuples (0-based)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)

    # For each position, which APs does it participate in?
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps_arr[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))

    # Convert to arrays for fast access
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
        """Count monochromatic 4-APs. color is 0/1 array of length n."""
        c = color[aps_arr]  # shape (num_aps, 4)
        mono = (c[:, 1] == c[:, 0]) & (c[:, 2] == c[:, 0]) & (c[:, 3] == c[:, 0])
        return int(np.sum(mono))

    def delta_flip(color, pos):
        """Compute change in mono-4AP count if we flip position 'pos'."""
        ap_idx = pos_ap_indices[pos]
        if len(ap_idx) == 0:
            return 0

        sub = color[aps_arr[ap_idx]]  # shape (len, 4)
        off = pos_ap_offsets[pos]  # shape (len,)

        curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
        curr_count = int(np.sum(curr_mono))

        new_sub = sub.copy()
        new_sub[np.arange(len(ap_idx)), off] = 1 - new_sub[np.arange(len(ap_idx)), off]
        new_mono = (new_sub[:, 1] == new_sub[:, 0]) & (new_sub[:, 2] == new_sub[:, 0]) & (new_sub[:, 3] == new_sub[:, 0])
        new_count = int(np.sum(new_mono))

        return new_count - curr_count

    def local_search(color, time_limit, max_passes=50):
        """Improve coloring via single-flip local search using delta computation."""
        start = time.time()
        current = count_mono_4ap(color)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for pos in range(n):
                if time.time() - start > time_limit:
                    return color, current
                delta = delta_flip(color, pos)
                if delta < 0:
                    color[pos] = 1 - color[pos]
                    current += delta
                    improved = True
            if time.time() - start > time_limit:
                return color, current
        return color, current

    def simulated_annealing(color, time_limit, T_start=2.0, T_end=0.01):
        """Run SA for a given time limit, return (color, score)."""
        start = time.time()
        current_score = count_mono_4ap(color)
        best_score = current_score
        best_color = color.copy()

        # Estimate iterations we can do
        # Rough: each iteration takes ~5-10us, so ~100k-200k in 1s
        # We'll just run until time limit
        rng = np.random.default_rng()
        T = T_start

        # We'll use a fixed number of iterations based on time
        # First, estimate time per iteration
        t0 = time.time()
        for _ in range(100):
            pos = rng.integers(0, n)
            _ = delta_flip(color, pos)
        time_per_iter = (time.time() - t0) / 100

        max_iters = int((time_limit - (time.time() - start)) / time_per_iter)
        if max_iters < 1000:
            max_iters = 1000

        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        for it in range(max_iters):
            if time.time() - start > time_limit:
                break

            pos = rng.integers(0, n)
            delta = delta_flip(color, pos)

            if delta <= 0:
                color[pos] = 1 - color[pos]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_color = color.copy()
            else:
                if rng.random() < np.exp(-delta / T):
                    color[pos] = 1 - color[pos]
                    current_score += delta

            T *= cool_rate

        return best_color, best_score

    def thue_morse():
        """Generate Thue-Morse sequence for 0..255."""
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def random_coloring(rng):
        return rng.integers(0, 2, size=n).astype(np.int8)

    def mutate(c, rng, rate=0.05):
        flips = rng.random(n) < rate
        c[flips] = 1 - c[flips]
        return c

    # Main algorithm
    rng = np.random.default_rng(42)
    best_color = thue_morse()
    best_score = count_mono_4ap(best_color)

    # Phase 1: Local search from Thue-Morse
    remaining = deadline - time.time()
    if remaining > 15:
        c, cnt = local_search(best_color.copy(), min(5.0, remaining - 10), max_passes=30)
        if cnt < best_score:
            best_score = cnt
            best_color = c

    # Phase 2: Multi-restart with SA + GA diversity
    remaining = deadline - time.time()
    if remaining > 30:
        # Generate diverse starting points
        starts = [best_color.copy(), thue_morse()]
        for _ in range(6):
            if rng.random() < 0.5:
                c = random_coloring(rng)
            else:
                c = best_color.copy()
                c = mutate(c, rng, rate=0.15)
            starts.append(c)

        # For each start, run SA + local search
        time_per_start = (remaining - 10) / len(starts)
        for i, start_c in enumerate(starts):
            if time.time() > deadline - 10:
                break

            time_left = deadline - time.time() - 10
            if time_left < 5:
                break

            # SA phase
            sa_time = min(8.0, time_left * 0.6)
            c_sa, score_sa = simulated_annealing(start_c.copy(), sa_time)

            # Local search refinement
            ls_time = min(3.0, time_left * 0.3)
            c_ls, score_ls = local_search(c_sa.copy(), ls_time, max_passes=20)

            if score_ls < best_score:
                best_score = score_ls
                best_color = c_ls

            # Also try local search directly from start
            if time.time() < deadline - 10:
                c_ls2, score_ls2 = local_search(start_c.copy(), min(2.0, deadline - time.time() - 10), max_passes=15)
                if score_ls2 < best_score:
                    best_score = score_ls2
                    best_color = c_ls2

    # Phase 3: Final thorough local search on best solution
    remaining = deadline - time.time()
    if remaining > 5:
        c, cnt = local_search(best_color.copy(), remaining - 3, max_passes=50)
        if cnt < best_score:
            best_score = cnt
            best_color = c

    return [int(c) for c in best_color]
