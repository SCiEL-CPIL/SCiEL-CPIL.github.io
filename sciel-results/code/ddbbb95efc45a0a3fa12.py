import numpy as np
import random
import time
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    N = 256

    # Precompute all 4-term arithmetic progressions
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)

    # Efficient precomputation using aps_at index
    aps_at = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            aps_at[pos].append(idx)

    others_for_pos = [None] * N
    for i in range(N):
        if not aps_at[i]:
            others_for_pos[i] = np.array([], dtype=np.int32).reshape(0, 3)
        else:
            others_list = []
            for ap_idx in aps_at[i]:
                row = aps[ap_idx]
                others = [p for p in row if p != i]
                others_list.append(others)
            others_for_pos[i] = np.array(others_list, dtype=np.int32)

    def count_mono(col):
        c = col[aps]
        return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))

    def fast_delta_single(col, i):
        others = others_for_pos[i]
        if others.shape[0] == 0:
            return 0
        others_colors = col[others]
        all_same = (others_colors[:, 0] == others_colors[:, 1]) & (others_colors[:, 1] == others_colors[:, 2])
        if not np.any(all_same):
            return 0
        c_other = others_colors[:, 0]
        c_i = col[i]
        mono_before = c_other == c_i
        mono_after = c_other == (1 - c_i)
        count_before = np.sum(mono_before & all_same)
        count_after = np.sum(mono_after & all_same)
        return int(count_after - count_before)

    def fast_delta_two(col, i, j):
        """Compute delta for flipping both positions i and j simultaneously."""
        if i == j:
            return 0
        others_i = others_for_pos[i]
        others_j = others_for_pos[j]
        
        # Collect all affected AP indices
        affected_aps = set()
        for ap_idx in aps_at[i]:
            affected_aps.add(ap_idx)
        for ap_idx in aps_at[j]:
            affected_aps.add(ap_idx)
        
        if not affected_aps:
            return 0
        
        affected_list = list(affected_aps)
        
        delta = 0
        for ap_idx in affected_list:
            row = aps[ap_idx]
            pos_in_ap = list(row)
            
            # Check if this AP contains i or j
            has_i = i in pos_in_ap
            has_j = j in pos_in_ap
            
            if not (has_i or has_j):
                continue
            
            # Get colors before flip
            before_colors = [col[p] for p in pos_in_ap]
            before_mono = before_colors[0] == before_colors[1] == before_colors[2] == before_colors[3]
            
            # Get colors after flip
            after_colors = []
            for p in pos_in_ap:
                if p == i or p == j:
                    after_colors.append(1 - col[p])
                else:
                    after_colors.append(col[p])
            after_mono = after_colors[0] == after_colors[1] == after_colors[2] == after_colors[3]
            
            if before_mono and not after_mono:
                delta -= 1
            elif not before_mono and after_mono:
                delta += 1
        
        return delta

    def sa_run(seed, max_steps, start_temp, end_temp):
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        col = np_rng.randint(0, 2, size=N).astype(np.int32)
        current_score = count_mono(col)
        best_col = col.copy()
        best_score = current_score

        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break
                
            t = start_temp * (end_temp / start_temp) ** (step / max_steps)

            # 80% single flip, 20% double flip
            if rng.random() < 0.8:
                i = rng.randint(0, N - 1)
                delta = fast_delta_single(col, i)
                if delta <= 0 or (t > 0 and rng.random() < np.exp(-delta / t)):
                    col[i] = 1 - col[i]
                    current_score += delta
            else:
                i = rng.randint(0, N - 1)
                j = rng.randint(0, N - 1)
                if i == j:
                    continue
                delta = fast_delta_two(col, i, j)
                if delta <= 0 or (t > 0 and rng.random() < np.exp(-delta / t)):
                    col[i] = 1 - col[i]
                    col[j] = 1 - col[j]
                    current_score += delta
            
            if current_score < best_score:
                best_score = current_score
                best_col = col.copy()

        return best_col, best_score

    def local_search(col, current_score, max_passes=15):
        """Iterative improvement with single flips and occasional double flips."""
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 5.0:
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                delta = fast_delta_single(col, i)
                if delta < 0:
                    col[i] = 1 - col[i]
                    current_score += delta
                    improved = True
            
            # Try some random double flips to escape local minima
            if time.time() > deadline - 5.0:
                break
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    continue
                delta = fast_delta_two(col, i, j)
                if delta < 0:
                    col[i] = 1 - col[i]
                    col[j] = 1 - col[j]
                    current_score += delta
                    improved = True
        
        return col, current_score

    best_overall = None
    best_overall_score = float('inf')

    # 1. Try structured colorings
    structured_patterns = [
        np.array([(i // 2) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i // 4) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i // 8) % 2 for i in range(N)], dtype=np.int32),
        np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32),
        np.array([(i % 3) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i % 5) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i % 7) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i // 16) % 2 for i in range(N)], dtype=np.int32),
    ]
    
    for pat in structured_patterns:
        s = count_mono(pat)
        if s < best_overall_score:
            best_overall_score = s
            best_overall = pat.copy()

    # 2. SA runs with mixed single/double flips
    num_restarts = 12
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        seed = 100 + r * 137
        max_steps = 60000
        start_temp = 3.0
        end_temp = 0.005

        best_col, best_score = sa_run(seed, max_steps, start_temp, end_temp)
        best_col, best_score = local_search(best_col, best_score, max_passes=15)
        
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall = best_col.copy()

    # 3. Final polish
    if best_overall is not None and time.time() < deadline - 2.0:
        col = best_overall.copy()
        current_score = best_overall_score
        col, current_score = local_search(col, current_score, max_passes=30)
        
        if current_score < best_overall_score:
            best_overall_score = current_score
            best_overall = col

    return [int(c) for c in best_overall]
# EVOLVE-BLOCK-END
