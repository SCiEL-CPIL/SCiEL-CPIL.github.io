import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 256
    
    # Precompute APs
    ap_list = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
            
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Precompute for each index, the list of AP indices containing it
    aps_at = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            aps_at[pos].append(idx)
            
    # Precompute "others" for each index in each AP it belongs to
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
    
    # Precompute same-color count matrix:
    # For each position i and each of its APs, store the 3 other positions.
    # We'll compute delta by checking if all 3 others have the same color,
    # and if so, whether that color matches col[i].
    # To speed up, precompute for each i: the flat array of other positions.
    others_flat = [None] * N
    others_ap_starts = [None] * N  # start index in others_flat for each AP
    others_ap_counts = [None] * N  # count of APs for each position
    for i in range(N):
        others = others_for_pos[i]
        others_flat[i] = others.flatten()
        others_ap_starts[i] = np.arange(0, others.shape[0] * 3, 3)
        others_ap_counts[i] = others.shape[0]

    def count_mono(col):
        c = col[aps]
        return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))

    def fast_delta(col, i):
        n_aps_i = others_ap_counts[i]
        if n_aps_i == 0:
            return 0
        flat = others_flat[i]
        oc = col[flat]  # shape (3 * n_aps_i,)
        c_i = col[i]
        # Reshape to (n_aps_i, 3)
        oc2d = oc.reshape(n_aps_i, 3)
        # Check if all 3 are same
        all_same = (oc2d[:, 0] == oc2d[:, 1]) & (oc2d[:, 1] == oc2d[:, 2])
        if not np.any(all_same):
            return 0
        # For those where all same, check if they match c_i
        same_color = oc2d[:, 0]
        # delta = (num where same_color != c_i) - (num where same_color == c_i) among all_same
        mask = all_same
        n_total = int(np.sum(mask))
        n_match = int(np.sum(same_color[mask] == c_i))
        n_mismatch = n_total - n_match
        return n_mismatch - n_match

    def local_search(col, max_time, seed):
        rng = random.Random(seed)
        current_score = count_mono(col)
        best_score = current_score
        best_col = col.copy()
        
        order = list(range(N))
        rng.shuffle(order)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in order:
                if time.time() > deadline - 2.0:
                    break
                delta = fast_delta(col, i)
                if delta < 0:
                    col[i] = 1 - col[i]
                    current_score += delta
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_col = col.copy()
        return best_col, best_score

    def sa_run(col_init, seed, max_steps, start_temp, end_temp):
        rng = random.Random(seed)
        
        col = col_init.copy()
        current_score = count_mono(col)
        best_score = current_score
        best_col = col.copy()
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            t = start_temp * (end_temp / start_temp) ** (step / max_steps)
            if t < 0.001:
                t = 0.001
            
            i = rng.randint(0, N - 1)
            delta = fast_delta(col, i)
            
            if delta <= 0:
                col[i] = 1 - col[i]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_col = col.copy()
            else:
                if t > 0 and rng.random() < np.exp(-delta / t):
                    col[i] = 1 - col[i]
                    current_score += delta
        
        return best_col, best_score

    best_overall = None
    best_overall_score = float('inf')
    
    # 1. Structured starts - include more diverse patterns
    structured = [
        np.array([ (i // 2) % 2 for i in range(N) ], dtype=np.int32),
        np.array([ (i // 4) % 2 for i in range(N) ], dtype=np.int32),
        np.array([ bin(i).count('1') % 2 for i in range(N) ], dtype=np.int32),  # Thue-Morse
        np.array([ (i % 3) % 2 for i in range(N) ], dtype=np.int32),
        np.array([ (i % 5) % 2 for i in range(N) ], dtype=np.int32),
        # 4-color Thue-Morse mapped to binary
        np.array([ bin(i).count('1') % 4 for i in range(N) ], dtype=np.int32) % 2,
        # Block pattern: 0011 repeated
        np.array([ ((i // 2) % 2) for i in range(N) ], dtype=np.int32),
        # Block pattern: 000111 repeated
        np.array([ ((i // 3) % 2) for i in range(N) ], dtype=np.int32),
        np.random.RandomState(42).randint(0, 2, size=N).astype(np.int32),
    ]
    
    for pat in structured:
        s = count_mono(pat)
        if s < best_overall_score:
            best_overall_score = s
            best_overall = pat.copy()
            
    # 2. SA + Local Search with more restarts
    # Run 1: SA on best structured
    if time.time() < deadline - 20.0:
        b_col, b_score = sa_run(best_overall, 123, 50000, 2.0, 0.01)
        b_col, b_score = local_search(b_col, deadline, 123)
        if b_score < best_overall_score:
            best_overall_score = b_score
            best_overall = b_col
    
    # Random restarts with SA + Local Search
    for r in range(2, 15):
        if time.time() > deadline - 10.0:
            break
        seed = 100 + r * 17
        init_col = np.random.RandomState(seed).randint(0, 2, size=N).astype(np.int32)
        b_col, b_score = sa_run(init_col, seed, 20000, 1.5, 0.005)
        b_col, b_score = local_search(b_col, deadline, seed)
        
        if b_score < best_overall_score:
            best_overall_score = b_score
            best_overall = b_col.copy()
            
    # Final Polish
    if best_overall is not None and time.time() < deadline - 2.0:
        b_col, b_score = local_search(best_overall, deadline, 999)
        if b_score < best_overall_score:
            best_overall_score = b_score
            best_overall = b_col

    return [int(c) for c in best_overall]
