"""
Max-SAT inspired approach: formulate mono-4AP minimization as weighted clause violation,
use noise-injected local search with dynamic temperature, combined with structured
initialization and multi-phase refinement.
"""
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 112  # safety margin

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
        """Compute change in mono-4AP count if we flip position pos."""
        ap_idx = pos_ap_indices[pos]
        if len(ap_idx) == 0:
            return 0
        sub = color[aps_arr[ap_idx]]
        off = pos_ap_offsets[pos]
        c0 = color[pos]
        other = sub.copy()
        other[np.arange(len(ap_idx)), off] = c0
        curr_mono = np.all(other == c0, axis=1)
        curr_count = int(np.sum(curr_mono))
        new_mono = np.all(other == (1 - c0), axis=1)
        new_count = int(np.sum(new_mono))
        return new_count - curr_count

    def maxsat_local_search(color, score, time_budget=5.0):
        """
        Noise-injected local search with dynamic temperature scaling.
        Treats each mono-4AP as a violated clause. Uses weighted sum with
        random noise to escape local minima, plus greedy descent.
        """
        start = time.time()
        color = color.copy()
        score = int(score)
        
        # Phase 1: Pure greedy descent
        improved = True
        while improved and time.time() - start < time_budget * 0.4:
            improved = False
            for pos in range(n):
                delta = delta_flip(color, pos)
                if delta < 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                    improved = True
        
        # Phase 2: Noise-injected search with dynamic temperature
        if time.time() - start < time_budget:
            rng = np.random.default_rng(int(time.time() * 1000) % 100000)
            T_start = 3.0
            T_end = 0.05
            best_local_score = score
            best_local_color = color.copy()
            
            while time.time() - start < time_budget:
                elapsed_frac = (time.time() - start) / time_budget
                T = T_start * (T_end / T_start) ** min(1.0, elapsed_frac)
                
                # Batch of random flips with noise
                batch_size = 50
                for _ in range(batch_size):
                    if time.time() - start >= time_budget:
                        break
                    pos = rng.integers(0, n)
                    delta = delta_flip(color, pos)
                    
                    # Accept with probability based on delta and temperature
                    if delta <= 0:
                        color[pos] = 1 - color[pos]
                        score += delta
                    else:
                        # Dynamic noise: accept worse moves with prob exp(-delta/T)
                        if rng.random() < np.exp(-delta / max(T, 0.01)):
                            color[pos] = 1 - color[pos]
                            score += delta
                    
                    if score < best_local_score:
                        best_local_score = score
                        best_local_color = color.copy()
                
                # Periodic greedy restart from best
                if time.time() - start < time_budget and score > best_local_score + 5:
                    color = best_local_color.copy()
                    score = best_local_score
                    # Quick greedy pass
                    for pos in range(n):
                        delta = delta_flip(color, pos)
                        if delta < 0:
                            color[pos] = 1 - color[pos]
                            score += delta
        
        # Final greedy polish
        if time.time() < deadline - 2:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for pos in range(n):
                    delta = delta_flip(color, pos)
                    if delta < 0:
                        color[pos] = 1 - color[pos]
                        score += delta
                        improved = True
        
        return color, int(score)

    def generate_candidates():
        """Generate diverse structured and random candidates."""
        candidates = []
        
        # Thue-Morse
        tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(tm.copy())
        
        # Bit-reversal Thue-Morse
        br = np.array([bin(i)[::-1].count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(br.copy())
        
        # Modular patterns
        for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            for q in range(1, p):
                c = np.array([(i % p) < q for i in range(n)], dtype=np.int8)
                candidates.append(c)
        
        # Walsh functions
        for k in range(8):
            c = np.array([(i >> k) & 1 for i in range(n)], dtype=np.int8)
            candidates.append(c)
        
        # XOR combinations (Walsh-Hadamard)
        for mask in range(1, 256, 7):  # sample every 7th to limit count
            c = np.array([bin(i & mask).count('1') % 2 for i in range(n)], dtype=np.int8)
            candidates.append(c)
        
        # Quadratic mod 2
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
        
        # Block patterns
        for block_size in [4, 8, 16, 32]:
            c = np.zeros(n, dtype=np.int8)
            for i in range(0, n, block_size * 2):
                c[i:i+block_size] = 0
                c[i+block_size:i+2*block_size] = 1
            candidates.append(c)
        
        # Random candidates
        rng = np.random.default_rng(99999)
        for _ in range(30):
            c = rng.integers(0, 2, n).astype(np.int8)
            candidates.append(c)
        
        return candidates

    # Generate and score all candidates
    candidates = generate_candidates()
    scored = []
    for c in candidates:
        s = count_mono_4ap(c)
        scored.append((s, c.copy()))
    
    scored.sort(key=lambda x: x[0])
    
    best_score = float('inf')
    best_color = None
    
    # Phase 1: Quick local search on top candidates
    top_k = min(15, len(scored))
    ls_results = []
    for rank in range(top_k):
        if time.time() > deadline - 40:
            break
        score, color = scored[rank]
        if score < best_score:
            best_score = score
            best_color = color.copy()
        
        color_ls, score_ls = maxsat_local_search(color, score, time_budget=3.0)
        if score_ls < best_score:
            best_score = score_ls
            best_color = color_ls.copy()
        ls_results.append((score_ls, color_ls.copy()))
    
    # Phase 2: Deeper search on best results
    ls_results.sort(key=lambda x: x[0])
    deep_starts = min(6, len(ls_results))
    
    for rank in range(deep_starts):
        if time.time() > deadline - 25:
            break
        score, color = ls_results[rank]
        
        time_remaining = deadline - time.time() - 15
        if time_remaining < 3:
            break
        budget = min(15.0, time_remaining * 0.5)
        
        color_deep, score_deep = maxsat_local_search(color, score, time_budget=budget)
        if score_deep < best_score:
            best_score = score_deep
            best_color = color_deep.copy()
    
    # Phase 3: Final polish on overall best
    if best_color is not None and time.time() < deadline - 5:
        color_final, score_final = maxsat_local_search(best_color, best_score, 
                                                        time_budget=deadline - time.time() - 3)
        if score_final < best_score:
            best_score = score_final
            best_color = color_final
    
    if best_color is None:
        best_color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        best_score = count_mono_4ap(best_color)
    
    return [int(c) for c in best_color]
