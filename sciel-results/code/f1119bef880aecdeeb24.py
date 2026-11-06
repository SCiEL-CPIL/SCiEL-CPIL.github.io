import random
import time

def construct():
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # For each index, store the indices of APs that contain it
    ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices[pos].append(idx)

    def is_violated_ap(ap, colors):
        return len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3

    def local_search_v2(K, max_steps=200000, time_limit=5.0):
        start_time = time.time()
        colors = [random.randint(0, K - 1) for _ in range(N)]
        
        # Initialize violated_aps set
        violated_aps = set()
        for idx, ap in enumerate(aps):
            if is_violated_ap(ap, colors):
                violated_aps.add(idx)
                
        if not violated_aps:
            return colors

        best_violations = len(violated_aps)
        best_colors = colors[:]

        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
                
            if not violated_aps:
                return colors

            # Pick a random index i
            i = random.randint(0, N - 1)
            if not ap_indices[i]:
                continue
                
            # Pick a random AP containing i
            ap_idx = random.choice(ap_indices[i])
            
            # Check if this AP is currently violated
            if ap_idx not in violated_aps:
                continue

            ap = aps[ap_idx]
            old_color = colors[i]
            
            # Try to find a better color for i
            # Calculate V_old: number of violated APs containing i
            V_old = sum(1 for ap_idx_i in ap_indices[i] if ap_idx_i in violated_aps)
            
            best_delta = 0
            best_c = None
            
            for c in range(K):
                if c == old_color:
                    continue
                
                # Calculate V_new(c)
                # Temporarily set color
                colors[i] = c
                V_new = 0
                for ap_idx_i in ap_indices[i]:
                    if is_violated_ap(aps[ap_idx_i], colors):
                        V_new += 1
                colors[i] = old_color # Revert
                
                delta = V_new - V_old
                if delta < best_delta:
                    best_delta = delta
                    best_c = c

            if best_c is None:
                continue
                
            # Apply the best change
            colors[i] = best_c
            
            # Update global violated_aps set
            for ap_idx_i in ap_indices[i]:
                was_violated = ap_idx_i in violated_aps
                is_violated_now = is_violated_ap(aps[ap_idx_i], colors)
                
                if was_violated and not is_violated_now:
                    violated_aps.remove(ap_idx_i)
                elif not was_violated and is_violated_now:
                    violated_aps.add(ap_idx_i)
            
            if len(violated_aps) < best_violations:
                best_violations = len(violated_aps)
                best_colors = colors[:]
                
            if not violated_aps:
                return colors

        return None

    # Try K=3, 4, 5...
    for K in range(3, 13):
        time_budget = 10.0 if K == 3 else 5.0
        res = local_search_v2(K, max_steps=200000, time_limit=time_budget)
        if res is not None:
            return res
            
    # Fallback: Deterministic greedy
    aps_end = [[] for _ in range(N)]
    for i in range(N):
        max_d = i // 3
        for d in range(1, max_d + 1):
            a = i - 3 * d
            b = i - 2 * d
            c = i - d
            aps_end[i].append((a, b, c))

    colors = []
    for x in range(1, 129):
        found = False
        for c in range(128):
            valid = True
            idx = x - 1
            for (a, b, cc) in aps_end[idx]:
                if len({colors[a], colors[b], colors[cc], c}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(c)
                found = True
                break
        if not found:
            colors.append(0)
            
    return colors
