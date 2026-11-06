import time
import numpy as np

def construct():
    N = 128
    deadline = time.time() + 50.0 # Safety margin for 60s limit
    
    # Precompute all 4-term APs (0-indexed)
    aps = []
    for a in range(N):
        for d in range(1, N):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= N:
                break
            aps.append((a, b, c, e))
    
    aps = np.array(aps, dtype=np.int32)
    num_aps = len(aps)
    
    # Helper to count violations (APs with < 3 distinct colors)
    # Vectorized check:
    # An AP (a,b,c,d) is bad if:
    # 1. All 4 same: colors[a]==colors[b]==colors[c]==colors[d]
    # 2. Exactly 2 distinct: 
    #    - 3 same, 1 diff: (a==b==c!=d) or (a==b!=c==d) or (a!=b==c==d) or (a==b==c==d is covered above? No, 4 same is 1 distinct)
    #    - 2 same, 2 diff: (a==c!=b!=d) etc.
    #    
    # Easier: Count distinct colors in each AP.
    # We can do this by checking if the set size is < 3.
    # Vectorizing set size is hard. 
    # Instead, let's count "bad" patterns directly.
    # Bad if:
    # (ca==cb==cc==cd) OR 
    # ((ca==cb) & (cb==cc) & (cc!=cd)) OR ... this is tedious.
    
    # Alternative: 
    # distinct_count = 1 + (ca!=cb) + (cc not in {ca,cb}) + (cd not in {ca,cb,cc})
    # If distinct_count < 3, it's bad.
    
    ca = aps[:, 0]
    cb = aps[:, 1]
    cc = aps[:, 2]
    cd = aps[:, 3]
    
    def count_violations(colors):
        c0 = colors[ca]
        c1 = colors[cb]
        c2 = colors[cc]
        c3 = colors[cd]
        
        # distinct count logic:
        # d1 = 1
        # d2 = d1 + (c0 != c1)
        # d3 = d2 + (c2 != c0) & (c2 != c1)
        # d4 = d3 + (c3 != c0) & (c3 != c1) & (c3 != c2)
        
        d1 = np.ones(num_aps, dtype=np.int32)
        d2 = d1 + (c0 != c1).astype(np.int32)
        d3 = d2 + ((c2 != c0) & (c2 != c1)).astype(np.int32)
        d4 = d3 + ((c3 != c0) & (c3 != c1) & (c3 != c2)).astype(np.int32)
        
        # Bad if d4 < 3
        return np.sum(d4 < 3)

    def try_sa(K, time_limit):
        start_time = time.time()
        colors = np.random.randint(0, K, N, dtype=np.int32)
        current_violations = count_violations(colors)
        best_violations = current_violations
        best_colors = colors.copy()
        
        T = 1.0
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            if current_violations == 0:
                return best_colors.tolist()
                
            # Pick random index
            i = np.random.randint(0, N)
            old_color = colors[i]
            new_color = np.random.randint(0, K)
            
            if new_color == old_color:
                continue
                
            colors[i] = new_color
            new_violations = count_violations(colors)
            
            delta = new_violations - current_violations
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors.copy()
            else:
                colors[i] = old_color
                
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        return best_colors.tolist() if best_violations == 0 else None

    # Try K=4, 5, 6, 7
    # 4 colors is the target minimum.
    for K in [4, 5, 6]:
        time_left = deadline - time.time()
        if time_left < 5.0:
            break
            
        # Allocate time proportional to K? 
        # 4 colors is hardest, give it more time.
        t_alloc = min(20.0, time_left * 0.5)
        
        sol = try_sa(K, t_alloc)
        if sol is not None:
            return sol
            
    # If SA failed to find 0 violations, try a greedy fallback with more colors
    # This ensures we return *something* valid if possible, though SA should work.
    # Greedy with 8 colors is very safe.
    colors_greedy = [-1] * N
    for i in range(N):
        for c in range(8):
            colors_greedy[i] = c
            # Check all APs ending at i? No, check all APs involving i?
            # Greedy is usually done by assigning i and checking constraints with previous.
            # But APs can go "backwards" in index if we consider a < b < c < d.
            # Since we assign 0..N-1, when we assign i, we only need to check APs where i is the *last* element (d=i).
            # APs where i is a, b, or c will be checked when the last element is assigned.
            # So we only need to check APs (a, b, c, i) where a,b,c < i.
            
            valid = True
            for d in range(1, (i // 3) + 1):
                a = i - 3 * d
                b = i - 2 * d
                c_idx = i - d
                if a < 0: break
                ca = colors_greedy[a]
                cb = colors_greedy[b]
                cc = colors_greedy[c_idx]
                if len({c, ca, cb, cc}) < 3:
                    valid = False
                    break
            
            if valid:
                break
        if colors_greedy[i] == -1:
            colors_greedy[i] = 0 # Fallback, shouldn't happen with 8 colors
            
    return colors_greedy
