import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute all 4-term APs
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        bad = 0
        for ap in ap_list:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                bad += 1
        return bad
    
    def fast_count_violations(colors, changed_pos, old_color, new_color, prev_bad):
        """Update violation count after changing one position."""
        # Reset violations for APs containing changed_pos
        # This is complex, so let's just recount affected APs
        # For simplicity, just recount all (fast enough for 2600 APs)
        bad = 0
        for ap in ap_list:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                bad += 1
        return bad
    
    best_colors = None
    best_bad = float('inf')
    
    # Try structured initial patterns
    patterns = []
    # Pattern 1: i % 3
    patterns.append(np.array([i % 3 for i in range(n)]))
    # Pattern 2: (i // 4) % 3
    patterns.append(np.array([(i // 4) % 3 for i in range(n)]))
    # Pattern 3: (i % 4) % 3
    patterns.append(np.array([(i % 4) % 3 for i in range(n)]))
    # Pattern 4: (i % 3) * (1 + (i // 3) % 2) % 3
    patterns.append(np.array([(i % 3) * (1 + (i // 3) % 2) % 3 for i in range(n)]))
    # Pattern 5: (i % 2) * 2 + ((i // 2) % 3) % 2
    patterns.append(np.array([(i % 2) * 2 + ((i // 2) % 3) % 2 for i in range(n)]))
    # Pattern 6: (i % 5) % 3
    patterns.append(np.array([(i % 5) % 3 for i in range(n)]))
    
    for pat in patterns:
        bad = count_violations(pat)
        if bad < best_bad:
            best_bad = bad
            best_colors = pat.copy()
            if bad == 0:
                return best_colors.tolist()
    
    # Local search from best pattern and random restarts
    for restart in range(30):
        if time.time() > deadline:
            break
        
        if restart == 0:
            colors = best_colors.copy()
        else:
            colors = np.random.randint(0, 3, n)
        
        current_bad = count_violations(colors)
        
        if current_bad < best_bad:
            best_bad = current_bad
            best_colors = colors.copy()
            if current_bad == 0:
                return best_colors.tolist()
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() < deadline and T > min_T:
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                pos = np.random.randint(0, n)
                old_color = colors[pos]
                new_color = np.random.randint(0, 3)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                new_bad = count_violations(colors)
                
                delta = new_bad - current_bad
                
                if delta <= 0 or np.random.rand() < np.exp(-delta / T):
                    if new_bad < best_bad:
                        best_bad = new_bad
                        best_colors = colors.copy()
                        if new_bad == 0:
                            return best_colors.tolist()
                    current_bad = new_bad
                else:
                    colors[pos] = old_color
            
            T *= cooling
            
            if current_bad == 0:
                break
    
    if best_colors is not None and best_bad == 0:
        return best_colors.tolist()
    
    # If we haven't found a perfect solution, return best found
    if best_colors is not None:
        return best_colors.tolist()
    
    # Fallback
    return [0] * n
