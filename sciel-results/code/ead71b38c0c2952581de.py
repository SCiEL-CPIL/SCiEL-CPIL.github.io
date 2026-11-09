"""Colorings with three colors in every four-term progression."""
import time
import random
import numpy as np

def construct():
    n = 128
    deadline = time.time() + 85
    
    # Precompute all 4-APs
    aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a+d, a+2*d, a+3*d))
    
    num_aps = len(aps)
    
    # Precompute AP index arrays for fast numpy operations
    a0_arr = np.array([ap[0] for ap in aps], dtype=np.int32)
    a1_arr = np.array([ap[1] for ap in aps], dtype=np.int32)
    a2_arr = np.array([ap[2] for ap in aps], dtype=np.int32)
    a3_arr = np.array([ap[3] for ap in aps], dtype=np.int32)
    
    # For each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a].append(idx)
        pos_to_aps[b].append(idx)
        pos_to_aps[c].append(idx)
        pos_to_aps[d].append(idx)
    
    def count_violations_fast(colors_np):
        c0 = colors_np[a0_arr]
        c1 = colors_np[a1_arr]
        c2 = colors_np[a2_arr]
        c3 = colors_np[a3_arr]
        
        all_same = (c0 == c1) & (c0 == c2) & (c0 == c3)
        three_same = ((c0 == c1) & (c0 == c2) & (c0 != c3)) | \
                     ((c0 == c1) & (c0 == c3) & (c0 != c2)) | \
                     ((c0 == c2) & (c0 == c3) & (c0 != c1)) | \
                     ((c1 == c2) & (c1 == c3) & (c1 != c0))
        two_two = ((c0 == c1) & (c2 == c3) & (c0 != c2)) | \
                  ((c0 == c2) & (c1 == c3) & (c0 != c1)) | \
                  ((c0 == c3) & (c1 == c2) & (c0 != c1))
        
        violations = all_same | three_same | two_two
        return int(np.sum(violations))
    
    def compute_ap_violation(colors_np, ap_idx):
        a, b, c, d = aps[ap_idx]
        c0 = colors_np[a]
        c1 = colors_np[b]
        c2 = colors_np[c]
        c3 = colors_np[d]
        if c0 == c1 == c2 == c3:
            return True
        if (c0 == c1 == c2 and c0 != c3) or (c0 == c1 == c3 and c0 != c2) or \
           (c0 == c2 == c3 and c0 != c1) or (c1 == c2 == c3 and c1 != c0):
            return True
        if (c0 == c1 and c2 == c3 and c0 != c2) or \
           (c0 == c2 and c1 == c3 and c0 != c1) or \
           (c0 == c3 and c1 == c2 and c0 != c1):
            return True
        return False
    
    best_colors = None
    best_violations = float('inf')
    
    # Try 3-color solutions first
    for restart in range(200):
        if time.time() > deadline:
            break
        
        if restart == 0:
            colors_np = np.array([i % 3 for i in range(n)])
        elif restart == 1:
            colors_np = np.array([(i // 2) % 3 for i in range(n)])
        elif restart == 2:
            colors_np = np.array([(i // 3) % 3 for i in range(n)])
        elif restart == 3:
            colors_np = np.array([(i * 2) % 3 for i in range(n)])
        elif restart == 4:
            colors_np = np.array([(i * 3 + i * i) % 3 for i in range(n)])
        elif restart == 5:
            colors_np = np.array([(i * 5) % 3 for i in range(n)])
        elif restart == 6:
            colors_np = np.array([(i * 7) % 3 for i in range(n)])
        else:
            colors_np = np.random.randint(0, 3, size=n)
        
        cur_viol = count_violations_fast(colors_np)
        
        if cur_viol == 0:
            best_colors = colors_np.copy()
            best_violations = 0
            break
        
        # Local search with simulated annealing
        T = 30.0
        T_min = 0.01
        cooling = 0.997
        
        for step in range(30000):
            if time.time() > deadline:
                break
            
            pos = random.randint(0, n - 1)
            old_color = colors_np[pos]
            new_color = random.randint(0, 2)
            if new_color == old_color:
                continue
            
            affected_aps = pos_to_aps[pos]
            delta = 0
            for ap_idx in affected_aps:
                was_viol = compute_ap_violation(colors_np, ap_idx)
                colors_np[pos] = new_color
                is_viol = compute_ap_violation(colors_np, ap_idx)
                colors_np[pos] = old_color
                if is_viol and not was_viol:
                    delta += 1
                elif was_viol and not is_viol:
                    delta -= 1
            
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                colors_np[pos] = new_color
                cur_viol += delta
            
            if cur_viol == 0:
                best_colors = colors_np.copy()
                best_violations = 0
                break
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if cur_viol < best_violations:
            best_violations = cur_viol
            best_colors = colors_np.copy()
        
        if best_violations == 0:
            break
    
    if best_colors is not None and best_violations == 0:
        assert count_violations_fast(best_colors) == 0
        return list(best_colors)
    
    # Try 4-color solutions
    for restart in range(300):
        if time.time() > deadline:
            break
        
        if restart == 0:
            colors_np = np.array([i % 4 for i in range(n)])
        elif restart == 1:
            colors_np = np.array([(i // 2) % 4 for i in range(n)])
        elif restart == 2:
            colors_np = np.array([(i // 4) % 4 for i in range(n)])
        elif restart == 3:
            colors_np = np.array([(i * 2) % 4 for i in range(n)])
        elif restart == 4:
            colors_np = np.array([(i * 3) % 4 for i in range(n)])
        else:
            colors_np = np.random.randint(0, 4, size=n)
        
        cur_viol = count_violations_fast(colors_np)
        
        if cur_viol == 0:
            if best_colors is None or cur_viol < best_violations:
                best_colors = colors_np.copy()
                best_violations = 0
            break
        
        T = 30.0
        T_min = 0.01
        cooling = 0.997
        
        for step in range(30000):
            if time.time() > deadline:
                break
            
            pos = random.randint(0, n - 1)
            old_color = colors_np[pos]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
            
            affected_aps = pos_to_aps[pos]
            delta = 0
            for ap_idx in affected_aps:
                was_viol = compute_ap_violation(colors_np, ap_idx)
                colors_np[pos] = new_color
                is_viol = compute_ap_violation(colors_np, ap_idx)
                colors_np[pos] = old_color
                if is_viol and not was_viol:
                    delta += 1
                elif was_viol and not is_viol:
                    delta -= 1
            
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                colors_np[pos] = new_color
                cur_viol += delta
            
            if cur_viol == 0:
                if best_colors is None or cur_viol < best_violations:
                    best_colors = colors_np.copy()
                    best_violations = 0
                break
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if cur_viol < best_violations:
            best_violations = cur_viol
            best_colors = colors_np.copy()
        
        if best_violations == 0:
            break
    
    if best_colors is not None and best_violations == 0:
        assert count_violations_fast(best_colors) == 0
        return list(best_colors)
    
    # Fallback: return best found even if not perfect (should not happen)
    if best_colors is not None:
        return list(best_colors)
    
    return list(range(n))
