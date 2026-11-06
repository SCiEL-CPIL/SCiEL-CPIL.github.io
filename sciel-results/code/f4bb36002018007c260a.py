import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 128
    
    # Precompute all 4-APs as numpy array
    ap_list = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 >= n:
                break
            ap_list.append((a, a+d, a+2*d, a+3*d))
    
    aps = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)
    
    random.seed(42)
    
    best_colors = None
    best_num_colors = 128
    
    for num_colors in range(3, 12):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            colors = np.array([random.randint(0, num_colors-1) for _ in range(n)], dtype=np.int32)
            
            def count_bad():
                # For each AP, check if <3 distinct colors
                c0 = colors[aps[:,0]]
                c1 = colors[aps[:,1]]
                c2 = colors[aps[:,2]]
                c3 = colors[aps[:,3]]
                # Count distinct: use pairwise comparisons
                # <3 distinct means at least 3 pairs are equal (out of 6 pairs)
                # Actually, simpler: check if all same, or exactly 2 colors
                all_same = (c0==c1) & (c1==c2) & (c2==c3)
                # For 2 colors: check various patterns
                # Pattern: 3 same, 1 different
                p1 = (c0==c1) & (c1==c2) & (c2!=c3)
                p2 = (c0==c1) & (c1!=c2) & (c2==c3)
                p3 = (c0!=c1) & (c1==c2) & (c2==c3)
                p4 = (c0==c2) & (c2==c3) & (c0!=c1)
                # Pattern: 2 and 2
                p5 = (c0==c1) & (c2==c3) & (c0!=c2)
                p6 = (c0==c2) & (c1==c3) & (c0!=c1)
                p7 = (c0==c3) & (c1==c2) & (c0!=c1)
                two_colors = p1|p2|p3|p4|p5|p6|p7
                bad = all_same | two_colors
                return np.sum(bad)
            
            current_bad = count_bad()
            
            for iteration in range(3000):
                if time.time() > deadline:
                    break
                
                if current_bad == 0:
                    found = True
                    break
                
                # Find bad APs
                c0 = colors[aps[:,0]]
                c1 = colors[aps[:,1]]
                c2 = colors[aps[:,2]]
                c3 = colors[aps[:,3]]
                all_same = (c0==c1) & (c1==c2) & (c2==c3)
                p1 = (c0==c1) & (c1==c2) & (c2!=c3)
                p2 = (c0==c1) & (c1!=c2) & (c2==c3)
                p3 = (c0!=c1) & (c1==c2) & (c2==c3)
                p4 = (c0==c2) & (c2==c3) & (c0!=c1)
                p5 = (c0==c1) & (c2==c3) & (c0!=c2)
                p6 = (c0==c2) & (c1==c3) & (c0!=c1)
                p7 = (c0==c3) & (c1==c2) & (c0!=c1)
                bad_mask = all_same | p1|p2|p3|p4|p5|p6|p7
                bad_indices = np.where(bad_mask)[0]
                
                if len(bad_indices) == 0:
                    found = True
                    break
                
                bad_ap_idx = bad_indices[random.randint(0, len(bad_indices)-1)]
                ap = aps[bad_ap_idx]
                pos = int(ap[random.randint(0,3)])
                old = colors[pos]
                
                # Try all colors
                best_new = old
                best_bad = current_bad
                for nc in range(num_colors):
                    if nc == old:
                        continue
                    colors[pos] = nc
                    nb = count_bad()
                    if nb < best_bad:
                        best_bad = nb
                        best_new = nc
                    colors[pos] = old
                
                if best_new != old:
                    colors[pos] = best_new
                    current_bad = best_bad
                else:
                    # No improvement, break
                    break
            
            if found:
                used = set(colors.tolist())
                if len(used) < best_num_colors:
                    best_num_colors = len(used)
                    best_colors = colors.tolist()
                break
    
    if best_colors is None:
        best_colors = [x % 43 for x in range(1, n+1)]
    
    return best_colors
