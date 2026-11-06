import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 128
    
    # Precompute all 4-APs
    ap_list = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 >= n:
                break
            ap_list.append((a, a+d, a+2*d, a+3*d))
    
    num_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)
    
    # For each position, precompute which APs it belongs to and its index in each
    pos_to_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        for j in range(4):
            pos = int(aps[idx, j])
            pos_to_aps[pos].append((idx, j))
    
    random.seed(42)
    
    best_colors = None
    best_num_colors = n
    
    for num_colors in range(3, 12):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Random initial coloring
            colors = np.array([random.randint(0, num_colors-1) for _ in range(n)], dtype=np.int32)
            
            # Compute initial violation count
            c0 = colors[aps[:,0]]
            c1 = colors[aps[:,1]]
            c2 = colors[aps[:,2]]
            c3 = colors[aps[:,3]]
            
            def is_bad(c0, c1, c2, c3):
                all_same = (c0==c1) & (c1==c2) & (c2==c3)
                p1 = (c0==c1) & (c1==c2) & (c2!=c3)
                p2 = (c0==c1) & (c1!=c2) & (c2==c3)
                p3 = (c0!=c1) & (c1==c2) & (c2==c3)
                p4 = (c0==c2) & (c2==c3) & (c0!=c1)
                p5 = (c0==c1) & (c2==c3) & (c0!=c2)
                p6 = (c0==c2) & (c1==c3) & (c0!=c1)
                p7 = (c0==c3) & (c1==c2) & (c0!=c1)
                return all_same | p1|p2|p3|p4|p5|p6|p7
            
            bad_mask = is_bad(c0, c1, c2, c3)
            current_bad = int(np.sum(bad_mask))
            
            # Simulated annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.9995
            max_iter = 50000
            
            for iteration in range(max_iter):
                if time.time() > deadline:
                    break
                
                if current_bad == 0:
                    found = True
                    break
                
                # Pick random position
                pos = random.randint(0, n-1)
                old = int(colors[pos])
                
                if old == 0 and num_colors > 1:
                    new = random.randint(1, num_colors-1)
                elif old == num_colors-1 and num_colors > 1:
                    new = random.randint(0, num_colors-2)
                else:
                    new = random.randint(0, num_colors-1)
                    if new == old:
                        new = (new + 1) % num_colors
                
                if new == old:
                    continue
                
                # Compute delta in violations for affected APs
                affected = pos_to_aps[pos]
                delta = 0
                for (ap_idx, j) in affected:
                    old_c0 = int(colors[aps[ap_idx,0]])
                    old_c1 = int(colors[aps[ap_idx,1]])
                    old_c2 = int(colors[aps[ap_idx,2]])
                    old_c3 = int(colors[aps[ap_idx,3]])
                    old_bad = 1 if is_bad(np.array([old_c0]), np.array([old_c1]), np.array([old_c2]), np.array([old_c3]))[0] else 0
                    
                    # Update position j with new color
                    new_c0 = old_c0 if j != 0 else new
                    new_c1 = old_c1 if j != 1 else new
                    new_c2 = old_c2 if j != 2 else new
                    new_c3 = old_c3 if j != 3 else new
                    new_bad = 1 if is_bad(np.array([new_c0]), np.array([new_c1]), np.array([new_c2]), np.array([new_c3]))[0] else 0
                    
                    delta += new_bad - old_bad
                
                # Accept or reject
                if delta <= 0:
                    colors[pos] = new
                    current_bad += delta
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[pos] = new
                        current_bad += delta
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            if found:
                # Try to merge colors to reduce count
                merged = colors.copy()
                used_colors = set(merged.tolist())
                
                # Try merging pairs of colors
                improved = True
                while improved:
                    improved = False
                    color_list = sorted(used_colors)
                    for i in range(len(color_list)):
                        for j in range(i+1, len(color_list)):
                            if time.time() > deadline:
                                break
                            ci, cj = color_list[i], color_list[j]
                            # Try merging cj into ci
                            test = merged.copy()
                            test[test == cj] = ci
                            
                            c0 = test[aps[:,0]]
                            c1 = test[aps[:,1]]
                            c2 = test[aps[:,2]]
                            c3 = test[aps[:,3]]
                            bad = is_bad(c0, c1, c2, c3)
                            if not np.any(bad):
                                merged = test
                                used_colors.discard(cj)
                                improved = True
                                break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                
                final_colors = merged.tolist()
                final_num = len(set(final_colors))
                if final_num < best_num_colors:
                    best_num_colors = final_num
                    best_colors = final_colors
                break
    
    if best_colors is None:
        # Fallback: use modular coloring
        best_colors = [x % 43 for x in range(n)]
    
    return best_colors
