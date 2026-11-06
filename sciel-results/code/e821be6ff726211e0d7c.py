"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    
    # Precompute all non-constant 4-APs and which positions they involve
    aps = []
    pos_to_aps = [[] for _ in range(n)]
    for a in range(1, n+1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            idx = [a-1, a+d-1, a+2*d-1, a+3*d-1]
            aps.append(idx)
            for pos in idx:
                pos_to_aps[pos].append(len(aps)-1)
    
    def count_violations(colors):
        violations = 0
        for idx in aps:
            c = {colors[idx[0]], colors[idx[1]], colors[idx[2]], colors[idx[3]]}
            if len(c) < 3:
                violations += 1
        return violations
    
    def try_local_search(k, max_iters=10000, seed=0):
        rng = random.Random(seed)
        colors = [rng.randint(0, k-1) for _ in range(n)]
        cur_viol = count_violations(colors)
        if cur_viol == 0:
            return colors
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            pos = rng.randint(0, n-1)
            old_color = colors[pos]
            new_color = rng.randint(0, k-1)
            if new_color == old_color:
                continue
            
            # Compute delta in violations
            delta = 0
            for ap_idx in pos_to_aps[pos]:
                idx = aps[ap_idx]
                old_set = {colors[idx[0]], colors[idx[1]], colors[idx[2]], colors[idx[3]]}
                old_bad = 1 if len(old_set) < 3 else 0
                
                # Temporarily change
                pos_in_ap = idx.index(pos)
                colors[idx[pos_in_ap]] = new_color
                new_set = {colors[idx[0]], colors[idx[1]], colors[idx[2]], colors[idx[3]]}
                new_bad = 1 if len(new_set) < 3 else 0
                colors[idx[pos_in_ap]] = old_color  # Restore
                
                delta += new_bad - old_bad
            
            if delta <= 0:
                colors[pos] = new_color
                cur_viol += delta
                if cur_viol == 0:
                    return colors
            # Else: don't flip (greedy)
        
        return None
    
    # Try increasing number of colors
    for k in [3, 4, 5]:
        for seed in range(10):
            if time.time() > deadline:
                break
            result = try_local_search(k, max_iters=5000, seed=seed)
            if result is not None:
                if count_violations(result) == 0:
                    return result
        if time.time() > deadline:
            break
    
    # Fallback to seed's greedy approach
    colors = []
    for x in range(1, n+1):
        for color in range(128):
            ok = True
            # Check all 4-APs that end at or before x and include x
            for d in range(1, (x-1)//3 + 1):
                if x-3*d >= 1:
                    prog_colors = {colors[x-1-j*d] for j in (0,1,2)}
                    if len({color} | prog_colors) < 3:
                        ok = False
                        break
            if ok:
                colors.append(color)
                break
        else:
            colors.append(0)
    
    return colors
# EVOLVE-BLOCK-END
