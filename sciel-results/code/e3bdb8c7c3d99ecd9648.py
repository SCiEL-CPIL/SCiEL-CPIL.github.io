import time
import random
import math

def _distinct4(a, b, c, d):
    cnt = 1
    if b != a:
        cnt += 1
    if c != a and c != b:
        cnt += 1
    if d != a and d != b and d != c:
        cnt += 1
    return cnt

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    n = 128
    # precompute 4-APs (0-indexed)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    m = len(aps)
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    def count_violations(colors):
        viol = 0
        for ap in aps:
            if _distinct4(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]) < 3:
                viol += 1
        return viol

    def seed_greedy():
        colors = []
        for x in range(1, n + 1):
            for color in range(128):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    s = {color, colors[x - 1 - d], colors[x - 1 - 2 * d], colors[x - 1 - 3 * d]}
                    if len(s) < 3:
                        ok = False
                        break
                if ok:
                    colors.append(color)
                    break
            else:
                colors.append(0)
        return colors

    best_colors = seed_greedy()
    best_k = len(set(best_colors))
    
    def is_valid(colors):
        return count_violations(colors) == 0

    current_colors = best_colors[:]
    rng = random.Random(42)
    
    # Iterative Color Merging
    while time.time() < deadline - 5.0:
        used_colors = list(set(current_colors))
        if len(used_colors) <= 3:
            break
            
        # Shuffle to avoid bias
        rng.shuffle(used_colors)
        
        merged_this_iter = False
        for i in range(len(used_colors)):
            for j in range(i + 1, len(used_colors)):
                c1 = used_colors[i]
                c2 = used_colors[j]
                
                # Try merging c2 into c1
                new_colors = current_colors[:]
                for idx in range(n):
                    if new_colors[idx] == c2:
                        new_colors[idx] = c1
                        
                if is_valid(new_colors):
                    current_colors = new_colors
                    best_colors = current_colors[:]
                    best_k -= 1
                    merged_this_iter = True
                    break
            if merged_this_iter:
                break
                
        if not merged_this_iter:
            # No merge worked. Try a perturbation to escape local optima?
            # Or just stop if we've tried many times.
            # For now, we can try a random swap or change to see if it unlocks a merge.
            # But simple merging is often sufficient for this size.
            # Let's try a few random perturbations if we are stuck.
            if rng.random() < 0.1:
                # Perturb
                pos = rng.randrange(n)
                old_col = current_colors[pos]
                new_col = rng.randrange(128)
                current_colors[pos] = new_col
                # Check if this perturbation allows a merge or is valid
                if is_valid(current_colors):
                    best_colors = current_colors[:]
                    best_k = len(set(best_colors))
                else:
                    # Revert if invalid
                    current_colors[pos] = old_col

    if not is_valid(best_colors):
        best_colors = seed_greedy()
        
    return best_colors
