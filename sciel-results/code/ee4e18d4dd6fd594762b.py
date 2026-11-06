import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 128
    
    # Precompute all 4-APs
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # For each position, which APs contain it
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    def compute_ap_distinct(colors):
        """Compute distinct color count for each AP."""
        return [len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) for ap in aps]
    
    def count_violations(ap_distinct):
        return sum(1 for d in ap_distinct if d < 3)
    
    def smart_init(K):
        """Generate a smart initial coloring with K colors."""
        candidates = []
        # i % K
        candidates.append([i % K for i in range(n)])
        # (i // 2) % K
        candidates.append([(i // 2) % K for i in range(n)])
        # (i // 4) % K
        candidates.append([(i // 4) % K for i in range(n)])
        # (i % 8) // 2
        candidates.append([(i % 8) // 2 for i in range(n)])
        # Random
        candidates.append([random.randint(0, K - 1) for _ in range(n)])
        
        best = None
        best_v = float('inf')
        for c in candidates:
            ad = compute_ap_distinct(c)
            v = count_violations(ad)
            if v < best_v:
                best_v = v
                best = c[:]
        return best, best_v
    
    best_colors = None
    best_K = float('inf')
    
    for K in range(3, 7):
        if time.time() > deadline:
            break
        
        found = False
        for r in range(25):
            if time.time() > deadline:
                break
            
            colors, v = smart_init(K)
            ap_distinct = compute_ap_distinct(colors)
            
            if v == 0:
                best_colors = colors
                best_K = K
                found = True
                break
            
            # Simulated annealing
            temp = 1.0
            cooling = 0.997
            best_local_v = v
            best_local_colors = colors[:]
            no_improve = 0
            
            for step in range(8000):
                if time.time() > deadline:
                    break
                
                if v == 0:
                    best_colors = colors
                    best_K = K
                    found = True
                    break
                
                # Get violated APs
                violated = [i for i, d in enumerate(ap_distinct) if d < 3]
                if not violated:
                    break
                
                ap_idx = random.choice(violated)
                ap = aps[ap_idx]
                pos = random.choice(ap)
                old_c = colors[pos]
                
                new_c = random.randint(0, K - 1)
                if new_c == old_c:
                    new_c = (new_c + 1) % K
                
                # Compute delta
                delta = 0
                for ap_i in pos_aps[pos]:
                    old_d = ap_distinct[ap_i]
                    ap = aps[ap_i]
                    # Compute new distinct without rebuilding set each time
                    # Just compute directly
                    new_colors = [colors[ap[0]] if j == 0 else colors[ap[1]] if j == 1 else colors[ap[2]] if j == 2 else colors[ap[3]] for j in range(4)]
                    # Simpler: build set
                    s_old = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                    s_new = set(s_old)
                    s_new.discard(old_c)
                    s_new.add(new_c)
                    new_d = len(s_new)
                    
                    was_viol = old_d < 3
                    is_viol = new_d < 3
                    if was_viol and not is_viol:
                        delta -= 1
                    elif not was_viol and is_viol:
                        delta += 1
                
                # Accept?
                if delta <= 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    colors[pos] = new_c
                    for ap_i in pos_aps[pos]:
                        ap = aps[ap_i]
                        ap_distinct[ap_i] = len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]})
                    v += delta
                    if v < best_local_v:
                        best_local_v = v
                        best_local_colors = colors[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
                temp *= cooling
                
                if no_improve > 300:
                    colors = best_local_colors[:]
                    v = best_local_v
                    ap_distinct = compute_ap_distinct(colors)
                    temp = 0.5
                    no_improve = 0
            
            if found:
                break
            if v == 0:
                best_colors = colors
                best_K = K
                found = True
                break
        
        if found:
            break
    
    if best_colors is None:
        return list(range(n))
    
    return best_colors
