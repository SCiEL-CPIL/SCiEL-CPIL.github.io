import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    n = 128
    
    # Precompute all nonconstant 4-APs: (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= n
    aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    m = len(aps)
    
    # For each position, store which APs it participates in
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    
    def count_violations(colors):
        """Count number of 4-APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            s = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(s) < 3:
                v += 1
        return v
    
    def is_valid(colors):
        """Check if coloring satisfies all constraints."""
        for ap in aps:
            s = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(s) < 3:
                return False
        return True
    
    def try_local_search(k, budget, seed=0):
        """Local search to find valid k-coloring."""
        rng = random.Random(seed)
        end_time = start_time + budget
        
        best_colors = None
        best_violations = m + 1
        
        # Try multiple random initializations
        while time.time() < end_time:
            # Initialize
            if rng.random() < 0.3:
                colors = [i % k for i in range(n)]
            elif rng.random() < 0.5:
                colors = [rng.randrange(k) for _ in range(n)]
            else:
                # Start with a structured pattern
                colors = [i % k for i in range(n)]
                for _ in range(rng.randint(5, 20)):
                    pos = rng.randrange(n)
                    colors[pos] = rng.randrange(k)
            
            violations = count_violations(colors)
            if violations == 0:
                return colors
            
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
            
            # Local search: flip colors to reduce violations
            stall = 0
            while time.time() < end_time and violations > 0:
                # Pick a random violating AP
                violating_aps = [ap for ap in aps if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3]
                if not violating_aps:
                    break
                ap = rng.choice(violating_aps)
                
                # Try flipping one of the positions in this AP to a different color
                best_move = None
                best_delta = 0
                for p in ap:
                    old_c = colors[p]
                    for new_c in range(k):
                        if new_c == old_c:
                            continue
                        # Estimate delta: how many APs containing p change status
                        delta = 0
                        for idx in pos_aps[p]:
                            other_ap = aps[idx]
                            # Old status
                            old_set = {colors[other_ap[0]], colors[other_ap[1]], colors[other_ap[2]], colors[other_ap[3]]}
                            old_bad = len(old_set) < 3
                            # New status (only p changes)
                            new_set = set()
                            for q in other_ap:
                                if q == p:
                                    new_set.add(new_c)
                                else:
                                    new_set.add(colors[q])
                            new_bad = len(new_set) < 3
                            if old_bad and not new_bad:
                                delta -= 1
                            elif not old_bad and new_bad:
                                delta += 1
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (p, new_c)
                
                if best_move is None:
                    break
                
                p, new_c = best_move
                if best_delta >= 0 and rng.random() > 0.1:
                    # Accept only if improving or with small probability for exploration
                    if best_delta > 0 and rng.random() > 0.05:
                        break
                    if best_delta == 0 and rng.random() > 0.3:
                        break
                
                colors[p] = new_c
                violations = count_violations(colors)
                
                if violations == 0:
                    return colors
                
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
                
                if best_delta > 0:
                    stall += 1
                else:
                    stall = 0
                
                if stall > 100:
                    break
        
        return best_colors
    
    def try_greedy_with_backtrack(k, budget):
        """Greedy with limited backtracking."""
        end_time = start_time + budget
        rng = random.Random(42)
        
        # Order positions by constraint tightness (most constrained first)
        # But for 4-APs, later positions are more constrained
        order = list(range(n))
        
        def assign(idx, colors):
            if idx == n:
                return True
            p = order[idx]
            # Try colors in order of preference
            color_order = list(range(k))
            rng.shuffle(color_order)
            
            for c in color_order:
                colors[p] = c
                # Check if this creates an immediate violation
                ok = True
                for idx2 in pos_aps[p]:
                    ap = aps[idx2]
                    # Check if all positions in this AP are assigned
                    if all(colors[q] is not None for q in ap):
                        s = {colors[q] for q in ap}
                        if len(s) < 3:
                            ok = False
                            break
                if ok:
                    if assign(idx + 1, colors):
                        return True
                colors[p] = None
            
            return False
        
        # Try with different orderings
        for attempt in range(20):
            if time.time() > end_time:
                break
            order = list(range(n))
            if attempt % 3 == 0:
                order.sort(key=lambda x: -len(pos_aps[x]))  # Most constrained first
            elif attempt % 3 == 1:
                order.sort(key=lambda x: len(pos_aps[x]))  # Least constrained first
            else:
                rng.shuffle(order)
            
            colors = [None] * n
            if assign(0, colors):
                return colors
        
        return None
    
    # Try 3 colors first
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
        
        # Try greedy with backtracking
        res = try_greedy_with_backtrack(k, 5.0)
        if res is not None:
            return res
        
        # Try local search with multiple seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            res = try_local_search(k, 3.0, seed)
            if res is not None and is_valid(res):
                return res
    
    # Fallback: use the seed's proven approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors
