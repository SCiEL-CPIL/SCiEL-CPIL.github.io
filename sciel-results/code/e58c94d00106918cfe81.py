import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128

    # Precompute 4-APs
    aps = []
    for a in range(n):
        for d in range(1, n):
            a3 = a + 3 * d
            if a3 < n:
                aps.append((a, a+d, a+2*d, a3))
    
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def is_valid_ap(colors, a, b, c, d):
        return len({colors[a], colors[b], colors[c], colors[d]}) >= 3

    def get_cost(colors):
        cost = 0
        for a, b, c, d in aps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                cost += 1
        return cost

    def local_search(initial_colors, max_c, max_time):
        rng = random.Random()
        colors = list(initial_colors)
        current_cost = get_cost(colors)
        
        if current_cost == 0:
            return colors

        best_colors = list(colors)
        best_cost = current_cost
        no_improve = 0
        max_no_improve = 2000

        while time.time() < max_time:
            pos = rng.randint(0, n - 1)
            old_c = colors[pos]
            
            # Try other colors
            candidates = [c for c in range(max_c) if c != old_c]
            rng.shuffle(candidates)
            
            improved = False
            for new_c in candidates:
                # Calculate delta
                delta = 0
                for ap_idx in pos_to_aps[pos]:
                    a, b, c, d = aps[ap_idx]
                    # Get old colors
                    ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
                    old_bad = 1 if len({ca, cb, cc, cd}) < 3 else 0
                    
                    # Get new colors
                    if a == pos: ca = new_c
                    elif b == pos: cb = new_c
                    elif c == pos: cc = new_c
                    elif d == pos: cd = new_c
                    
                    new_bad = 1 if len({ca, cb, cc, cd}) < 3 else 0
                    delta += new_bad - old_bad
                
                new_cost = current_cost + delta
                if new_cost < current_cost:
                    colors[pos] = new_c
                    current_cost = new_cost
                    improved = True
                    if current_cost == 0:
                        return colors
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = list(colors)
                    break
                elif new_cost == current_cost and rng.random() < 0.05:
                    colors[pos] = new_c
                    improved = True
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    # Restart
                    # Randomly perturb
                    for _ in range(10):
                        p = rng.randint(0, n-1)
                        colors[p] = rng.randint(0, max_c - 1)
                    current_cost = get_cost(colors)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = list(colors)
                    no_improve = 0
            else:
                no_improve = 0

        return best_colors

    # Try 3 colors
    for i in range(100):
        if time.time() > deadline - 10:
            break
        init = [random.randint(0, 2) for _ in range(n)]
        res = local_search(init, 3, deadline - 10)
        if res and get_cost(res) == 0:
            return res

    # Try 4 colors
    for i in range(100):
        if time.time() > deadline - 5:
            break
        init = [random.randint(0, 3) for _ in range(n)]
        res = local_search(init, 4, deadline - 5)
        if res and get_cost(res) == 0:
            return res

    # Fallback
    colors = [0] * n
    for i in range(n):
        for c in range(4):
            colors[i] = c
            valid = True
            for ap_idx in pos_to_aps[i]:
                a, b, cd, d = aps[ap_idx]
                if len({colors[a], colors[b], colors[cd], colors[d]}) < 3:
                    valid = False
                    break
            if valid:
                break
    return colors
