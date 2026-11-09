import time
import numpy as np
import random

def construct():
    n = 256
    deadline = time.time() + 140.0  # 10s safety margin under 150s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 0-indexed
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    aps = np.array(aps, dtype=np.int32)  # shape (M, 4)
    M = len(aps)
    
    def count_mono(colors):
        """Count monochromatic 4-APs."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())
    
    def all_deltas(colors):
        """Compute delta for flipping each position. Returns array of length n."""
        # For each AP, determine if it's currently mono
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        mono_before = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # For each position x, compute the effect of flipping it
        # We need to check: for each AP containing x, does flipping x change mono status?
        # An AP is mono iff all 4 colors are equal.
        # Flipping position x in an AP: the AP becomes mono iff the other 3 positions
        # are all equal to (1 - colors[x]).
        
        deltas = np.zeros(n, dtype=np.int32)
        
        # For role 0 (position a): flip changes c0
        # AP becomes mono iff c1 == c2 == c3 == (1 - c0)
        # AP stops being mono iff it was mono (c0==c1==c2==c3)
        for role in range(4):
            col = aps[:, role]
            # For each AP, the other 3 colors
            other_cols = []
            for r in range(4):
                if r != role:
                    other_cols.append(aps[:, r])
            o0 = colors[other_cols[0]]
            o1 = colors[other_cols[1]]
            o2 = colors[other_cols[2]]
            
            # AP becomes mono after flip iff o0 == o1 == o2 == (1 - colors[col])
            becomes_mono = (o0 == o1) & (o1 == o2) & (o0 == (1 - colors[col]))
            # AP was mono before
            was_mono = mono_before
            
            # Net change per AP: +1 if becomes_mono and not was_mono, -1 if was_mono and not becomes_mono
            # Actually: delta = after - before
            # If was_mono and becomes_mono: 0
            # If was_mono and not becomes_mono: -1
            # If not was_mono and becomes_mono: +1
            # If not was_mono and not becomes_mono: 0
            ap_delta = np.where(was_mono & ~becomes_mono, -1,
                       np.where(~was_mono & becomes_mono, 1, 0)).astype(np.int32)
            
            # Accumulate to positions
            np.add.at(deltas, col, ap_delta)
        
        return deltas
    
    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    def greedy_local_search(colors, max_iters=10000):
        """Greedy: repeatedly flip the position with the best (most negative) delta."""
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            deltas = all_deltas(colors)
            best_delta = int(deltas.min())
            if best_delta >= 0:
                break
            best_x = int(np.argmin(deltas))
            colors[best_x] = 1 - colors[best_x]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
        
        return best_colors, best_cost
    
    def simulated_annealing(colors, max_iters=5000):
        """Simulated annealing to escape local optima."""
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        T_start = 5.0
        T_end = 0.01
        cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
        T = T_start
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            T *= cooling
            
            # Pick a random position to flip
            x = random.randint(0, n - 1)
            deltas = all_deltas(colors)
            delta = int(deltas[x])
            
            colors[x] = 1 - colors[x]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
            
            # Accept worse solutions with probability exp(-delta/T)
            if delta > 0:
                if T < 1e-6:
                    colors[x] = 1 - colors[x]
                    current_cost -= delta
                elif random.random() < np.exp(-delta / T):
                    pass  # accept
                else:
                    colors[x] = 1 - colors[x]
                    current_cost -= delta
        
        return best_colors, best_cost
    
    def iterated_local_search(colors, max_iters=200, perturb_strength=10):
        """ILS: local search, then perturb, repeat."""
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            # Local search
            c, cost = greedy_local_search(colors.copy(), max_iters=5000)
            if cost < best_cost:
                best_cost = cost
                best_colors = c.copy()
            
            # Perturb: flip a random number of positions
            num_flips = random.randint(perturb_strength // 2, perturb_strength)
            flip_positions = random.sample(range(n), num_flips)
            colors = c.copy()
            for x in flip_positions:
                colors[x] = 1 - colors[x]
        
        return best_colors, best_cost
    
    # Multi-start: Thue-Morse + random perturbations
    best_overall = thue_morse(n)
    best_cost_overall = count_mono(best_overall)
    
    # Start 1: pure Thue-Morse with ILS
    c, cost = iterated_local_search(best_overall.copy(), max_iters=80, perturb_strength=8)
    if cost < best_cost_overall:
        best_cost_overall = cost
        best_overall = c
    
    # Start 2: simulated annealing from Thue-Morse
    if time.time() < deadline - 30.0:
        c, cost = simulated_annealing(best_overall.copy(), max_iters=8000)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
        # Polish with greedy
        c2, cost2 = greedy_local_search(c.copy(), max_iters=5000)
        if cost2 < best_cost_overall:
            best_cost_overall = cost2
            best_overall = c2
    
    # Additional starts with random perturbations of Thue-Morse
    tm = thue_morse(n)
    num_starts = 12
    for s in range(num_starts):
        if time.time() > deadline - 20.0:
            break
        # Perturb Thue-Morse with random flips
        colors = tm.copy()
        num_flips = random.randint(10, 40)
        flip_positions = random.sample(range(n), num_flips)
        for x in flip_positions:
            colors[x] = 1 - colors[x]
        
        c, cost = iterated_local_search(colors.copy(), max_iters=40, perturb_strength=6)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
    
    # Final greedy polish
    if time.time() < deadline - 5.0:
        c, cost = greedy_local_search(best_overall.copy(), max_iters=10000)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
    
    # Final simulated annealing polish if time permits
    if time.time() < deadline - 10.0:
        c, cost = simulated_annealing(best_overall.copy(), max_iters=5000)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
        # Final greedy
        c2, cost2 = greedy_local_search(c.copy(), max_iters=5000)
        if cost2 < best_cost_overall:
            best_cost_overall = cost2
            best_overall = c2
    
    return [int(x) for x in best_overall]
