"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85.0
    n = 128

    # Precompute all nonconstant 4-term progressions as 0-indexed tuples
    all_4tps = []
    for a in range(1, n - 2):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            all_4tps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_4tps = len(all_4tps)
    
    # For each position, which 4TPs does it belong to
    pos_to_4tps = [[] for _ in range(n)]
    for idx, tps in enumerate(all_4tps):
        for pos in tps:
            pos_to_4tps[pos].append(idx)

    def count_violations(colors):
        """Count number of 4TPs with fewer than 3 distinct colors."""
        viol = 0
        for tp in all_4tps:
            c0, c1, c2, c3 = colors[tp[0]], colors[tp[1]], colors[tp[2]], colors[tp[3]]
            if len({c0, c1, c2, c3}) < 3:
                viol += 1
        return viol

    def is_valid(colors):
        """Check if every 4TP uses at least 3 distinct colors."""
        for tp in all_4tps:
            c0, c1, c2, c3 = colors[tp[0]], colors[tp[1]], colors[tp[2]], colors[tp[3]]
            if len({c0, c1, c2, c3}) < 3:
                return False
        return True

    def local_search(k, max_iters=5000, seed=0):
        """Local search: start from random assignment, iteratively fix violations."""
        rng = random.Random(seed)
        colors = [rng.randint(0, k - 1) for _ in range(n)]
        
        current_viol = count_violations(colors)
        if current_viol == 0:
            return colors
        
        best_colors = colors[:]
        best_viol = current_viol
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Pick a random violated 4TP
            # Find a violated progression
            viol_tps = []
            for idx, tp in enumerate(all_4tps):
                c0, c1, c2, c3 = colors[tp[0]], colors[tp[1]], colors[tp[2]], colors[tp[3]]
                if len({c0, c1, c2, c3}) < 3:
                    viol_tps.append(idx)
            
            if not viol_tps:
                return colors
            
            # Pick a random violated 4TP
            vi = rng.choice(viol_tps)
            tp = all_4tps[vi]
            
            # Try changing one of the 4 positions to a new color
            # Try all 4 positions, and for each try all k colors
            improved = False
            for pos_idx in range(4):
                pos = tp[pos_idx]
                old_color = colors[pos]
                for new_c in range(k):
                    if new_c == old_color:
                        continue
                    colors[pos] = new_c
                    new_viol = count_violations(colors)
                    if new_viol < current_viol:
                        current_viol = new_viol
                        improved = True
                        if current_viol < best_viol:
                            best_viol = current_viol
                            best_colors = colors[:]
                        break
                    else:
                        colors[pos] = old_color
                if improved:
                    break
            
            if not improved:
                # No single change improves; try a random perturbation
                pos = rng.randint(0, n - 1)
                colors[pos] = rng.randint(0, k - 1)
                current_viol = count_violations(colors)
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_colors = colors[:]
        
        if best_viol == 0:
            return best_colors
        return None

    # Try k=3 first
    best_solution = None
    
    for k in [3, 4, 5, 6, 8, 10, 12, 16]:
        if time.time() > deadline - 5:
            break
        
        max_restarts = 50 if k <= 4 else 30
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break
            
            result = local_search(k, max_iters=3000, seed=restart * 7919)
            if result is not None:
                best_solution = result
                break
        
        if best_solution is not None:
            break
    
    if best_solution is None:
        # Fallback: greedy with more colors
        for k in range(8, 32):
            if time.time() > deadline - 2:
                break
            for restart in range(20):
                if time.time() > deadline - 1:
                    break
                colors = [-1] * n
                colors[0] = restart % k
                success = True
                for x in range(1, n):
                    color_order = list(range(k))
                    random.shuffle(color_order)
                    placed = False
                    for c in color_order:
                        valid = True
                        for tp_idx in pos_to_4tps[x]:
                            tp = all_4tps[tp_idx]
                            all_assigned = True
                            distinct = {c}
                            for pos in tp:
                                if pos == x:
                                    continue
                                if colors[pos] == -1:
                                    all_assigned = False
                                    break
                                distinct.add(colors[pos])
                            if all_assigned and len(distinct) < 3:
                                valid = False
                                break
                        if valid:
                            colors[x] = c
                            placed = True
                            break
                    if not placed:
                        success = False
                        break
                if success and is_valid(colors):
                    best_solution = colors
                    break
            if best_solution is not None:
                break
    
    if best_solution is None:
        # Ultimate fallback
        best_solution = list(range(128))
    
    return best_solution

# EVOLVE-BLOCK-END
