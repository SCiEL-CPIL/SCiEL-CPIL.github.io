"""Colorings with three colors in every four-term progression."""
import time
import random
from itertools import product

def construct():
    n = 128
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d) with 0-indexed
    # Positions are 0..127, corresponding to 1..128
    taps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(n - 3 * d):
            taps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, precompute which taps it belongs to and its index in that tap
    pos_taps = [[] for _ in range(n)]
    for ti, (a, b, c, d) in enumerate(taps):
        pos_taps[a].append((ti, 0))
        pos_taps[b].append((ti, 1))
        pos_taps[c].append((ti, 2))
        pos_taps[d].append((ti, 3))
    
    def count_colors(colors):
        return len(set(colors))
    
    def check_violation_at(pos, color, colors):
        """Check if assigning 'color' to 'pos' creates any violation in taps where
        all other positions are already assigned."""
        for ti, idx in pos_taps[pos]:
            a, b, c, d = taps[ti]
            pts = [a, b, c, d]
            others = [p for j, p in enumerate(pts) if j != idx]
            other_colors = [colors[p] for p in others]
            if -1 in other_colors:
                continue
            full_set = set(other_colors) | {color}
            if len(full_set) < 3:
                return True
        return False
    
    def count_violations(colors):
        viol = 0
        for (a, b, c, d) in taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                viol += 1
        return viol
    
    def greedy_construct(max_colors, seed=0):
        """Greedy left-to-right with constraint checking."""
        rng = random.Random(seed)
        colors = [-1] * n
        
        for pos in range(n):
            forbidden = set()
            stuck = False
            
            for ti, idx in pos_taps[pos]:
                a, b, c, d = taps[ti]
                pts = [a, b, c, d]
                others = [p for j, p in enumerate(pts) if j != idx]
                other_colors = [colors[p] for p in others]
                if -1 in other_colors:
                    continue
                other_set = set(other_colors)
                if len(other_set) == 1:
                    # Three same color -> impossible to satisfy
                    stuck = True
                    break
                elif len(other_set) == 2:
                    # Need a third distinct color
                    forbidden.update(other_set)
            
            if stuck:
                return None
            
            # Try colors in shuffled order for diversity
            color_order = list(range(max_colors))
            rng.shuffle(color_order)
            
            assigned = False
            for c in color_order:
                if c not in forbidden:
                    colors[pos] = c
                    assigned = True
                    break
            
            if not assigned:
                return None
        
        if count_violations(colors) > 0:
            return None
        
        return colors
    
    def local_search(colors, max_iters=20000, time_limit=80.0):
        """Local search: try to reduce number of colors by recoloring."""
        start_ls = time.time()
        colors = list(colors)
        current_colors = set(colors)
        best_colors = list(colors)
        best_num = len(current_colors)
        
        # Try to reduce color palette
        for target_colors in range(best_num - 1, 0, -1):
            if time.time() - start_ls > time_limit:
                break
            
            # Try to map all colors to target_colors colors
            success = False
            for attempt in range(50):
                if time.time() - start_ls > time_limit:
                    break
                
                # Randomly assign original colors to target palette
                rng = random.Random(attempt * 7919 + int(time.time() * 1000) % 10000)
                color_map = {}
                orig_colors = sorted(set(colors))
                for oc in orig_colors:
                    color_map[oc] = rng.randint(0, target_colors - 1)
                
                new_colors = [color_map[c] for c in colors]
                
                # Try to fix violations by local moves
                improved = True
                while improved and time.time() - start_ls < time_limit:
                    improved = False
                    viol_count = count_violations(new_colors)
                    if viol_count == 0:
                        if len(set(new_colors)) < best_num:
                            best_num = len(set(new_colors))
                            best_colors = list(new_colors)
                        success = True
                        break
                    
                    # Find a violating tap and try to fix it
                    for (a, b, c, d) in taps:
                        cs = [new_colors[a], new_colors[b], new_colors[c], new_colors[d]]
                        if len(set(cs)) < 3:
                            # Try changing one position
                            for pos in [a, b, c, d]:
                                old_c = new_colors[pos]
                                for new_c in range(target_colors):
                                    if new_c == old_c:
                                        continue
                                    new_colors[pos] = new_c
                                    # Quick check: does this fix the violation without creating too many new ones?
                                    new_viol = count_violations(new_colors)
                                    if new_viol < viol_count:
                                        improved = True
                                        break
                                    else:
                                        new_colors[pos] = old_c
                                if improved:
                                    break
                            if improved:
                                break
                    
                    if not improved:
                        break
                
                if success:
                    break
            
            if success:
                continue  # Try to reduce further
            else:
                break
        
        return best_colors, best_num
    
    def backtracking_refine(colors, max_depth=30, time_limit=80.0):
        """Limited backtracking to try to reduce color count."""
        start_bt = time.time()
        colors = list(colors)
        current_colors = set(colors)
        best = list(colors)
        best_num = len(current_colors)
        
        # Try to find a position where we can use a color not in the current palette
        # or reduce palette usage
        for target in range(best_num - 1, 1, -1):
            if time.time() - start_bt > time_limit:
                break
            
            # Try different random seeds for greedy with fewer colors
            found = False
            for seed in range(20):
                if time.time() - start_bt > time_limit:
                    break
                result = greedy_construct(target, seed=seed)
                if result is not None:
                    if len(set(result)) < best_num:
                        best_num = len(set(result))
                        best = list(result)
                    found = True
                    break
            
            if found:
                continue
            else:
                break
        
        return best, best_num
    
    # Strategy:
    # 1. Try greedy with increasing color counts from minimum possible
    # 2. Apply local search to improve
    # 3. Use backtracking refinement
    
    best_result = None
    best_num = 128
    
    # First, get a valid coloring with greedy (use enough colors to guarantee success)
    for max_c in range(4, 13):
        if time.time() > deadline:
            break
        result = greedy_construct(max_c, seed=42)
        if result is not None:
            num = len(set(result))
            if num < best_num:
                best_num = num
                best_result = list(result)
            break
    
    if best_result is None:
        # Fallback: use more colors
        for max_c in range(13, 20):
            if time.time() > deadline:
                break
            result = greedy_construct(max_c, seed=42)
            if result is not None:
                num = len(set(result))
                if num < best_num:
                    best_num = num
                    best_result = list(result)
                break
    
    if best_result is None:
        # Last resort: greedy with many colors
        result = greedy_construct(20, seed=42)
        if result is not None:
            best_result = list(result)
            best_num = len(set(result))
    
    if best_result is None:
        # Emergency fallback: return something (shouldn't happen)
        return list(range(128))
    
    # Try local search to improve
    if time.time() < deadline - 10:
        ls_result, ls_num = local_search(best_result, max_iters=30000, time_limit=75.0)
        if ls_num < best_num:
            best_num = ls_num
            best_result = ls_result
    
    # Try backtracking refinement
    if time.time() < deadline - 5:
        bt_result, bt_num = backtracking_refine(best_result, max_depth=30, time_limit=78.0)
        if bt_num < best_num:
            best_num = bt_num
            best_result = bt_result
    
    # Final: try more greedy seeds with the current best_num and below
    if time.time() < deadline - 3:
        for target in range(2, best_num):
            if time.time() > deadline - 2:
                break
            for seed in range(50):
                if time.time() > deadline - 2:
                    break
                result = greedy_construct(target, seed=seed)
                if result is not None:
                    num = len(set(result))
                    if num < best_num:
                        best_num = num
                        best_result = list(result)
                    break
    
    return best_result
