import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline for safety
    
    n = 128
    
    # Precompute all 4-TPs and for each position, which TPs it's in
    def get_4tps():
        tps = []
        pos_to_tps = [[] for _ in range(n)]
        for a in range(1, n + 1):
            for d in range(1, n):
                a3 = a + 3 * d
                if a3 <= n:
                    tp = (a - 1, a + d - 1, a + 2 * d - 1, a3 - 1)
                    tp_idx = len(tps)
                    tps.append(tp)
                    for pos in tp:
                        pos_to_tps[pos].append(tp_idx)
        return tps, pos_to_tps
    
    all_tps, pos_to_tps = get_4tps()
    num_tps = len(all_tps)
    
    def count_distinct4(coloring, tp):
        return len({coloring[tp[0]], coloring[tp[1]], coloring[tp[2]], coloring[tp[3]]})
    
    def is_valid(coloring):
        for tp in all_tps:
            if count_distinct4(coloring, tp) < 3:
                return False
        return True
    
    def count_violations(coloring):
        count = 0
        for tp in all_tps:
            if count_distinct4(coloring, tp) < 3:
                count += 1
        return count
    
    def count_violations_for_pos(coloring, pos):
        """Count violations among TPs that include position pos."""
        count = 0
        for tp_idx in pos_to_tps[pos]:
            tp = all_tps[tp_idx]
            if count_distinct4(coloring, tp) < 3:
                count += 1
        return count
    
    # Try with increasing number of colors
    for max_colors in range(3, 129):
        if time.time() > deadline:
            break
        
        best_coloring = None
        best_violations = float('inf')
        
        num_restarts = 100
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initial coloring
            coloring = [random.randint(0, max_colors - 1) for _ in range(n)]
            
            # Local search with efficient violation counting
            current_violations = count_violations(coloring)
            
            for iteration in range(5000):
                if time.time() > deadline:
                    break
                
                if current_violations == 0:
                    best_coloring = coloring[:]
                    best_violations = 0
                    break
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
                
                # Pick a random position to change
                pos = random.randint(0, n - 1)
                old_color = coloring[pos]
                new_color = random.randint(0, max_colors - 1)
                if new_color == old_color:
                    continue
                
                # Compute change in violations
                # Only TPs containing pos are affected
                delta = 0
                for tp_idx in pos_to_tps[pos]:
                    tp = all_tps[tp_idx]
                    old_distinct = count_distinct4(coloring, tp)
                    coloring[pos] = new_color
                    new_distinct = count_distinct4(coloring, tp)
                    coloring[pos] = old_color
                    
                    old_bad = 1 if old_distinct < 3 else 0
                    new_bad = 1 if new_distinct < 3 else 0
                    delta += new_bad - old_bad
                
                if delta <= 0:
                    # Accept the move
                    coloring[pos] = new_color
                    current_violations += delta
                else:
                    # Revert (keep old color)
                    pass
            
            if best_violations == 0:
                break
        
        if best_violations == 0 and best_coloring is not None:
            # We found a valid coloring with at most max_colors colors
            # Now try to minimize the actual number of colors used
            coloring = best_coloring[:]
            
            # Try to reduce colors by merging
            for _ in range(200):
                if time.time() > deadline:
                    break
                
                used_colors = sorted(set(coloring))
                if len(used_colors) <= 2:
                    break
                
                # Try to merge two colors
                improved = False
                for i in range(len(used_colors)):
                    for j in range(i + 1, len(used_colors)):
                        if time.time() > deadline:
                            break
                        ci = used_colors[i]
                        cj = used_colors[j]
                        
                        # Try merging cj into ci
                        test_coloring = coloring[:]
                        for k in range(n):
                            if test_coloring[k] == cj:
                                test_coloring[k] = ci
                        
                        if is_valid(test_coloring):
                            coloring = test_coloring
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            
            return coloring
    
    # Fallback: return a simple valid coloring if we didn't find a better one
    # Use the greedy approach as fallback
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
