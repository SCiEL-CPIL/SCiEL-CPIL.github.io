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
    
    def compute_delta_fast(coloring, pos, old_color, new_color):
        """Compute change in violations when changing position pos from old_color to new_color."""
        delta = 0
        for tp_idx in pos_to_tps[pos]:
            tp = all_tps[tp_idx]
            others = [p for p in tp if p != pos]
            other_colors = [coloring[p] for p in others]
            
            old_distinct = len({old_color} | set(other_colors))
            new_distinct = len({new_color} | set(other_colors))
            
            old_bad = 1 if old_distinct < 3 else 0
            new_bad = 1 if new_distinct < 3 else 0
            delta += new_bad - old_bad
        return delta
    
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
            
            # Structured initial coloring for first few restarts
            if restart == 0:
                coloring = [i % max_colors for i in range(n)]
            elif restart == 1:
                coloring = [(i * 7) % max_colors for i in range(n)]
            else:
                coloring = [random.randint(0, max_colors - 1) for _ in range(n)]
            
            # Local search with simulated annealing
            current_violations = count_violations(coloring)
            temperature = 5.0
            cooling = 0.995
            
            for iteration in range(8000):
                if time.time() > deadline:
                    break
                
                if current_violations == 0:
                    best_coloring = coloring[:]
                    best_violations = 0
                    break
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
                
                # Pick a position: bias toward positions in violating TPs
                violating_positions = set()
                for tp_idx in range(num_tps):
                    tp = all_tps[tp_idx]
                    if count_distinct4(coloring, tp) < 3:
                        for pos in tp:
                            violating_positions.add(pos)
                
                if violating_positions and random.random() < 0.7:
                    pos = random.choice(list(violating_positions))
                else:
                    pos = random.randint(0, n - 1)
                
                old_color = coloring[pos]
                new_color = random.randint(0, max_colors - 1)
                if new_color == old_color:
                    continue
                
                delta = compute_delta_fast(coloring, pos, old_color, new_color)
                
                # Simulated annealing acceptance
                if delta <= 0:
                    accept = True
                else:
                    accept = random.random() < pow(2.718281828, -delta / temperature)
                
                if accept:
                    coloring[pos] = new_color
                    current_violations += delta
                
                temperature *= cooling
            
            if best_violations == 0:
                break
        
        if best_violations == 0 and best_coloring is not None:
            # We found a valid coloring with at most max_colors colors
            # Now try to minimize the actual number of colors used
            coloring = best_coloring[:]
            
            # Try to reduce colors by merging - try smaller targets first
            for target_colors in range(3, max_colors + 1):
                if time.time() > deadline:
                    break
                
                used_colors = sorted(set(coloring))
                if len(used_colors) <= target_colors:
                    break
                
                # Try to merge colors to reach target_colors
                color_map = {c: c for c in used_colors}
                current_num_colors = len(used_colors)
                
                for _ in range(500):
                    if time.time() > deadline:
                        break
                    
                    if current_num_colors <= target_colors:
                        break
                    
                    # Try merging each pair of colors
                    improved = False
                    colors_list = sorted(set(color_map.values()))
                    for i in range(len(colors_list)):
                        for j in range(i + 1, len(colors_list)):
                            if time.time() > deadline:
                                break
                            ci = colors_list[i]
                            cj = colors_list[j]
                            
                            # Try merging cj into ci
                            test_map = color_map.copy()
                            for c in test_map:
                                if test_map[c] == cj:
                                    test_map[c] = ci
                            
                            # Check validity
                            test_coloring = [test_map[c] for c in coloring]
                            if is_valid(test_coloring):
                                color_map = test_map
                                current_num_colors -= 1
                                improved = True
                                break
                        if improved:
                            break
                    
                    if not improved:
                        break
                
                if current_num_colors <= target_colors:
                    break
            
            # Apply the color map
            final_coloring = [color_map[c] for c in coloring]
            
            # Verify
            if is_valid(final_coloring):
                return final_coloring
            else:
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
