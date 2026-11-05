import time

def construct():
    deadline = time.time() + 140
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def is_adjacent(p, q):
        return dist_sq(p, q) == 1

    # 1. Generate Hexagon of radius 2
    # Condition: max(|a|, |b|, |a+b|) <= 2
    hex2 = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a+b)) <= 2:
                hex2.add((a, b))
                
    # 2. Generate Shell of radius 3
    # Condition: max(|a|, |b|, |a+b|) == 3
    shell3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) == 3:
                shell3.append((a, b))
                
    # 3. Greedy selection of 13 points from shell3
    current_set = set(hex2)
    selected_shell = set()
    
    # We need to pick 13 points.
    for _ in range(13):
        best_point = None
        best_gain = -1
        
        remaining = [p for p in shell3 if p not in selected_shell]
        
        for p in remaining:
            # Gain = edges to current_set + edges to selected_shell
            gain = 0
            for q in current_set:
                if is_adjacent(p, q):
                    gain += 1
            for q in selected_shell:
                if is_adjacent(p, q):
                    gain += 1
            
            # Tie-breaking: prefer points that are "central" or have more potential?
            # For now, just max gain. If tie, pick first.
            if gain > best_gain:
                best_gain = gain
                best_point = p
                
        if best_point:
            selected_shell.add(best_point)
            current_set.add(best_point)
            
    # Verify we have 32 points
    if len(current_set) != 32:
        # Fallback to random if something went wrong (should not happen)
        # But let's assume the logic is correct.
        pass

    # Calculate score to ensure it's 76
    # If not 76, we might need a local search refinement.
    # But greedy on this symmetric problem should work.
    
    # Optional: Local search on the final set to ensure 76.
    # If score < 76, try swaps.
    
    points_list = list(current_set)
    
    def count_edges(pts):
        cnt = 0
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                if is_adjacent(pts[i], pts[j]):
                    cnt += 1
        return cnt
    
    current_score = count_edges(points_list)
    
    # If we didn't hit 76, run a quick hill climb
    if current_score < 76:
        # Generate a larger pool for swaps if needed
        # But let's assume greedy is good.
        # To be safe, run a limited local search.
        
        # Pool: Hexagon radius 6 (larger than before to escape local minima)
        R = 6
        pool = []
        for a in range(-R, R+1):
            for b in range(-R, R+1):
                if max(abs(a), abs(b), abs(a+b)) <= R:
                    pool.append((a, b))
                    
        current_set = set(points_list)
        
        improved = True
        stagnation = 0
        max_stagnation = 50
        while improved and current_score < 76 and time.time() < deadline - 5:
            improved = False
            best_swap_score = current_score
            best_swap_in = None
            best_swap_out = None
            
            # Try swaps
            # Optimization: only try swapping points on the boundary with points in the pool
            # For simplicity, try all, but limit iterations or use random sampling if too slow
            # 32 * (pool size ~ 127) = ~4000 swaps.
            # Each swap check O(32). 4000 * 32 = 128000 ops. Very fast.
            
            for p_in in current_set:
                deg_in = 0
                for p_other in current_set:
                    if p_in != p_other and is_adjacent(p_in, p_other):
                        deg_in += 1
                            
                for p_out in pool:
                    if p_out in current_set:
                        continue
                    
                    deg_out = 0
                    for p_other in current_set:
                        if p_out != p_other and is_adjacent(p_out, p_other):
                            deg_out += 1
                            
                    if is_adjacent(p_in, p_out):
                        deg_out -= 1
                            
                    new_score = current_score - deg_in + deg_out
                    
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap_in = p_in
                        best_swap_out = p_out
                        
            if best_swap_in is not None:
                current_set.remove(best_swap_in)
                current_set.add(best_swap_out)
                current_score = best_swap_score
                improved = True
                stagnation = 0
                if current_score > 76: # Should not exceed
                    break
            else:
                stagnation += 1
                if stagnation > max_stagnation:
                    # Try a random perturbation to escape local minimum
                    if current_score < 76 and time.time() < deadline - 10:
                        # Pick a random point in current_set and a random point in pool (not in current_set)
                        # and swap them, even if it decreases score slightly, to jump over a barrier?
                        # Or just break if we are stuck.
                        # Given the small size, random restart might be better, but we can't restart easily.
                        # Let's just break to save time if stuck.
                        break

    return [[a, b] for a, b in current_set]
