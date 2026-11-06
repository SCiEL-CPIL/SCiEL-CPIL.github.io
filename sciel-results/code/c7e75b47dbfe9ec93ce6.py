"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def _get_neighbors(pts, i):
    """Return dict: line_key -> set of indices (excluding i) on that line."""
    x1, y1 = pts[i]
    neighbors = {}
    for j in range(len(pts)):
        if j == i:
            continue
        x2, y2 = pts[j]
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        if g > 0:
            a //= g
            b //= g
            c //= g
        if a < 0 or (a == 0 and b < 0):
            a, b, c = -a, -b, -c
        key = (a, b, c)
        if key not in neighbors:
            neighbors[key] = set()
        neighbors[key].add(j)
    return neighbors

def construct():
    deadline = time.time() + 145
    
    # Initial configuration: 4x3 grid
    # This is a known good starting point with 12 ordinary lines
    start_pts = [[x, y] for x in range(4) for y in range(3)]
    
    if not _is_valid(start_pts):
        return start_pts
        
    best_pts = [list(p) for p in start_pts]
    best_score = _count_ordinary(best_pts)
    
    # Incremental local search
    # We maintain the line_map to allow O(n) updates
    line_map = {}
    for i in range(12):
        x1, y1 = best_pts[i]
        for j in range(i + 1, 12):
            x2, y2 = best_pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    current_score = best_score
    
    # Helper to calculate score change
    def calc_delta(pts, i, new_pos, line_map):
        # 1. Identify lines to remove
        lines_to_remove = []
        x1, y1 = pts[i]
        for j in range(12):
            if j == i:
                continue
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            lines_to_remove.append((a, b, c))
        
        # 2. Identify lines to add
        lines_to_add = []
        nx, ny = new_pos
        for j in range(12):
            if j == i:
                continue
            x2, y2 = pts[j]
            a = y2 - ny
            b = nx - x2
            c = x2 * ny - nx * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            lines_to_add.append((a, b, c))
            
        delta = 0
        
        # Process removals
        # We need to be careful: if multiple points are on the same line, we remove i once per line.
        # But line_map[key] is a set of all points on that line.
        # Removing i:
        # If len(S) == 2: line becomes empty (or 1 point? No, line requires 2 points). 
        # Wait, the line_map stores sets of points. A line is "ordinary" if len(set) == 2.
        # When we remove i from a line L:
        # If len(L) == 2: L was ordinary. Now len(L) becomes 1 (if we remove the other point too? No, we only remove i).
        # Actually, if we move i, we remove i from all lines passing through i.
        # If a line had {i, j}, removing i leaves {j}. The line no longer exists in the set of "lines with >= 2 points".
        # So if len(L) == 2, we lose 1 ordinary line.
        # If len(L) > 2, it wasn't ordinary, and remains non-ordinary (unless it drops to 2? No, it drops by 1).
        
        # Wait, what if the new position puts i on a line with {j, k}?
        # Then that line becomes {j, k, i}. If it was {j, k} (ordinary), it becomes non-ordinary. Delta -1.
        # If it was {j, k, l} (non-ordinary), it remains non-ordinary. Delta 0.
        
        # So:
        # Removal:
        for key in set(lines_to_remove):
            if key in line_map:
                pts_on_line = line_map[key]
                if i in pts_on_line:
                    old_size = len(pts_on_line)
                    if old_size == 2:
                        delta -= 1 # Lost an ordinary line
                    # If old_size > 2, no change to ordinary count
                    # Note: we don't actually update line_map yet, just calculating delta
            # If key not in line_map, it means no other points were on that line? 
            # No, line_map contains ALL lines defined by pairs. 
            # If i is the only point on a "line" (which is impossible as lines are defined by pairs), 
            # but here we are removing i from lines that i was part of.
            # Every line i was on had at least one other point j.
            # So key is always in line_map.
            
        # Addition:
        for key in set(lines_to_add):
            if key in line_map:
                pts_on_line = line_map[key]
                # i is not in pts_on_line yet (conceptually)
                # The line currently has pts_on_line.
                # If we add i, the line will have pts_on_line U {i}.
                # If len(pts_on_line) == 1: 
                #    Wait, can a line in line_map have size 1?
                #    No, line_map is built from pairs. So every key in line_map has size >= 2.
                #    UNLESS: we are in the middle of an update and some lines became size 1?
                #    No, we are calculating delta for a single move from a valid state.
                #    In a valid state, all lines in line_map have size >= 2.
                #    
                #    So, if key is in line_map, len(pts_on_line) >= 2.
                #    If len(pts_on_line) == 2: The line was ordinary. Adding i makes it size 3 (non-ordinary). Delta -1.
                #    If len(pts_on_line) > 2: The line was non-ordinary. Adding i makes it size+1 (non-ordinary). Delta 0.
                #    
                #    BUT: What if the new line key is NOT in line_map?
                #    That means no other pair of points defines this line.
                #    So the new line will contain only {i, j} (where j is the point defining the direction).
                #    Wait, lines_to_add contains keys for lines i-j.
                #    If key (i-j) is not in line_map, it means no other pair (k, l) is on this line.
                #    So the line will have exactly {i, j}.
                #    This is a NEW ordinary line. Delta +1.
                
                if len(pts_on_line) == 2:
                    delta -= 1
                # else delta 0
            else:
                # New line with exactly 2 points (i and j)
                delta += 1
                
        return delta

    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(12))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            orig_x, orig_y = best_pts[i]
            
            # Try small moves
            # We can't just try -1..1, we need to try moves that might align with existing lines
            # But for now, let's stick to a broader random search of small deltas
            # To speed up, we can limit the number of trials per point per iteration
            tried = 0
            while tried < 20 and time.time() < deadline:
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = orig_x + dx, orig_y + dy
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                
                # Check distinctness
                valid = True
                for j in range(12):
                    if j == i:
                        continue
                    if best_pts[j][0] == new_x and best_pts[j][1] == new_y:
                        valid = False
                        break
                if not valid:
                    tried += 1
                    continue
                
                # Calculate delta
                d = calc_delta(best_pts, i, (new_x, new_y), line_map)
                
                if d < 0:
                    # Apply move
                    best_pts[i] = [new_x, new_y]
                    current_score += d
                    improved = True
                    
                    # Update line_map
                    # 1. Remove i from old lines
                    for key in set(_get_neighbors(best_pts, i).keys()): # Wait, best_pts is already updated?
                         # No, we need old neighbors.
                         pass
                     
                    # Let's re-implement the update carefully.
                    # We have best_pts updated.
                    # We need to update line_map.
                    
                    # Old neighbors (calculated before move? No, we can recalculate or store)
                    # It's safer to recalculate the lines involving i and update the map.
                    
                    # 1. Find all lines that i was on (old position)
                    # We lost this info if we didn't store it. 
                    # Let's store old_pos.
                    pass
                    
                    # Actually, the calc_delta function didn't update the map.
                    # Let's do the map update now.
                    
                    # We need to know which lines i was on.
                    # Let's re-calculate neighbors for the OLD position? 
                    # We can just iterate all pairs and rebuild? No, too slow.
                    
                    # Let's modify the logic:
                    # Before updating best_pts, we should have identified the lines.
                    # Let's re-do the move application properly.
                    
                    # Rollback best_pts
                    best_pts[i] = [orig_x, orig_y]
                    
                    # Get old neighbors
                    old_neighbors = _get_neighbors(best_pts, i)
                    
                    # Get new neighbors
                    new_pos = (new_x, new_y)
                    # Temporarily set i to new pos to get new neighbors?
                    # Or calculate directly.
                    new_neighbors = {}
                    for j in range(12):
                        if j == i: continue
                        x2, y2 = best_pts[j]
                        a = y2 - new_y
                        b = new_x - x2
                        c = x2 * new_y - new_x * y2
                        g = gcd(gcd(abs(a), abs(b)), abs(c))
                        if g > 0:
                            a //= g; b //= g; c //= g
                        if a < 0 or (a == 0 and b < 0):
                            a, b, c = -a, -b, -c
                        key = (a, b, c)
                        if key not in new_neighbors:
                            new_neighbors[key] = set()
                        new_neighbors[key].add(j)
                        
                    # Update line_map
                    # Remove i from old lines
                    for key, pts_set in old_neighbors.items():
                        if key in line_map:
                            line_map[key].discard(i)
                            if len(line_map[key]) < 2:
                                del line_map[key]
                    
                    # Add i to new lines
                    for key, pts_set in new_neighbors.items():
                        if key not in line_map:
                            line_map[key] = set()
                        line_map[key].add(i)
                        for j in pts_set:
                            line_map[key].add(j)
                        # Note: if key existed, we just add i. The other points are already there.
                        # But wait, if key existed, it means some other pair defined it.
                        # The set in line_map[key] already contains those points.
                        # We just need to add i.
                        # My code above: line_map[key].add(i) is sufficient if key exists.
                        # If key doesn't exist, we create it with {i} U pts_set.
                        # The code above does:
                        # if key not in line_map: line_map[key] = set()
                        # line_map[key].add(i)
                        # for j in pts_set: line_map[key].add(j)
                        # This is correct.
                        
                    # Apply move to best_pts
                    best_pts[i] = [new_x, new_y]
                    current_score += d
                    
                    if current_score < best_score:
                        best_score = current_score
                        # Copy best
                        # We are modifying best_pts in place, so it is the best.
                    break # Move accepted, try next point? Or continue with same point?
                    # Usually, accept and move to next point to allow global exploration.
                    # But "improved" flag is set, so we will loop again.
                    # Breaking the inner while loop is good.
                
                tried += 1
        
        # If no improvement in a full pass, we might be stuck.
        # We can try a random restart or larger jumps.
        # For now, if improved is False, we break.
        # To avoid premature termination, we can allow a few non-improving passes?
        # Or just break. Given the time limit, breaking is safer.
        if not improved:
            break
            
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
