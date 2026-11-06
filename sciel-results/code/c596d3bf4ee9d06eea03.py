"""Combined program for 20-point no-4-collinear configuration."""
import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. Use 140s limit to be safe.
    timeout = 140
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                count = 0
                for k in range(n):
                    if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 2:
                    return True
        return False

    def get_collinear_triples(points):
        n = len(points)
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    def max_no_3_collinear_exact(points):
        """
        Exact branch and bound for max subset with no 3 collinear.
        Uses bitmasks for speed.
        """
        n = len(points)
        if n <= 2:
            return n
        
        triples = get_collinear_triples(points)
        if not triples:
            return n
            
        # Precompute for each point, the set of pairs it forms a triple with
        # conflict_pairs[i] is a list of (j, k) such that (i, j, k) is a triple
        # This is useful for checking if adding i is valid given current set S.
        # Actually, for B&B, we just need to know if a subset is valid.
        
        # Let's use a recursive solver with bitmasks.
        # State: available_mask (points not yet decided), current_mask (points selected)
        # Constraint: current_mask must not contain any triple.
        
        all_mask = (1 << n) - 1
        best_size = 0
        
        # Optimization: If a point is in many triples, branching on it is good.
        # Precompute degree of each point in the hypergraph
        deg = [0] * n
        for u, v, w in triples:
            deg[u] += 1
            deg[v] += 1
            deg[w] += 1
            
        # Order points by degree descending to improve pruning?
        # Or just use the natural order.
        
        def is_valid_subset(mask):
            # Check if mask contains any triple
            for u, v, w in triples:
                if (mask & (1 << u)) and (mask & (1 << v)) and (mask & (1 << w)):
                    return False
            return True
            
        def solve(available_mask, current_size, current_mask):
            nonlocal best_size
            
            # Pruning: Even if we take all available, we can't beat best
            if current_size + bin(available_mask).count('1') <= best_size:
                return
                
            if available_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return
                
            # Heuristic: Pick the point with highest degree in the remaining hypergraph
            # Or just pick the lowest set bit for simplicity, but degree-based is better.
            # Let's pick the lowest set bit for now, as n=20 is small.
            # If performance is an issue, switch to degree-based.
            
            # Find lowest set bit
            v = (available_mask & -available_mask).bit_length() - 1
            v_bit = 1 << v
            
            # Option 1: Exclude v
            solve(available_mask ^ v_bit, current_size, current_mask)
            
            # Option 2: Include v
            # Check if including v is valid with current_mask
            # We need to ensure no triple (v, u, w) exists in current_mask
            # Actually, current_mask is the set of *selected* points.
            # If we select v, we must ensure that for every triple (v, u, w), 
            # we haven't already selected both u and w.
            
            valid_to_include = True
            for u, w in conflict_pairs[v]:
                if (current_mask & (1 << u)) and (current_mask & (1 << w)):
                    valid_to_include = False
                    break
                    
            if valid_to_include:
                solve(available_mask ^ v_bit, current_size + 1, current_mask | v_bit)
                
        # Precompute conflict_pairs for fast checking
        conflict_pairs = [[] for _ in range(n)]
        for u, v, w in triples:
            # For u, conflict is (v, w)
            conflict_pairs[u].append((v, w))
            # For v, conflict is (u, w)
            conflict_pairs[v].append((u, w))
            # For w, conflict is (u, v)
            conflict_pairs[w].append((u, v))
            
        solve(all_mask, 0, 0)
        return best_size

    def score_config(points):
        return max_no_3_collinear_exact(points)
    
    best_score = 20
    best_points = None
    
    # 1. Generate structured configurations
    # Strategy: Use lines with max 3 points.
    # Try to create a configuration where points are heavily clustered on lines.
    
    # Config A: Grid-like but perturbed to avoid 4-collinear
    # Use a 5x4 grid? 20 points.
    # If we use a standard grid, there are many 4-collinear.
    # We need to "break" the 4-collinear lines by moving points.
    
    # Let's try a specific known good structure: 
    # Points on a circle? No, integer coordinates.
    # Points on a parabola y=x^2? No 3 collinear? 
    # For y=x^2, any 3 points (x1, x1^2), (x2, x2^2), (x3, x3^2) are never collinear 
    # unless x1=x2=x3. 
    # So a parabola has score 20. We want to MINIMIZE score, so we want MAX collinearities.
    # So we want points to be ON lines.
    
    # Let's try to construct a set of lines covering the 20 points.
    # Lines: y=0 (3 pts), y=1 (3 pts), y=2 (3 pts), y=3 (3 pts), y=4 (3 pts), y=5 (3 pts), y=6 (2 pts).
    # Total 20 points.
    # Check 4-collinear:
    # Horizontal lines have 3 pts. OK.
    # Vertical lines?
    # If we place them at x = 0, 10, 20 for all y, then vertical lines have 7 points! BAD.
    # So we must stagger the x-coordinates.
    
    # Let's define the points explicitly for a staggered grid.
    # y=0: (0,0), (10,0), (20,0)
    # y=1: (1,1), (11,1), (21,1)
    # y=2: (2,2), (12,2), (22,2)
    # y=3: (3,3), (13,3), (23,3)
    # y=4: (4,4), (14,4), (24,4)
    # y=5: (5,5), (15,5), (25,5)
    # y=6: (6,6), (16,6)
    
    # Check verticals:
    # x=0: (0,0)
    # x=1: (1,1)
    # ...
    # x=6: (6,6)
    # x=10: (10,0), (11,1)? No.
    # Let's check diagonals.
    # (0,0), (1,1), (2,2)... (6,6) -> 7 points on y=x! BAD.
    
    # We need to break diagonals too.
    # Use a "Latin Square" like distribution or random offset per row.
    
    def generate_staggered_grid():
        pts = []
        used = set()
        base_x = 10
        # 7 rows
        for y in range(7):
            # 3 points per row, except last
            count = 3 if y < 6 else 2
            # Randomize x offset for this row to break verticals and diagonals
            offset = random.randint(-5, 5)
            for i in range(count):
                x = base_x * i + offset + y # Add y to break pure verticals?
                # Ensure distinct and in bounds
                # Simple retry loop
                while (x, y) in used:
                    x += 1
                    if x > 100: x = -100
                if abs(x) <= 100:
                    pts.append([x, y])
                    used.add((x, y))
        return pts
    
    # This random generation might still have 4-collinear.
    # Better: Use the random search with local optimization, but start from good seeds.
    
    # Seed 1: The staggered grid above, then optimize.
    # Seed 2: Random points.
    
    random.seed(42)
    
    # Run local search on a few initial candidates
    candidates = []
    
    # Candidate 1: Staggered grid
    c1 = generate_staggered_grid()
    if len(c1) == 20 and not has_4_collinear(c1):
        candidates.append(c1)
    
    # Candidate 2: Random
    for _ in range(10):
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        if not has_4_collinear(pts):
            candidates.append(pts)
            
    # Local Search
    def local_search(points):
        current_points = [p[:] for p in points]
        current_score = score_config(current_points)
        improved = True
        while improved and (time.time() - start_time) < timeout - 10:
            improved = False
            for i in range(20):
                old_x, old_y = current_points[i]
                # Try a few random moves
                for _ in range(20):
                    new_x = random.randint(max(-100, old_x - 15), min(100, old_x + 15))
                    new_y = random.randint(max(-100, old_y - 15), min(100, old_y + 15))
                    if (new_x, new_y) == (old_x, old_y):
                        continue
                    # Check if new point collides
                    if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(current_points) if j != i):
                        continue
                    
                    current_points[i] = [new_x, new_y]
                    
                    if has_4_collinear(current_points):
                        current_points[i] = [old_x, old_y]
                        continue
                        
                    new_score = score_config(current_points)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current_points[i] = [old_x, old_y]
                if improved:
                    break
        return current_points, current_score
        
    for cand in candidates:
        if time.time() - start_time > timeout - 20:
            break
        pts, s = local_search(cand)
        if s < best_score:
            best_score = s
            best_points = pts
            # Print for debugging if allowed? No, just keep quiet.
            # print(f"Found score {s}")

    # If we didn't find anything better than 20 (unlikely), fallback
    if best_points is None:
        # Fallback to a simple valid config
        # Parabola y = x^2 for x in -5..5 (11 pts) + y = x^2+10 for x in -5..5 (11 pts)
        # Check 4-collinear?
        # Parabolas don't have 3 collinear, so no 4.
        pts = []
        for x in range(-5, 6):
            pts.append([x, x*x])
        for x in range(-5, 6):
            pts.append([x, x*x + 20])
        best_points = pts
        best_score = score_config(best_points)

    return best_points

# EVOLVE-BLOCK-END
