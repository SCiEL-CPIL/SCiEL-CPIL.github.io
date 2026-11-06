"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _edge_count(points):
    """Count unit-distance edges among the given set of lattice points."""
    s = set(tuple(p) for p in points)
    count = 0
    for (a, b) in s:
        for da, db in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)):
            if (a + da, b + db) in s:
                count += 1
    return count // 2

def _moment_of_inertia(points):
    """Calculate sum of squared distances from centroid."""
    if not points:
        return 0
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    return sum((p[0] - cx)**2 + (p[1] - cy)**2 for p in points)

def construct():
    deadline = time.time() + 110.0
    N = 32
    OPT = 76
    
    # 1. Generate initial candidate: Hexagon radius 3 minus 5 corners
    hex_pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if abs(a) + abs(b) + abs(a + b) <= 6:
                hex_pts.append([a, b])
    
    inner = [p for p in hex_pts if abs(p[0]) + abs(p[1]) + abs(p[0]+p[1]) < 6]
    outer = [p for p in hex_pts if abs(p[0]) + abs(p[1]) + abs(p[0]+p[1]) == 6]
    
    # Identify corners (points with 3 neighbors in hexagon, specifically 1 inner, 2 outer)
    # Corners are at (3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3)
    corners = [[3,0], [0,3], [-3,3], [-3,0], [0,-3], [3,-3]]
    
    # Remove 5 corners to maximize edges (lose 3 edges each) and minimize inertia (remove far points)
    # We have 6 corners, remove 5. Keep the one that minimizes inertia? No, remove the 5 that maximize inertia?
    # Actually, all corners are symmetric. Removing any 5 leaves a shape with 1 corner.
    # To minimize inertia, we should keep the shape as compact as possible.
    # Removing 5 corners breaks the symmetry.
    # Let's try removing 5 specific corners and check inertia.
    
    best_seed = None
    best_seed_inertia = float('inf')
    best_seed_edges = -1
    
    # Try all combinations of 5 corners to remove? 6C5 = 6 cases.
    for i in range(6):
        remove_set = set(tuple(c) for c in corners if c != corners[i])
        candidate = [p for p in hex_pts if tuple(p) not in remove_set]
        edges = _edge_count(candidate)
        if edges > best_seed_edges or (edges == best_seed_edges and _moment_of_inertia(candidate) < best_seed_inertia):
            best_seed_edges = edges
            best_seed_inertia = _moment_of_inertia(candidate)
            best_seed = candidate
            
    best_pts = [list(p) for p in best_seed]
    best_edges = best_seed_edges
    
    # 2. Simulated Annealing to minimize Moment of Inertia while keeping Edges == OPT
    # If we don't have OPT edges, we try to maximize edges first.
    
    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    if best_edges < OPT:
        # Hill Climbing to increase edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            pts = [list(p) for p in best_pts]
            pt_set = set(tuple(p) for p in pts)
            for i in range(N):
                if time.time() > deadline: break
                a, b = pts[i]
                for da, db in directions:
                    na, nb = a + da, b + db
                    if (na, nb) in pt_set or abs(na) > 64 or abs(nb) > 64: continue
                    pts[i] = [na, nb]
                    pt_set.discard((a, b))
                    pt_set.add((na, nb))
                    new_edges = _edge_count(pts)
                    if new_edges > best_edges:
                        best_pts = [list(p) for p in pts]
                        best_edges = new_edges
                        improved = True
                        if best_edges >= OPT: break
                    else:
                        pts[i] = [a, b]
                        pt_set.discard((na, nb))
                        pt_set.add((a, b))
                if best_edges >= OPT: break
    else:
        # Simulated Annealing to minimize Inertia
        # Objective: Minimize Inertia, Constraint: Edges >= OPT (or == OPT)
        current_pts = [list(p) for p in best_pts]
        current_inertia = _moment_of_inertia(current_pts)
        best_sa_pts = [list(p) for p in current_pts]
        best_sa_inertia = current_inertia
        
        T = 10.0
        cooling = 0.999
        
        while time.time() < deadline:
            T *= cooling
            if T < 0.01: break
            
            i = random.randint(0, N-1)
            a, b = current_pts[i]
            # Random move: pick a random neighbor
            da, db = random.choice(directions)
            na, nb = a + da, b + db
            if abs(na) > 64 or abs(nb) > 64: continue
            
            pt_set = set(tuple(p) for p in current_pts)
            if (na, nb) in pt_set: continue
            
            # Tentative move
            current_pts[i] = [na, nb]
            pt_set.discard((a, b))
            pt_set.add((na, nb))
            
            new_edges = _edge_count(current_pts)
            new_inertia = _moment_of_inertia(current_pts)
            
            # Acceptance criteria
            if new_edges >= OPT:
                if new_inertia < best_sa_inertia:
                    best_sa_inertia = new_inertia
                    best_sa_pts = [list(p) for p in current_pts]
                
                # Accept if inertia improves, or with probability exp(-dE/T)
                d_inertia = new_inertia - current_inertia
                if d_inertia < 0 or random.random() < math.exp(-d_inertia / T):
                    current_inertia = new_inertia
                else:
                    # Revert
                    current_pts[i] = [a, b]
                    pt_set.discard((na, nb))
                    pt_set.add((a, b))
            else:
                # Reject if edges drop below OPT (unless we allow temporary drops? No, keep constraint)
                current_pts[i] = [a, b]
                pt_set.discard((na, nb))
                pt_set.add((a, b))
                
            if T < 0.01:
                break
                
        best_pts = best_sa_pts
        best_edges = _edge_count(best_pts)

    # 3. Final Centering
    if best_pts:
        sa = sum(p[0] for p in best_pts)
        sb = sum(p[1] for p in best_pts)
        ideal_a = -sa / N
        ideal_b = -sb / N
        
        best_shift = (0, 0)
        best_metric = float('inf')
        
        for da in range(int(ideal_a) - 2, int(ideal_a) + 3):
            for db in range(int(ideal_b) - 2, int(ideal_b) + 3):
                shifted = [[p[0] + da, p[1] + db] for p in best_pts]
                if any(abs(x) > 64 or abs(y) > 64 for x, y in shifted):
                    continue
                metric = sum(x*x + y*y for x, y in shifted)
                if metric < best_metric:
                    best_metric = metric
                    best_shift = (da, db)
        
        best_pts = [[p[0] + best_shift[0], p[1] + best_shift[1]] for p in best_pts]

    return best_pts
# EVOLVE-BLOCK-END
