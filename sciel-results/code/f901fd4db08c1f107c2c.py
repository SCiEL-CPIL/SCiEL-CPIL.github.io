import math
import time
import random

def _count_edges(points):
    """Count pairs at distance exactly 1 in the triangular lattice."""
    pts = set(points)
    count = 0
    for a, b in pts:
        # Six neighbors in triangular lattice
        for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
            if (a+da, b+db) in pts:
                count += 1
    return count // 2

def _generate_hex_cluster(n):
    """
    Generate a compact hexagonal cluster of n points.
    We generate points in order of increasing hexagonal distance from origin.
    """
    # Hexagonal distance d(a,b) = max(|a|, |b|, |a+b|) is not quite right for this basis.
    # For basis a, b with omega = e^{i pi/3}, the distance is sqrt(a^2 + ab + b^2).
    # We want to sort by this distance.
    
    # Max coordinate needed for n=32 is small. r=3 gives 37 points.
    # So coords in [-4, 4] are enough.
    max_coord = 10
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            dist_sq = a*a + a*b + b*b
            # Tie-breaking: use angle to form a spiral
            x = a + b * 0.5
            y = b * math.sqrt(3) / 2.0
            angle = math.atan2(y, x)
            candidates.append((dist_sq, angle, a, b))
    
    candidates.sort(key=lambda x: (x[0], x[1]))
    return [(a, b) for _, _, a, b in candidates[:n]]

def _local_search(points, max_time=10.0):
    """
    Perform local search to maximize edge count.
    Moves: Swap a point in the set with a point in the candidate neighborhood.
    """
    start_time = time.time()
    current_points = set(points)
    current_edges = _count_edges(current_points)
    best_points = set(current_points)
    best_edges = current_edges
    
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    # Candidate pool: points adjacent to the current set, or within a small radius
    # We can generate candidates on the fly
    
    while time.time() - start_time < max_time:
        improved = False
        
        # 1. Try to replace a point
        # To speed up, we can just iterate through all points in the set
        # and all potential new points in the bounding box
        
        # Bounding box of current set
        min_a = min(p[0] for p in current_points)
        max_a = max(p[0] for p in current_points)
        min_b = min(p[1] for p in current_points)
        max_b = max(p[1] for p in current_points)
        
        # Expand bounding box by 2 to find candidates
        cand_min_a = min_a - 2
        cand_max_a = max_a + 2
        cand_min_b = min_b - 2
        cand_max_b = max_b + 2
        
        candidates = set()
        for a in range(cand_min_a, cand_max_a + 1):
            for b in range(cand_min_b, cand_max_b + 1):
                candidates.add((a, b))
        
        candidates -= current_points
        
        # Try swapping each point in current_points with each candidate
        # This is O(N * M * Cost). N=32, M~100. Cost ~ 32. Total ~ 100k ops. Very fast.
        
        for p_out in list(current_points):
            for p_in in candidates:
                new_points = set(current_points)
                new_points.remove(p_out)
                new_points.add(p_in)
                
                # Quick check: if p_in is not adjacent to any point in new_points, it adds 0 edges.
                # But it might break edges from p_out.
                
                new_edges = _count_edges(new_points)
                if new_edges > current_edges:
                    current_points = new_points
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_points = set(current_points)
                    improved = True
                    # Restart candidate generation or break to re-evaluate?
                    # Usually better to break and restart loop with new set
                    break
            if improved:
                break
        
        if not improved:
            # No single swap improves. Try a random restart or 2-swap?
            # For n=32, 1-swap should be enough to reach optimum if started close.
            # If stuck, try a random perturbation: remove a random point, add a random candidate.
            if random.random() < 0.1:
                # Random restart
                p_out = random.choice(list(current_points))
                p_in = random.choice(list(candidates))
                new_points = set(current_points)
                new_points.remove(p_out)
                new_points.add(p_in)
                current_points = new_points
                current_edges = _count_edges(current_points)
            else:
                # Try 2-swap? Or just break?
                # Given the landscape is likely smooth for this problem, 
                # if 1-swap fails, we might be at a local optimum.
                # But 76 is the global max. 
                # Let's try a few random restarts of the whole process if stuck.
                break

    return list(best_points)

def construct():
    deadline = time.time() + 145.0
    
    # Generate initial hex cluster
    points = _generate_hex_cluster(32)
    
    # Perform local search
    # We can run multiple restarts if time permits
    best_points = points
    best_edges = _count_edges(points)
    
    restarts = 0
    while time.time() < deadline:
        # Try local search
        search_points = _local_search(points, max_time=5.0)
        search_edges = _count_edges(search_points)
        
        if search_edges > best_edges:
            best_edges = search_edges
            best_points = search_points
        
        if best_edges == 76:
            break
        
        # If not optimal, try a different initial configuration?
        # Or just restart local search with a perturbed initial state.
        restarts += 1
        if restarts > 20:
            break
            
        # Perturb initial points for next restart
        # Remove 2 random points, add 2 random candidates
        pts_set = set(points)
        pts_list = list(pts_set)
        for _ in range(2):
            if pts_list:
                p_out = pts_list.pop(random.randint(0, len(pts_list)-1))
                pts_set.remove(p_out)
                
                # Find a candidate
                candidates = set()
                min_a = min(p[0] for p in pts_set)
                max_a = max(p[0] for p in pts_set)
                min_b = min(p[1] for p in pts_set)
                max_b = max(p[1] for p in pts_set)
                for a in range(min_a-2, max_a+3):
                    for b in range(min_b-2, max_b+3):
                        if (a,b) not in pts_set:
                            candidates.add((a,b))
                if candidates:
                    p_in = random.choice(list(candidates))
                    pts_set.add(p_in)
        
        points = list(pts_set)
        
    return [[a, b] for a, b in best_points]
