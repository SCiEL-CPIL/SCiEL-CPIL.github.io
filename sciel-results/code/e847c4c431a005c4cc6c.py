"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Generate hexagonal spiral points
    # Layer 0: (0,0)
    # Layer k has 6k points.
    # We generate points layer by layer.
    
    points = [(0,0)]
    
    # Pre-define layers up to k=5 to be safe
    # k=1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # k=2: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1)
    # k=3: (3,0), (2,1), (1,2), (0,3), (-1,3), (-2,3), (-3,3), (-3,2), (-3,1), (-3,0), (-2,-1), (-1,-2), (0,-3), (1,-3), (2,-3), (3,-3), (3,-2), (3,-1)
    # k=4: (4,0), (3,1), (2,2), (1,3), (0,4), (-1,4), (-2,4), (-3,4), (-4,4), (-4,3), (-4,2), (-4,1), (-4,0), (-3,-1), (-2,-2), (-1,-3), (0,-4), (1,-4), (2,-4), (3,-4), (4,-4), (4,-3), (4,-2), (4,-1)
    
    # Function to generate layer k
    def gen_layer(k):
        if k == 0:
            return []
        pts = []
        # Top-right edge: (k, 0) to (0, k) ? No, spiral direction.
        # Standard spiral:
        # (k, 0)
        # (k-1, 1) ... (0, k)
        # (-1, k) ... (-k, k)
        # (-k, k-1) ... (-k, 0)
        # (-k+1, -1) ... (0, -k)
        # (1, -k) ... (k, -k)
        # (k, -k+1) ... (k, -1)
        
        # Let's verify k=1:
        # (1,0)
        # (0,1)
        # (-1,1)
        # (-1,0)
        # (0,-1)
        # (1,-1)
        # Matches.
        
        # k=2:
        # (2,0)
        # (1,1), (0,2)
        # (-1,2), (-2,2)
        # (-2,1), (-2,0)
        # (-1,-1), (0,-2)
        # (1,-2), (2,-2)
        # (2,-1)
        # Matches.
        
        pts.append((k, 0))
        for i in range(k-1, -1, -1):
            pts.append((i, k-i)) # (k-1, 1) to (0, k)
        for i in range(-1, -k-1, -1):
            pts.append((i, k-i)) # (-1, k) to (-k, k) -- Wait, i goes -1 to -k. k-i is k+1 to 2k? No.
            # Let's re-derive.
            # Corner 1: (k,0)
            # Edge 1: (k-1, 1) to (0, k). a goes k-1 to 0, b goes 1 to k. a+b=k.
            # Corner 2: (0, k)
            # Edge 2: (-1, k) to (-k, k). a goes -1 to -k, b is k.
            # Corner 3: (-k, k)
            # Edge 3: (-k, k-1) to (-k, 0). a is -k, b goes k-1 to 0.
            # Corner 4: (-k, 0)
            # Edge 4: (-k+1, -1) to (0, -k). a goes -k+1 to 0, b goes -1 to -k. a-b = k-1? No.
            # Let's check k=1. (-1, -1) is not in layer 1.
            # My previous manual list for k=1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
            # Let's check k=2 manual list: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1).
            
            # Let's just hardcode the generation logic for k up to 5 or use the explicit lists.
            # Given the small size, explicit lists are safer.
            
        return pts

    # Since I already have the lists in the previous thought, I will just use them.
    layers = [
        [], # k=0
        [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)],
        [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1)],
        [(3,0), (2,1), (1,2), (0,3), (-1,3), (-2,3), (-3,3), (-3,2), (-3,1), (-3,0), (-2,-1), (-1,-2), (0,-3), (1,-3), (2,-3), (3,-3), (3,-2), (3,-1)],
        [(4,0), (3,1), (2,2), (1,3), (0,4), (-1,4), (-2,4), (-3,4), (-4,4), (-4,3), (-4,2), (-4,1), (-4,0), (-3,-1), (-2,-2), (-1,-3), (0,-4), (1,-4), (2,-4), (3,-4), (4,-4), (4,-3), (4,-2), (4,-1)]
    ]
    
    for k in range(1, 5):
        points.extend(layers[k])
        
    # We have 1 + 6 + 12 + 18 + 24 = 61 points.
    # Optimal 32 points are within H_4 (first 37 points).
    
    def count_edges(pts):
        pts_set = set(pts)
        edges = 0
        for p in pts:
            a, b = p
            neighbors = [
                (a+1, b), (a, b+1), (a-1, b-1),
                (a-1, b), (a, b-1), (a+1, b-1)
            ]
            for n in neighbors:
                if n in pts_set:
                    edges += 1
        return edges // 2

    # Initial solution: first 32 points
    initial_solution = points[:32]
    best_solution = set(initial_solution)
    best_edges = count_edges(list(best_solution))
    
    # Candidate pool: first 40 points (includes all H_4)
    candidate_pool = set(points[:40])
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p_in in list(best_solution):
            if time.time() > deadline:
                break
            for p_out in (candidate_pool - best_solution):
                new_set = best_solution - {p_in} | {p_out}
                new_edges = count_edges(list(new_set))
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_solution = new_set
                    improved = True
                    break # Restart outer loop to re-evaluate all swaps with new set
                elif new_edges == best_edges:
                    # Optional: prefer smaller coordinates? Not needed.
                    pass
    
    result = [[a, b] for a, b in best_solution]
    
    # Verify constraints
    if len(result) != 32:
        result = [[a, b] for a, b in initial_solution]
        
    return result
# EVOLVE-BLOCK-END
