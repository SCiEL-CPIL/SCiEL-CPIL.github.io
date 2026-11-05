import math

def construct():
    n = 32
    
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2

    def is_neighbor(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        da = a1 - a2
        db = b1 - b2
        return (abs(da) == 1 and db == 0) or \
               (da == 0 and abs(db) == 1) or \
               (da == 1 and db == -1) or \
               (da == -1 and db == 1)

    selected_points = [(0, 0)]
    
    # Search space: points within hex distance 5 from origin
    # 32 points will fit well within this
    max_k = 5
    candidates = []
    for a in range(-max_k, max_k + 1):
        for b in range(-max_k, max_k + 1):
            if hex_dist(a, b) <= max_k:
                candidates.append((a, b))
                
    # Remove origin from candidates as it's already selected
    candidates.remove((0, 0))
    
    while len(selected_points) < n:
        best_point = None
        best_score = -1
        best_dist = float('inf')
        
        for p in candidates:
            # Check if p is at distance >= 1 from all selected points
            valid = True
            neighbor_count = 0
            for sp in selected_points:
                if is_neighbor(p, sp):
                    neighbor_count += 1
                elif hex_dist(p[0] - sp[0], p[1] - sp[1]) < 1:
                    # This check is redundant if is_neighbor covers dist 1 and hex_dist covers dist 0
                    # But we need to ensure dist >= 1.
                    # Dist 0 means same point. We already removed selected points?
                    # No, we don't remove from candidates, we just check validity.
                    # If p is in selected_points, hex_dist is 0.
                    # But we don't add p to selected_points until end of loop.
                    # So we just need to check if p is already in selected_points?
                    # No, p is from candidates. If p was selected in a previous step,
                    # we should have removed it from candidates or marked it.
                    # Better to maintain a set of selected points for O(1) lookup?
                    # Or just check if p is in selected_points.
                    pass
            
            # Simpler validity check:
            # p must not be in selected_points (dist 0)
            # p must not be a neighbor? No, neighbors are allowed (dist 1).
            # p must not be at dist < 1. Only dist 0 is < 1.
            # So we just need to ensure p is not in selected_points.
            
            if p in selected_points:
                continue
                
            # Count neighbors
            for sp in selected_points:
                if is_neighbor(p, sp):
                    neighbor_count += 1
            
            # Score: prefer more neighbors, then closer to origin
            dist_from_origin = hex_dist(p[0], p[1])
            
            if neighbor_count > best_score or (neighbor_count == best_score and dist_from_origin < best_dist):
                best_score = neighbor_count
                best_dist = dist_from_origin
                best_point = p
                
        if best_point is None:
            break
            
        selected_points.append(best_point)
        
    return [[a, b] for a, b in selected_points]
