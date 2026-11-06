"""No-isosceles set construction for 64x64 grid."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # Safety margin before 330s kill
    
    def sq_dist(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    
    def can_add(point, selected, dist_cache):
        """Check if point can be added without creating isosceles triangle."""
        px, py = point
        for i, a in enumerate(selected):
            ax, ay = a
            d_pa = (px - ax)**2 + (py - ay)**2
            for j in range(i + 1, len(selected)):
                b = selected[j]
                bx, by = b
                d_pb = (px - bx)**2 + (py - by)**2
                d_ab = dist_cache.get((i, j))
                if d_ab is None:
                    d_ab = (ax - bx)**2 + (ay - by)**2
                    dist_cache[(i, j)] = d_ab
                # Check all three conditions for isosceles
                if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                    return False
        return True
    
    def greedy_with_order(order, selected, dist_cache):
        """Greedy add points in given order."""
        for p in order:
            if can_add(p, selected, dist_cache):
                selected.append(p)
                # Update dist_cache for new point
                for i, a in enumerate(selected[:-1]):
                    dist_cache[(i, len(selected) - 1)] = sq_dist(a, p)
        return len(selected)
    
    def local_search(selected, dist_cache, time_limit):
        """Try to improve by removing and re-adding points."""
        current_size = len(selected)
        selected_set = set(selected)
        
        # Precompute all grid points not in selected
        available = [(x, y) for x in range(N) for y in range(N) if (x, y) not in selected_set]
        
        # Try removing one point and adding up to two new ones
        for _ in range(100):  # Limit iterations
            if time.time() > time_limit:
                break
            if not selected:
                break
            
            # Try each point for removal
            for remove_idx in range(len(selected)):
                if time.time() > time_limit:
                    break
                removed = selected[remove_idx]
                new_selected = selected[:remove_idx] + selected[remove_idx+1:]
                new_selected_set = set(new_selected)
                
                # Rebuild dist_cache for new_selected
                new_dist_cache = {}
                for i in range(len(new_selected)):
                    for j in range(i + 1, len(new_selected)):
                        new_dist_cache[(i, j)] = sq_dist(new_selected[i], new_selected[j])
                
                # Try to add points
                added_count = 0
                new_available = available[:]
                random.shuffle(new_available)
                
                for p in new_available:
                    if (p[0], p[1]) in new_selected_set:
                        continue
                    if can_add(p, new_selected, new_dist_cache):
                        new_selected.append(p)
                        new_selected_set.add(p)
                        new_dist_cache[(len(new_selected) - 2, len(new_selected) - 1)] = sq_dist(new_selected[-2], p)
                        for k in range(len(new_selected) - 2):
                            new_dist_cache[(k, len(new_selected) - 1)] = sq_dist(new_selected[k], p)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if len(new_selected) > current_size:
                    selected = new_selected
                    dist_cache = new_dist_cache
                    current_size = len(selected)
                    selected_set = set(selected)
                    # Update available
                    available = [(x, y) for x in range(N) for y in range(N) if (x, y) not in selected_set]
                    break
        
        return selected, dist_cache, current_size
    
    # Generate all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_selected = []
    best_dist_cache = {}
    best_size = 0
    
    # Multiple random restarts
    num_restarts = 50
    random.seed(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:  # Leave time for final polish
            break
        
        # Create a random order
        order = all_points[:]
        random.shuffle(order)
        
        # Greedy construction
        selected = []
        dist_cache = {}
        size = greedy_with_order(order, selected, dist_cache)
        
        # Local search improvement
        if size > best_size and time.time() < deadline - 60:
            selected, dist_cache, size = local_search(selected, dist_cache, deadline - 30)
        
        if size > best_size:
            best_selected = selected
            best_dist_cache = dist_cache
            best_size = size
    
    # Final: try to add any remaining points
    if time.time() < deadline - 5:
        selected_set = set(best_selected)
        for p in all_points:
            if p not in selected_set and can_add(p, best_selected, best_dist_cache):
                best_selected.append(p)
                selected_set.add(p)
                for i, a in enumerate(best_selected[:-1]):
                    best_dist_cache[(i, len(best_selected) - 1)] = sq_dist(a, p)
    
    return best_selected
# EVOLVE-BLOCK-END
