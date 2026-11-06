"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import heapq

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Precompute unique orientations for each box
    # Orientation is a tuple (w, h, d) representing dimensions along x, y, z
    box_orientations = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            orients.add(perm)
        box_orientations.append(list(orients))
    
    # We will use a "Corner List" or "Free Space" based approach.
    # A simpler and effective heuristic for 3D bin packing is the "Next Fit" or "First Fit" 
    # with a list of available "corners" (bottom-left-front positions).
    # We maintain a list of candidate positions (x, y, z) where a new box can be placed.
    # Initially, the only corner is (0, 0, 0).
    # When a box is placed at (x, y, z) with dimensions (w, h, d), we add new corners:
    # (x+w, y, z), (x, y+h, z), (x, y, z+d)
    # We then check if these corners are inside the container and if a box can fit there.
    # To avoid duplicates and invalid corners, we can prune the list.
    
    def try_pack(order_indices, time_limit):
        """
        Attempt to pack boxes in the given order using the corner list method.
        Returns (placements, total_volume).
        """
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        total_volume = 0
        # List of (x, y, z) corners. We use a list, and we can sort by x, then y, then z to prefer bottom-left-front.
        corners = [(0, 0, 0)]
        
        # To make it faster, we can keep the corners sorted.
        # However, inserting into a sorted list is O(N). 
        # Given N=60, the number of corners won't be huge.
        
        for idx in order_indices:
            if time.time() > time_limit:
                break
            
            placed = False
            w_orig, h_orig, d_orig = boxes[idx]
            orients = box_orientations[idx]
            
            # Try each corner
            # We should try corners in a specific order to fill bottom-left-front first.
            # Let's sort the current corners by (x, y, z) before trying.
            # But sorting every iteration is expensive.
            # Instead, we can just iterate through the existing corners.
            # To ensure we fill "front" (low z), "left" (low x), "bottom" (low y),
            # we should prioritize corners with smaller coordinates.
            
            # Optimization: Instead of checking all corners for all boxes,
            # we can maintain a min-heap of corners based on (z, y, x) or similar?
            # But corners are added dynamically.
            
            # Let's stick to a simple list iteration, but we need to be careful about which corner to pick if multiple fit.
            # The "best" corner is usually the one that minimizes (x, y, z) in some lexicographic order, 
            # or the one that is "most filled" (i.e., closest to the bottom-left-front).
            
            # Let's try all corners and pick the one that results in the "tightest" fit?
            # Or just the first one that fits?
            # "First Fit" with corners sorted by (z, y, x) is a good heuristic.
            
            # Since we can't easily sort the list of corners efficiently on every insertion,
            # we can just iterate through them. To bias towards bottom-left-front,
            # we can iterate through the list and check if a box fits.
            # If we add new corners to the end, they might be "further away".
            # So we should probably check older corners first?
            
            # Let's try a different strategy:
            # Maintain a list of free spaces? No, corner list is standard.
            
            # Let's sort the corners list before each box placement to ensure we try the best corner first.
            # N is small (60 boxes), so number of corners is at most 3*60 = 180.
            # Sorting 180 items 60 times is 60 * 180 * log(180) ~ 60 * 180 * 8 ~ 86,400 ops. Very fast.
            
            corners.sort(key=lambda c: (c[2], c[1], c[0])) # Sort by z, then y, then x
            
            for cx, cy, cz in corners:
                if placed:
                    break
                for ow, oh, od in orients:
                    if cx + ow <= W and cy + oh <= H and cz + od <= D:
                        # Place the box
                        placements[idx] = (0, cx, cy, cz, ow, oh, od)
                        total_volume += ow * oh * od
                        placed = True
                        
                        # Add new corners
                        new_corners = [
                            (cx + ow, cy, cz),
                            (cx, cy + oh, cz),
                            (cx, cy, cz + od)
                        ]
                        
                        # Add only if they are inside the container
                        for nc in new_corners:
                            if nc[0] <= W and nc[1] <= H and nc[2] <= D:
                                # Check if this corner is already in the list?
                                # To avoid duplicates, we could use a set, but then we lose order.
                                # Duplicates in the corner list just mean we check the same position twice.
                                # It's not a correctness issue, just a minor perf hit.
                                # Given the small size, we can just append.
                                corners.append(nc)
                        break
            
        return placements, total_volume

    # Generate different ordering strategies
    indexed_boxes = list(range(n))
    
    # Heuristic 1: Volume descending
    order1 = sorted(indexed_boxes, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Heuristic 2: Max dimension descending
    order2 = sorted(indexed_boxes, key=lambda i: max(boxes[i]), reverse=True)
    
    # Heuristic 3: Sum of dimensions descending
    order3 = sorted(indexed_boxes, key=lambda i: sum(boxes[i]), reverse=True)
    
    # Heuristic 4: Height (Y-axis in standard orientation? No, any axis) 
    # Let's sort by the largest dimension of the box, which is order2.
    
    # Heuristic 5: Sort by "difficulty" - boxes with fewer orientations first?
    # All boxes have 1, 2, or 6 orientations.
    # Boxes with all distinct dims have 6. Boxes with two equal have 3. Boxes with all equal have 1.
    # It's often better to place "harder" to fit boxes (more symmetric?) or "easier" ones?
    # Usually, larger/more constrained first.
    
    # Let's also try a "Greedy" approach where we pick the box that fits in the most "tight" spot?
    # That's more complex. Let's stick to static orderings first.
    
    # Also, we should consider the container dimensions.
    # If W is the smallest, maybe we should prioritize boxes with small X-dimension?
    # This depends on the specific instance.
    
    strategies = [
        order1,
        order2,
        order3,
        # Try ascending? Sometimes small boxes fill gaps? 
        # With corner list, small boxes might get placed in tight spots, leaving large gaps for big boxes.
        # So descending is usually better.
    ]
    
    best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    best_volume = 0
    
    # Run each strategy
    # We split the time budget among strategies.
    # Let's give each strategy 1/3 of the time, leaving some buffer.
    # Or just run them sequentially and stop if we hit the deadline.
    
    for order in strategies:
        if time.time() > deadline:
            break
        
        # Give each strategy a sub-deadline
        sub_deadline = min(deadline, time.time() + (deadline - time.time()) / 3.0)
        # Actually, just use the main deadline. If one strategy is fast, we can try more.
        # But if one is slow, it might eat all the time.
        # Let's just run them and check the global deadline inside try_pack.
        
        p, v = try_pack(order, deadline)
        
        if v > best_volume:
            best_volume = v
            best_placements = p
    
    # If we have time left, we can try a local search or a different heuristic.
    # But for now, this should be a solid improvement over the naive shelf pack.
    
    return best_placements
# EVOLVE-BLOCK-END
