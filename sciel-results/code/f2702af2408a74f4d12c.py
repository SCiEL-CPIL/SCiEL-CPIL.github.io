import time
import random
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (evaluator kills at 150s)
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Pre-compute all unique rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        box_rotations.append(list(rots))
    
    # Volume of each box (invariant under rotation)
    box_volumes = [w * h * d for w, h, d in boxes]
    
    def solve_with_order(order):
        """
        Greedy placement with candidate corner generation.
        Returns (placements, vol).
        placements is a list of (bin, x, y, z, w, h, d) per box index.
        """
        placed = []  # list of (x, y, z, w, h, d)
        placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
        candidates = {(0, 0, 0)}
        
        # Track current occupancy in a grid-like structure for fast overlap check?
        # For 60 boxes, O(N) overlap check is fine (60*60*6*60 = ~1.3M ops per solve, times 1000 solves = 1.3B... maybe too slow?)
        # Let's optimize can_place.
        
        for i in order:
            box_id = i
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for rot in box_rotations[box_id]:
                rw, rh, rd = rot
                # Quick bounds check
                if rw > W or rh > H or rd > D:
                    continue
                    
                for cx, cy, cz in candidates:
                    # Bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    # Overlap check
                    overlap = False
                    for px, py, pz, pw, ph, pd in placed:
                        # Disjoint if:
                        # x ranges don't overlap OR y ranges don't overlap OR z ranges don't overlap
                        if not (cx + rw <= px or px + pw <= cx or cy + rh <= py or py + ph <= cy or cz + rd <= pz or pz + pd <= cz):
                            overlap = True
                            break
                    
                    if not overlap:
                        # Heuristic: prefer positions that minimize the "cost"
                        # Cost = x + y + z (pack towards origin)
                        # Or maybe cost = x*H*D + y*W*D + z*W*H (weighted by volume impact?)
                        # Simple sum is usually good for corner-based.
                        score = cx + cy + cz
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = rot
            
            if best_pos:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed.append((bx, by, bz, bw, bh, bd))
                placements[box_id] = (0, bx, by, bz, bw, bh, bd)
                
                # Generate new candidates
                # The new box occupies [bx, bx+bw], [by, by+bh], [bz, bz+bd]
                # New corners are exposed faces.
                # (bx+bw, by, bz), (bx+bw, by+bh, bz), (bx+bw, by, bz+bd), (bx+bw, by+bh, bz+bd)
                # (bx, by+bh, bz), (bx, by+bh, bz+bd)
                # (bx, by, bz+bd)
                
                new_candidates = [
                    (bx + bw, by, bz),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                    (bx, by + bh, bz),
                    (bx, by + bh, bz + bd),
                    (bx, by, bz + bd)
                ]
                
                for nc in new_candidates:
                    if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                        candidates.add(nc)
            else:
                # Cannot place this box
                pass
        
        vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        return placements, vol

    # --- Search Strategies ---
    
    best_placements = None
    best_vol = 0
    
    indices = list(range(n))
    
    # 1. Volume Descending (Large First)
    order_vol = sorted(indices, key=lambda i: box_volumes[i], reverse=True)
    p, v = solve_with_order(order_vol)
    if v > best_vol:
        best_vol = v
        best_placements = p
    
    # 2. Max Dimension Descending
    order_max = sorted(indices, key=lambda i: max(boxes[i]), reverse=True)
    p, v = solve_with_order(order_max)
    if v > best_vol:
        best_vol = v
        best_placements = p
    
    # 3. Min Dimension Descending (Flat boxes first?)
    order_min = sorted(indices, key=lambda i: min(boxes[i]), reverse=True)
    p, v = solve_with_order(order_min)
    if v > best_vol:
        best_vol = v
        best_placements = p
        
    # 4. Random Restarts
    # Use remaining time for random restarts.
    # Each solve is O(N^2 * R * C). N=60, R=6, C~300. ~648k ops.
    # Python is slow, so ~1ms per solve? Maybe 5ms.
    # 100 seconds -> 20,000 restarts? Let's cap at 2000 to be safe and see timing.
    
    restarts_done = 0
    # Adaptive restart count based on time
    while time.time() < deadline - 5.0:
        random.shuffle(indices)
        p, v = solve_with_order(indices)
        if v > best_vol:
            best_vol = v
            best_placements = p
        restarts_done += 1
        
        # Check if we should stop early if we found a perfect pack?
        if best_vol == container_volume:
            break

    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
        
    return best_placements
