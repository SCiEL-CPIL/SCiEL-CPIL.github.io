import numpy as np
import random
import time

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-compute valid rotations for each box to save time in the loop
    box_rotations = []
    for w, h, d in boxes:
        rots = set()
        for rw, rh, rd in [(w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)]:
            if rw <= W and rh <= H and rd <= D:
                rots.add((rw, rh, rd))
        box_rotations.append(list(rots))
        
    # Check if grid is feasible
    grid_size = W * H * D
    use_grid = grid_size < 50_000_000
    
    best_vol = -1
    best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
    
    start_time = time.time()
    max_time = 110.0
    
    # Increase restarts to explore more solutions
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        # Shuffle order
        order = list(range(n))
        random.shuffle(order)
        
        placements = [None] * n
        current_vol = 0
        
        # Initialize grid or fallback structures
        if use_grid:
            try:
                grid = np.zeros((D, H, W), dtype=bool)
            except MemoryError:
                use_grid = False
                grid = None
        else:
            grid = None
            
        candidates = set([(0, 0, 0)])
        
        for idx in order:
            rots = box_rotations[idx]
            if not rots:
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                continue
                
            placed = False
            # Sort candidates to prefer "bottom-left-front" (low x, low y, low z)
            # This helps pack into the origin corner.
            sorted_cands = sorted(candidates, key=lambda p: (p[0], p[1], p[2]))
            
            for cx, cy, cz in sorted_cands:
                if placed:
                    break
                
                for rw, rh, rd in rots:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    if use_grid:
                        # Check collision using numpy slicing
                        if not grid[cz:cz+rd, cy:cy+rh, cx:cx+rw].any():
                            # Place
                            grid[cz:cz+rd, cy:cy+rh, cx:cx+rw] = True
                            placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                            current_vol += rw * rh * rd
                            placed = True
                            
                            # Add new candidates (corners of the placed box)
                            new_cands = [
                                (cx + rw, cy, cz),
                                (cx, cy + rh, cz),
                                (cx, cy, cz + rd),
                                (cx + rw, cy + rh, cz),
                                (cx + rw, cy, cz + rd),
                                (cx, cy + rh, cz + rd),
                                (cx + rw, cy + rh, cz + rd)
                            ]
                            candidates.update(new_cands)
                            break
                    else:
                        # Fallback: Check against placed boxes (slower)
                        # This is rarely used
                        conflict = False
                        for p in placements:
                            if p is None or p[0] == -1:
                                continue
                            px, py, pz, pw, ph, pd = p[1], p[2], p[3], p[4], p[5], p[6]
                            # Check overlap
                            if not (cx + rw <= px or px + pw <= cx or
                                    cy + rh <= py or py + ph <= cy or
                                    cz + rd <= pz or pz + pd <= cz):
                                conflict = True
                                break
                        if not conflict:
                            placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                            current_vol += rw * rh * rd
                            placed = True
                            
                            new_cands = [
                                (cx + rw, cy, cz),
                                (cx, cy + rh, cz),
                                (cx, cy, cz + rd),
                                (cx + rw, cy + rh, cz),
                                (cx + rw, cy, cz + rd),
                                (cx, cy + rh, cz + rd),
                                (cx + rw, cy + rh, cz + rd)
                            ]
                            candidates.update(new_cands)
                            break
                        
            if not placed:
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                
        if current_vol > best_vol:
            best_vol = current_vol
            best_placements = placements
            
    return best_placements
