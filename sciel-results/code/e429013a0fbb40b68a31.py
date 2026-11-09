import time
import random

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    W, H, D = container
    n = len(boxes)
    
    # 1. Prepare boxes with indices and sort by volume descending
    # We keep original index to map back
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    placements = [None] * n
    placed_boxes = [] # List of (x, y, z, w, h, d)
    
    # Candidate points: list of (x, y, z)
    # We will maintain this list sorted by (z, y, x) to implement Bottom-Left-Far
    candidates = [(0, 0, 0)]
    
    def get_rotations(w, h, d):
        # Generate 6 unique permutations
        # Using a set to avoid duplicates if dims are equal
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)

    def fits(x, y, z, bw, bh, bd):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        # Check overlap with placed boxes
        for px, py, pz, pw, ph, pd in placed_boxes:
            # If they overlap in all 3 dimensions, it's a collision
            if not (x + bw <= px or px + pw <= x or 
                    y + bh <= py or py + ph <= y or 
                    z + bd <= pz or pz + pd <= z):
                return False
        return True

    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # If we run out of time, mark remaining as left out
            for j in range(i, n):
                # We need to know which original indices are remaining
                # This logic is tricky if we break early. 
                # Better to check time before starting each box.
                pass
            break

        best_pos = None
        best_rot = None
        
        # Sort candidates by (z, y, x) to find the "best" (lowest) position
        # Since we add new candidates, the list might not be sorted.
        # Sorting a small list (size < 200) is cheap.
        candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        rotations = get_rotations(w, h, d)
        
        found = False
        for cx, cy, cz in candidates:
            if time.time() > deadline:
                break
                
            for rw, rh, rd in rotations:
                if fits(cx, cy, cz, rw, rh, rd):
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
                    found = True
                    break
            if found:
                break
        
        if found:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            
            # Place the box
            placed_boxes.append((bx, by, bz, bw, bh, bd))
            placements[i] = (0, bx, by, bz, bw, bh, bd)
            
            # Update candidates
            # Remove the used candidate
            # Since we sorted, we can just remove the first occurrence of (bx, by, bz)
            # But there might be duplicates? We should avoid duplicates in candidates.
            # Using a set for candidates is better, but we need sorted iteration.
            # Let's use a set for unique candidates, and sort the list each time.
            
            # Add new corner candidates
            new_cands = [
                (bx + bw, by, bz),
                (bx, by + bh, bz),
                (bx, by, bz + bd)
            ]
            
            # Add to candidates set/list
            for nc in new_cands:
                if nc not in candidates:
                    candidates.append(nc)
            
            # Remove the used candidate
            # It's safe to remove one instance
            candidates.remove((bx, by, bz))
            
        else:
            placements[i] = (-1, 0, 0, 0, w, h, d)

    # Fill any remaining None (if loop broke early due to time)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)

    return placements
