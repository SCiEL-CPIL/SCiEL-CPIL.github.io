import time
import itertools

def load(boxes, container):
    start_time = time.time()
    # Set a safety deadline slightly below the 120s limit
    # The evaluator kills at 150s, but we want to return safely.
    deadline = start_time + 115.0
    
    W, H, D = container
    # Store original dimensions and index
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    # Sort by volume descending to place large boxes first
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    # List of placed boxes: (x, y, z, w, h, d)
    placed = []
    
    # Set of candidate positions (x, y, z)
    # We start with the origin
    candidates = set([(0, 0, 0)])
    
    # Initialize results with "left out" status (-1)
    # Format: (bin, x, y, z, w, h, d)
    results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    for idx, w, h, d in indexed_boxes:
        # Check time limit
        if time.time() > deadline:
            break
        
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Get unique rotations
        # Permutations of (w, h, d)
        rotations = list(set(itertools.permutations((w, h, d))))
        
        for rw, rh, rd in rotations:
            for cx, cy, cz in candidates:
                # Check if box fits in container bounds
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                
                # Check for overlap with all placed boxes
                valid = True
                for px, py, pz, pw, ph, pd in placed:
                    # Check intersection in X, Y, Z
                    if (cx < px + pw and px < cx + rw and
                        cy < py + ph and py < cy + rh and
                        cz < pz + pd and pz < cz + rd):
                        valid = False
                        break
                
                if valid:
                    # Heuristic Score:
                    # We want to minimize Z (height) first, then Y (depth/back), then X (width/left).
                    # This creates a "Bottom-Left-Back" packing.
                    # Using large multipliers ensures lexicographic ordering.
                    score = cz * 1000000 + cy * 1000 + cx
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            
            # Record placement
            placed.append((bx, by, bz, bw, bh, bd))
            results[idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Generate new candidate points
            # 1. On top of the placed box
            candidates.add((bx, by, bz + bd))
            # 2. To the right of the placed box
            candidates.add((bx + bw, by, bz))
            # 3. In front of the placed box
            candidates.add((bx, by + bh, bz))
            # 4. Corner (right and front) - often useful for filling gaps
            candidates.add((bx + bw, by + bh, bz))
            
            # Optional: Add (bx, by, bz) itself? It's already there.
            # Optional: Add (bx + bw, by, bz + bd)? 
            # The "Top" and "Right" and "Front" usually suffice to define the new boundaries.
            
    return results
