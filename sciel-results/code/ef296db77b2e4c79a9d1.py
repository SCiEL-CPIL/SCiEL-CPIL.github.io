"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _try_pack(boxes, W, H, D, rng, deadline):
    """
    Attempts to pack boxes using a randomized greedy best-fit decreasing strategy.
    Returns a list of placements for the given boxes.
    """
    # 1. Sort boxes by volume descending, with a small random perturbation for tie-breaking
    #    and to explore different orderings in multi-start.
    indexed_boxes = list(enumerate(boxes))
    # Add a tiny random noise to volume to break ties differently each restart
    volumes = [w*h*d for w,h,d in boxes]
    noisy_volumes = [v + rng.uniform(0, 0.01 * max(1, max(volumes))) for v in volumes]
    indexed_boxes.sort(key=lambda x: noisy_volumes[x[0]], reverse=True)
    
    placed = [] # List of (x, y, z, w, h, d, orig_index)
    results = [None] * len(boxes)
    
    # Candidate points for placement corners
    # Start with origin
    candidate_points = set([(0, 0, 0)])
    
    total_volume = 0
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
            
        # Generate unique orientations
        box_dims = sorted([w, h, d], reverse=True)
        seen_orientations = set()
        orientations = []
        for perm in itertools.permutations(box_dims):
            if perm not in seen_orientations:
                seen_orientations.add(perm)
                orientations.append(perm)
                
        # Shuffle orientations to randomize which one is picked if scores are close
        rng.shuffle(orientations)

        best_pos = None
        best_score = float('inf')
        best_orient = None
        found_placement = False
        
        # To speed up, we can limit the number of candidate points checked if there are too many,
        # but for N=60, 3*N+1 is small. We check all.
        
        # Randomly select a subset of candidate points if the set grows large?
        # For now, check all.
        
        for cx, cy, cz in candidate_points:
            if time.time() > deadline:
                break
                
            for ow, oh, od in orientations:
                # Check bounds
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                
                # Check overlap with all placed boxes
                overlap = False
                for px, py, pz, pw, ph, pd, _ in placed:
                    # Separation check
                    if not (cx + ow <= px or px + pw <= cx or 
                            cy + oh <= py or py + ph <= cy or 
                            cz + od <= pz or pz + pd <= cz):
                        overlap = True
                        break
                
                if not overlap:
                    # Scoring function:
                    # Primary: Minimize max extent (keeps bounding box tight)
                    # Secondary: Minimize sum of coordinates (packs towards origin)
                    # Tertiary: Random tie-breaker to encourage diversity in multi-start
                    
                    extent_score = max(cx + ow, cy + oh, cz + od)
                    sum_score = cx + cy + cz
                    
                    # Combine into a single score. 
                    # We want to minimize extent, but also fill space densely.
                    # A common heuristic is to minimize the "wasted space" around the box,
                    # but minimizing max extent is a good proxy for keeping the cluster compact.
                    
                    # Let's use a weighted sum to allow some flexibility
                    score = extent_score * 1.0 + sum_score * 0.1 + rng.uniform(0, 0.5)
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
                        found_placement = True
        
        if found_placement:
            bx, by, bz = best_pos
            bw, bh, bd = best_orient
            
            # Place the box
            placed.append((bx, by, bz, bw, bh, bd, orig_idx))
            results[orig_idx] = (0, bx, by, bz, bw, bh, bd)
            total_volume += bw * bh * bd
            
            # Generate new candidate positions
            new_candidates = [
                (bx + bw, by, bz),
                (bx, by + bh, bz),
                (bx, by, bz + bd)
            ]
            
            for nc in new_candidates:
                if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                    candidate_points.add(nc)
        else:
            results[orig_idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining results
    for i in range(len(boxes)):
        if results[i] is None:
            w, h, d = boxes[i]
            results[i] = (-1, 0, 0, 0, w, h, d)
            
    return results, total_volume

def load(boxes, container):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We should stop well before.
    # Let's aim to finish within 140s to be safe.
    deadline = start_time + 135.0
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_volume = W * H * D
    if container_volume == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    best_result = None
    best_volume = -1
    
    # Multi-start strategy
    # Run multiple independent greedy attempts with different random seeds
    # Each attempt uses a slightly different ordering and tie-breaking.
    
    # Determine how many restarts we can afford.
    # A single run takes T seconds. We can do roughly (deadline - start_time) / T runs.
    # We don't know T exactly, so we will run as many as we can in a loop,
    # checking time before each new attempt.
    
    attempt_num = 0
    while time.time() < deadline - 5.0: # Leave 5s buffer for final return
        attempt_num += 1
        # Use a deterministic seed for reproducibility in evaluation, 
        # but different for each attempt to ensure diversity.
        seed = 42 + attempt_num
        rng = random.Random(seed)
        
        res, vol = _try_pack(boxes, W, H, D, rng, deadline)
        
        if vol > best_volume:
            best_volume = vol
            best_result = res
            
        # Early stopping if we've packed everything
        if best_volume == sum(w*h*d for w,h,d in boxes):
            break
            
        # If we are running out of time, break
        if time.time() > deadline - 10.0:
            break
            
    if best_result is None:
        # Fallback if no attempt completed or all failed
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_result
# EVOLVE-BLOCK-END
