"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import permutations

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples, ordered by volume impact."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    # Sort by smallest dimension first to try "flatter" orientations first
    result.sort(key=lambda r: (r[2], r[1], r[0]))
    return result

def _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
    """Check if a box of size (bw,bh,bd) fits at (x,y,z) without overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        # Separation check: if not separated in any axis, they overlap
        if not (x + bw <= px or px + pw <= x):
            if not (y + bh <= py or py + ph <= y):
                if not (z + bd <= pz or pz + pd <= z):
                    return False
    return True

def _generate_candidates(x, y, z, bw, bh, bd, placed, W, H, D):
    """
    Generate candidate positions using the 'floor' and 'wall' points.
    The best position is typically at the lowest possible z, then lowest y, then lowest x.
    """
    candidates = set()
    
    # Always try the origin
    candidates.add((0, 0, 0))
    
    # For each placed box, generate points "on top of" it, "beside" it in x, and "beside" it in y
    for (px, py, pz, pw, ph, pd) in placed:
        # Points on the floor (z=0)
        # x = px + pw (to the right)
        candidates.add((px + pw, py, 0))
        candidates.add((px, py + ph, 0))
        candidates.add((px + pw, py + ph, 0))
        
        # Points "on top of" the box (z = pz + pd)
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        
        # Points "beside" in x but at the same z level (sliding along z)
        # This is often not needed if we check all z levels, but helps with packing
        # candidates.add((px + pw, py, pz))
        
        # Points "beside" in y but at the same z level
        # candidates.add((px, py + ph, pz))

    # Filter and sort candidates
    valid_candidates = []
    for (cx, cy, cz) in candidates:
        if cx < 0 or cy < 0 or cz < 0:
            continue
        if cx + bw > W or cy + bh > H or cz + bd > D:
            continue
        if _fits_at(cx, cy, cz, bw, bh, bd, placed, W, H, D):
            valid_candidates.append((cx, cy, cz))
            
    # Sort by z, then y, then x to prioritize lower positions
    valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
    return valid_candidates

def _solve_instance(boxes, container, seed, time_limit):
    """
    Solve a single instance using a greedy heuristic with limited backtracking/search.
    """
    W, H, D = container
    random.seed(seed)
    
    # Pre-compute rotations for each box
    box_rots = []
    for w, h, d in boxes:
        rots = _rotations(w, h, d)
        box_rots.append(rots)
    
    # Sort boxes by volume descending, with some randomness to break ties
    # This helps pack large items first
    indices = list(range(len(boxes)))
    indices.sort(key=lambda i: -boxes[i][0]*boxes[i][1]*boxes[i][2])
    
    # We will try multiple random shuffles of the sorted indices
    # But for now, let's stick to a deterministic greedy with the best candidate position
    
    best_placements = None
    best_volume = 0
    
    # Try a few different sorting strategies or random perturbations
    # Strategy 1: Volume descending
    # Strategy 2: Max dimension descending
    # Strategy 3: Random
    
    strategies = [
        sorted(indices, key=lambda i: -boxes[i][0]*boxes[i][1]*boxes[i][2]),
        sorted(indices, key=lambda i: -max(boxes[i])),
        sorted(indices, key=lambda i: -min(boxes[i])),
        indices # original order
    ]
    
    # Add a few random shuffles
    for _ in range(5):
        shuffled = indices.copy()
        random.shuffle(shuffled)
        strategies.append(shuffled)
    
    start_time = time.time()
    
    for strategy in strategies:
        if time.time() - start_time > time_limit:
            break
            
        placed = []
        placements = [None] * len(boxes)
        current_volume = 0
        
        for idx in strategy:
            w, h, d = boxes[idx]
            rots = box_rots[idx]
            
            placed_idx = False
            for rot in rots:
                bw, bh, bd = rot
                candidates = _generate_candidates(0, 0, 0, bw, bh, bd, placed, W, H, D)
                
                if candidates:
                    # Pick the first valid candidate (lowest z, y, x)
                    cx, cy, cz = candidates[0]
                    placed.append((cx, cy, cz, bw, bh, bd))
                    placements[idx] = (0, cx, cy, cz, bw, bh, bd)
                    current_volume += bw * bh * bd
                    placed_idx = True
                    break
            
            if not placed_idx:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        if current_volume > best_volume:
            best_volume = current_volume
            best_placements = [p if p is not None else (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i, p in enumerate(placements)]
            
    if best_placements is None:
        # Fallback: all left out
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_placements

def load(boxes, container):
    """
    Main entry point.
    """
    start_time = time.time()
    # Use a deadline of 110s to be safe
    deadline = 110.0
    
    # Try multiple seeds to find a better packing
    best_placements = None
    best_volume = 0
    
    # We can try a few different random seeds
    # Since the problem is deterministic given the seed, we can just use seed 0, 1, 2...
    # But the strategies inside _solve_instance already vary.
    # Let's just call _solve_instance with a few different seeds if we have time.
    
    # For now, just one robust call with seed 0
    # If we have time, we can add more.
    
    # The time limit per instance should be small if we run multiple, 
    # but we only have one instance per call.
    # So we can use most of the time for one instance.
    
    placements = _solve_instance(boxes, container, seed=0, time_limit=deadline - (time.time() - start_time))
    
    return placements

# EVOLVE-BLOCK-END
