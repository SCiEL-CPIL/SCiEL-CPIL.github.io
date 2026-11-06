"""Improved program for 3D Container Loading using Multi-Start Max-Stack Greedy."""
import numpy as np
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    # Precompute all valid orientations for each box
    box_options = []
    for idx, (w, h, d) in enumerate(boxes):
        perms = set()
        perms.add((w, h, d))
        perms.add((w, d, h))
        perms.add((h, w, d))
        perms.add((h, d, w))
        perms.add((d, w, h))
        perms.add((d, h, w))
        
        valid_perms = []
        for p in perms:
            if p[0] <= W and p[1] <= H and p[2] <= D:
                valid_perms.append(p)
        
        if not valid_perms:
            valid_perms = [(w, h, d)]
            
        box_options.append(valid_perms)
    
    # Helper function to check if a box fits at (x, y, z) with dims (bw, bh, bd)
    def fits(placed, x, y, z, bw, bh, bd):
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + bw <= px or px + pw <= x):
                if not (y + bh <= py or py + ph <= y):
                    if not (z + bd <= pz or pz + pd <= z):
                        return False
        return True

    # Helper function to perform greedy placement given a sorted list of indices and a sorting key
    def greedy_place(sorted_indices):
        placed = []
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        total_volume = 0
        
        for idx in sorted_indices:
            found = False
            perms = box_options[idx]
            
            # Generate candidate coordinates
            # Max-Stack strategy:
            # 1. Potential Z levels: 0 and top of every placed box
            # 2. Potential Y levels: 0 and side of every placed box
            # 3. Potential X levels: 0 and side of every placed box
            
            z_candidates = {0}
            y_candidates = {0}
            x_candidates = {0}
            
            for (px, py, pz, pw, ph, pd) in placed:
                z_candidates.add(pz + pd) # Top of box
                y_candidates.add(py + ph) # Side of box
                x_candidates.add(px + pw) # Side of box
            
            # Sort candidates to prioritize lowest values (Max-Stack: Low Z, then Low Y, then Low X)
            z_candidates = sorted(z_candidates)
            y_candidates = sorted(y_candidates)
            x_candidates = sorted(x_candidates)
            
            # We want to find the position with the lowest Z.
            # If multiple have same Z, lowest Y. If same Y, lowest X.
            # So we iterate Z outer, Y middle, X inner.
            
            for z in z_candidates:
                # If min depth of box doesn't fit, break
                min_d = min(p[2] for p in perms)
                if z + min_d > D:
                    break
                
                for y in y_candidates:
                    min_h = min(p[1] for p in perms)
                    if y + min_h > H:
                        break
                    
                    for x in x_candidates:
                        min_w = min(p[0] for p in perms)
                        if x + min_w > W:
                            break
                        
                        # Try orientations
                        # Prefer orientations that keep the box "compact" or maybe just first fit?
                        # Let's try all orientations and pick the one that fits.
                        # To make it deterministic and fast, we can just pick the first valid one.
                        # Or we can pick the one that minimizes the "bounding box" expansion?
                        # For now, first valid orientation.
                        
                        valid_orientation = None
                        for (bw, bh, bd) in perms:
                            if fits(placed, x, y, z, bw, bh, bd):
                                valid_orientation = (bw, bh, bd)
                                break
                        
                        if valid_orientation:
                            bw, bh, bd = valid_orientation
                            placed.append((x, y, z, bw, bh, bd))
                            results[idx] = (0, x, y, z, bw, bh, bd)
                            total_volume += bw * bh * bd
                            found = True
                            break
                    
                    if found:
                        break
                
                if found:
                    break
            
        return results, total_volume

    # Define different sorting strategies
    # 1. Volume Desc
    # 2. Max Dimension Desc
    # 3. Height Desc
    # 4. Width Desc
    # 5. Depth Desc
    # 6. Volume Asc (sometimes helps to fill gaps? Unlikely for Max-Stack, but let's try)
    
    strategies = []
    
    # Strategy 1: Volume Desc
    indices = list(range(n))
    indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    strategies.append(indices)
    
    # Strategy 2: Max Dim Desc
    indices = list(range(n))
    indices.sort(key=lambda i: max(boxes[i]), reverse=True)
    strategies.append(indices)
    
    # Strategy 3: Height Desc
    indices = list(range(n))
    indices.sort(key=lambda i: boxes[i][1], reverse=True)
    strategies.append(indices)
    
    # Strategy 4: Width Desc
    indices = list(range(n))
    indices.sort(key=lambda i: boxes[i][0], reverse=True)
    strategies.append(indices)
    
    # Strategy 5: Depth Desc
    indices = list(range(n))
    indices.sort(key=lambda i: boxes[i][2], reverse=True)
    strategies.append(indices)
    
    # Strategy 6: Volume Desc, then Height Desc
    indices = list(range(n))
    indices.sort(key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2], boxes[i][1]), reverse=True)
    strategies.append(indices)
    
    best_results = None
    best_volume = -1
    
    for strat in strategies:
        res, vol = greedy_place(strat)
        if vol > best_volume:
            best_volume = vol
            best_results = res
            
    if best_results is None:
        # Fallback: no box could be placed?
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results
# EVOLVE-BLOCK-END
