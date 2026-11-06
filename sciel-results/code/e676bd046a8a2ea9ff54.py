"""
Strategy: Coordinate-Descent Geometry Optimization for 3D Container Loading.

Instead of searching for a discrete ordering of boxes, this strategy:
1. Uses a fast greedy heuristic (Bottom-Left-Back-Farthest) to find an initial
   feasible placement for a subset of boxes.
2. Treats the coordinates (x, y, z) of each placed box as continuous variables.
3. Minimizes an energy function defined as the sum of pairwise overlap volumes
   and wall violation volumes using L-BFGS-B.
4. Snaps the resulting coordinates to integers and verifies feasibility.
5. Iterates this process with different greedy seeds and box subsets to improve
   the total packed volume.

This is a fundamentally different design family: it optimizes the geometry of
the packing directly in continuous space rather than searching the discrete
space of box orderings.
"""

import time
import itertools
import random
import numpy as np
from scipy.optimize import minimize

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    deadline = time.time() + 110.0

    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]

    def greedy_place(indices, order=None):
        """
        Greedy placement of a subset of boxes.
        Returns a list of (idx, x, y, z, w, h, d) for placed boxes.
        """
        if order is None:
            order = indices
        placed = []
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = orient_cache[idx]
            best_placement = None
            best_score = None

            for (bw, bh, bd) in orients:
                # Try positions based on "Bottom-Left-Back-Farthest" heuristic
                # We generate candidate positions from existing placed boxes
                candidate_positions = [(0, 0, 0)]
                for (_, px, py, pz, pw, ph, pd) in placed:
                    candidate_positions.extend([
                        (px + pw, py, pz),
                        (px, py + ph, pz),
                        (px, py, pz + pd),
                        (px + pw, py + ph, pz),
                        (px, py + ph, pz + pd),
                        (px + pw, py, pz + pd),
                        (px + pw, py + ph, pz + pd),
                    ])
                
                for (px, py, pz) in candidate_positions:
                    if px < 0 or py < 0 or pz < 0:
                        continue
                    if px + bw > W or py + bh > H or pz + bd > D:
                        continue
                    # Check overlap
                    overlap = False
                    for (_, ox, oy, oz, ow, oh, od) in placed:
                        if px < ox + ow and px + bw > ox and py < oy + oh and py + bh > oy and pz < oz + od and pz + bd > oz:
                            overlap = True
                            break
                    if not overlap:
                        # Score: prefer lower z, then lower y, then lower x
                        score = (pz, py, px)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_placement = (px, py, pz, bw, bh, bd)
            
            if best_placement is not None:
                bx, by, bz, bw, bh, bd = best_placement
                placed.append((idx, bx, by, bz, bw, bh, bd))
        
        return placed

    def energy_and_grad(x, placed_info, W, H, D):
        """
        Compute energy (sum of overlaps and wall violations) and its gradient.
        x: flattened array of [x0, y0, z0, x1, y1, z1, ...]
        placed_info: list of (idx, w, h, d) for each box in the placement
        """
        n_boxes = len(placed_info)
        energy = 0.0
        grad = np.zeros_like(x)

        # Wall violations
        for i in range(n_boxes):
            w, h, d = placed_info[i][1], placed_info[i][2], placed_info[i][3]
            xi, yi, zi = x[3*i], x[3*i+1], x[3*i+2]
            
            # Overlap with walls
            # If xi < 0, overlap width is -xi. Energy += (-xi) * h * d. Grad_xi += -h*d
            if xi < 0:
                energy += (-xi) * h * d
                grad[3*i] += -h * d
            if xi + w > W:
                energy += (xi + w - W) * h * d
                grad[3*i] += h * d
                
            if yi < 0:
                energy += (-yi) * w * d
                grad[3*i+1] += -w * d
            if yi + h > H:
                energy += (yi + h - H) * w * d
                grad[3*i+1] += w * d
                
            if zi < 0:
                energy += (-zi) * w * h
                grad[3*i+2] += -w * h
            if zi + d > D:
                energy += (zi + d - D) * w * h
                grad[3*i+2] += w * h

        # Pairwise overlaps
        for i in range(n_boxes):
            wi, hi, di = placed_info[i][1], placed_info[i][2], placed_info[i][3]
            xi, yi, zi = x[3*i], x[3*i+1], x[3*i+2]
            for j in range(i + 1, n_boxes):
                wj, hj, dj = placed_info[j][1], placed_info[j][2], placed_info[j][3]
                xj, yj, zj = x[3*j], x[3*j+1], x[3*j+2]
                
                # Overlap in x
                ox = min(xi + wi, xj + wj) - max(xi, xj)
                if ox <= 0:
                    continue
                # Overlap in y
                oy = min(yi + hi, yj + hj) - max(yi, yj)
                if oy <= 0:
                    continue
                # Overlap in z
                oz = min(zi + di, zj + dj) - max(zi, zj)
                if oz <= 0:
                    continue
                
                energy += ox * oy * oz
                
                # Gradients
                # d(ox)/dx_i = 1 if xi + wi < xj + wj else -1 (if max(xi,xj) is xi)
                # Actually, simpler:
                # If xi < xj: ox = min(xi+wi, xj+wj) - xj. d/ox/dxi = 1 if xi+wi < xj+wj else 0.
                # Let's use the standard subgradient approach.
                # ox = min(xi+wi, xj+wj) - max(xi, xj)
                
                # Contribution to grad[3*i] (xi)
                if xi < xj:
                    # max is xj, min is min(xi+wi, xj+wj)
                    if xi + wi < xj + wj:
                        grad[3*i] += oy * oz
                    else:
                        grad[3*i] += 0
                else:
                    # max is xi, min is min(xi+wi, xj+wj)
                    if xi + wi < xj + wj:
                        grad[3*i] += -oy * oz
                    else:
                        grad[3*i] += 0
                        
                # Contribution to grad[3*j] (xj)
                if xj < xi:
                    if xj + wj < xi + wi:
                        grad[3*j] += oy * oz
                    else:
                        grad[3*j] += 0
                else:
                    if xj + wj < xi + wi:
                        grad[3*j] += -oy * oz
                    else:
                        grad[3*j] += 0

                # Similarly for y and z
                if yi < yj:
                    if yi + hi < yj + hj:
                        grad[3*i+1] += ox * oz
                    else:
                        grad[3*i+1] += 0
                else:
                    if yi + hi < yj + hj:
                        grad[3*i+1] += -ox * oz
                    else:
                        grad[3*i+1] += 0

                if yj < yi:
                    if yj + hj < yi + hi:
                        grad[3*j+1] += ox * oz
                    else:
                        grad[3*j+1] += 0
                else:
                    if yj + hj < yi + hi:
                        grad[3*j+1] += -ox * oz
                    else:
                        grad[3*j+1] += 0

                if zi < zj:
                    if zi + di < zj + dj:
                        grad[3*i+2] += ox * oy
                    else:
                        grad[3*i+2] += 0
                else:
                    if zi + di < zj + dj:
                        grad[3*i+2] += -ox * oy
                    else:
                        grad[3*i+2] += 0

                if zj < zi:
                    if zj + dj < zi + di:
                        grad[3*j+2] += ox * oy
                    else:
                        grad[3*j+2] += 0
                else:
                    if zj + dj < zi + di:
                        grad[3*j+2] += -ox * oy
                    else:
                        grad[3*j+2] += 0

        return energy, grad

    def optimize_placement(placed):
        """
        Takes a list of placed boxes (idx, x, y, z, w, h, d), optimizes their
        coordinates to remove overlaps, and returns the optimized integer
        coordinates if feasible, else None.
        """
        if not placed:
            return []
            
        n_boxes = len(placed)
        x0 = np.array([p[1] for p in placed] + [p[2] for p in placed] + [p[3] for p in placed])
        
        # Prepare info for energy function
        placed_info = [(p[0], p[4], p[5], p[6]) for p in placed]
        
        # Bounds: [0, W-w_i] for x, etc.
        bounds = []
        for p in placed:
            w, h, d = p[4], p[5], p[6]
            bounds.append((0, W - w))
            bounds.append((0, H - h))
            bounds.append((0, D - d))
            
        # L-BFGS-B
        try:
            res = minimize(
                lambda x: energy_and_grad(x, placed_info, W, H, D)[0],
                x0,
                jac=lambda x: energy_and_grad(x, placed_info, W, H, D)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4}
            )
        except:
            return None
            
        x_opt = res.x
        # Snap to integers
        x_int = np.round(x_opt).astype(int)
        
        # Reconstruct placements
        new_placed = []
        for i, p in enumerate(placed):
            idx, x, y, z, w, h, d = p
            new_placed.append((idx, x_int[3*i], x_int[3*i+1], x_int[3*i+2], w, h, d))
            
        # Verify feasibility
        for i in range(len(new_placed)):
            _, xi, yi, zi, wi, hi, di = new_placed[i]
            if xi < 0 or yi < 0 or zi < 0:
                return None
            if xi + wi > W or yi + hi > H or zi + di > D:
                return None
            for j in range(i + 1, len(new_placed)):
                _, xj, yj, zj, wj, hj, dj = new_placed[j]
                if xi < xj + wj and xi + wi > xj and yi < yj + hj and yi + hi > yj and zi < zj + dj and zi + di > zj:
                    return None
        return new_placed

    # Main Loop: Try multiple greedy seeds and subsets
    best_result = [None] * n
    best_volume = 0
    best_placed = []

    # Heuristic orders
    indexed = list(range(n))
    base_orders = [
        sorted(indexed, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True),
        sorted(indexed, key=lambda i: max(boxes[i]), reverse=True),
        sorted(indexed, key=lambda i: (boxes[i][0]+boxes[i][1]+boxes[i][2]), reverse=True),
    ]

    for order in base_orders:
        if time.time() > deadline:
            break
            
        # Try full set
        placed = greedy_place(indexed, order)
        if placed:
            optimized = optimize_placement(placed)
            if optimized:
                vol = sum(p[4]*p[5]*p[6] for p in optimized)
                if vol > best_volume:
                    best_volume = vol
                    best_placed = optimized

        # Try subsets (drop the smallest boxes to allow optimization to succeed)
        # We'll try dropping the last K boxes from the sorted order
        for k in [5, 10, 15]:
            if time.time() > deadline:
                break
            subset_indices = order[:-k]
            placed = greedy_place(subset_indices, subset_indices)
            if placed:
                optimized = optimize_placement(placed)
                if optimized:
                    vol = sum(p[4]*p[5]*p[6] for p in optimized)
                    if vol > best_volume:
                        best_volume = vol
                        best_placed = optimized

    # Randomized restarts
    for _ in range(30):
        if time.time() > deadline:
            break
        rand_order = base_orders[0][:]
        random.shuffle(rand_order)
        placed = greedy_place(indexed, rand_order)
        if placed:
            optimized = optimize_placement(placed)
            if optimized:
                vol = sum(p[4]*p[5]*p[6] for p in optimized)
                if vol > best_volume:
                    best_volume = vol
                    best_placed = optimized

    # Finalize result
    if best_placed:
        for p in best_placed:
            idx, x, y, z, w, h, d = p
            best_result[idx] = (0, x, y, z, w, h, d)
        for i in range(n):
            if best_result[i] is None:
                best_result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    else:
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    return best_result
