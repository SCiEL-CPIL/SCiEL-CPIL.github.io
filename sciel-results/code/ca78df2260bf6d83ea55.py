"""3D Container Loading - Occupancy-grid packer with candidate-point refinement."""
import time
import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    t0 = time.time()
    DEADLINE = 110  # seconds, leave margin for safety

    n = len(boxes)
    if n == 0:
        return []

    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        orients = set(permutations([w, h, d]))
        return list(orients)

    box_orientations = [get_orientations(boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]

    # Sort boxes by volume descending
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: -box_volumes[i])

    # We'll use a list-based approach for overlap checking but with candidate points
    # sorted by (z, y, x) to find the "lowest" position first.
    # To speed up, we'll maintain a list of placed boxes and check against them.
    # For better performance, we can use a 3D grid if dimensions are small,
    # but since we don't know the scale, we'll stick with list-based but optimize the search.

    def try_place(boxes_order, t0, deadline):
        """Try to place boxes in the given order. Returns (placements, filled_volume)."""
        placed_boxes = []  # (x, y, z, w, h, d, idx)
        candidates = [(0, 0, 0)]
        placements = [None] * n
        filled_vol = 0

        for idx in boxes_order:
            if time.time() - t0 > deadline:
                break

            w, h, d = boxes[idx]
            orientations = box_orientations[idx]

            best_pos = None
            best_score = float('inf')

            # Sort candidates by (z, y, x) to prioritize lower positions
            # This is a heuristic for "lowest point" which tends to pack well
            sorted_candidates = sorted(candidates, key=lambda c: (c[2], c[1], c[0]))

            found = False
            for cx, cy, cz in sorted_candidates:
                if found:
                    break
                for ow, oh, od in orientations:
                    # Check if this orientation fits in the container
                    if ow > W or oh > H or od > D:
                        continue

                    # Check if the box fits within the container bounds
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue

                    # Check overlap with placed boxes
                    overlap = False
                    for px, py, pz, pw, ph, pd, _ in placed_boxes:
                        if (cx < px + pw and px < cx + ow and
                            cy < py + ph and py < cy + oh and
                            cz < pz + pd and pz < cz + od):
                            overlap = True
                            break
                    if overlap:
                        continue

                    # Valid placement found
                    best_pos = (cx, cy, cz, ow, oh, od)
                    found = True
                    break

            if best_pos is not None:
                bx, by, bz, bw, bh, bd = best_pos
                placements[idx] = (0, bx, by, bz, bw, bh, bd)
                placed_boxes.append((bx, by, bz, bw, bh, bd, idx))
                filled_vol += bw * bh * bd

                # Generate new candidate positions
                # The 8 corners of the placed box
                new_candidates = [
                    (bx, by, bz),
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx + bw, by + bh, bz),
                    (bx, by, bz + bd),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                ]
                for nc in new_candidates:
                    # Only add if within bounds (optional optimization)
                    if nc[0] <= W and nc[1] <= H and nc[2] <= D:
                        candidates.append(nc)
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)

        # Fill in any remaining boxes
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)

        return placements, filled_vol

    # First pass: standard best-fit decreasing
    best_placements, best_vol = try_place(box_indices, t0, DEADLINE)

    # If we have time left, try a few random permutations to see if we can do better
    # This is a simple local search / simulated annealing lite
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5:  # Only if we have enough time
        import random
        random.seed(42)  # Fixed seed for reproducibility
        
        # Try a few random orderings
        for attempt in range(10):
            if time.time() - t0 > DEADLINE - 2:
                break
            
            # Create a perturbed ordering: mostly sorted by volume, but with some random swaps
            order = box_indices[:]
            # Randomly swap a few pairs
            for _ in range(5):
                i, j = random.randint(0, n-1), random.randint(0, n-1)
                if i != j:
                    order[i], order[j] = order[j], order[i]
            
            placements, vol = try_place(order, t0, DEADLINE)
            
            if vol > best_vol:
                best_vol = vol
                best_placements = placements

    return best_placements
