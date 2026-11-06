"""3D Container Loading using Best-Fit Decreasing with face-intersection candidate positions."""
import time
import heapq
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    W, H, D = container
    n = len(boxes)
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Generate all 6 rotations for each box, deduplicate
    def get_rotations(w, h, d):
        rots = set()
        for perm in permutations((w, h, d)):
            rots.add(perm)
        return list(rots)

    # Sort boxes by volume descending (best-fit decreasing)
    box_volumes = [(w * h * d, i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    box_volumes.sort(key=lambda x: -x[0])

    placements = [None] * n  # will fill with (bin, x, y, z, w, h, d)
    placed_boxes = []  # list of (x, y, z, w, h, d) for placed boxes

    # Candidate positions: start with origin
    # We'll maintain a set of candidate (x, y, z) positions
    # Initially just (0,0,0)
    candidates = [(0, 0, 0)]

    for vol, orig_idx, w, h, d in box_volumes:
        if time.time() > deadline:
            # Fill remaining as unplaced
            for i in range(n):
                if placements[i] is None:
                    bw, bh, bd = boxes[i]
                    placements[i] = (-1, 0, 0, 0, bw, bh, bd)
            return placements

        rots = get_rotations(w, h, d)
        best_pos = None
        best_score = None
        best_rot = None

        # For each rotation, try to find the best position
        for rw, rh, rd in rots:
            # Check if fits in container
            if rw > W or rh > H or rd > D:
                continue

            # Try all candidate positions
            for cx, cy, cz in candidates:
                # Check if box fits at this position
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue

                # Check overlap with all placed boxes
                overlap = False
                for px, py, pz, pw, ph, pd in placed_boxes:
                    # Check if boxes overlap
                    if (cx < px + pw and cx + rw > px and
                        cy < py + ph and cy + rh > py and
                        cz < pz + pd and cz + rd > pz):
                        overlap = True
                        break
                if overlap:
                    continue

                # Score: prefer lower z, then lower y, then lower x
                # Also prefer tighter fit (smaller wasted space)
                score = (cz, cy, cx, rw * rh * rd)
                # We want to minimize z, y, x
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)

        if best_pos is not None:
            bx, by, bz = best_pos
            rw, rh, rd = best_rot
            placements[orig_idx] = (0, bx, by, bz, rw, rh, rd)
            placed_boxes.append((bx, by, bz, rw, rh, rd))

            # Generate new candidate positions from this placement
            # New positions: on each face of the placed box
            new_candidates = [
                (bx + rw, by, bz),  # right face
                (bx, by + rh, bz),  # top face
                (bx, by, bz + rd),  # front face
                (bx + rw, by + rh, bz),
                (bx + rw, by, bz + rd),
                (bx, by + rh, bz + rd),
                (bx + rw, by + rh, bz + rd),
            ]
            for nc in new_candidates:
                if nc not in candidates:
                    candidates.append(nc)
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)

    # Final pass: ensure all are set
    for i in range(n):
        if placements[i] is None:
            bw, bh, bd = boxes[i]
            placements[i] = (-1, 0, 0, 0, bw, bh, bd)

    return placements
# EVOLVE-BLOCK-END
