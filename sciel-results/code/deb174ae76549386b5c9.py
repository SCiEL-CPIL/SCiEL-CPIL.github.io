"""Program using 3D occupancy grid (voxel map) for fast placement checks."""
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    vol = W * H * D
    if vol <= 0:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    # Use a finer voxel grid for better precision, but cap total voxels
    max_voxels = 5_000_000  # ~40 MB for a uint8 array
    # Target ~200 voxels per unit dimension, capped
    def _res(dim, target_max=200, min_res=1):
        if dim <= 0:
            return 1
        n = max(min_res, int(np.ceil(dim * 2)))  # 2 voxels per unit
        if n > target_max:
            n = target_max
        return n

    nx = _res(W)
    ny = _res(H)
    nz = _res(D)

    # If total voxels too big, reduce the largest dimension
    while nx * ny * nz > max_voxels and max(nx, ny, nz) > 1:
        if nx >= ny and nx >= nz:
            nx = max(1, nx - 1)
        elif ny >= nx and ny >= nz:
            ny = max(1, ny - 1)
        else:
            nz = max(1, nz - 1)

    # Scale factors: how many units per voxel
    sx = W / nx if nx > 0 else 1.0
    sy = H / ny if ny > 0 else 1.0
    sz = D / nz if nz > 0 else 1.0

    # Occupancy grid
    occ = np.zeros((nx, ny, nz), dtype=np.uint8)

    def _box_to_vox(x, y, z, w, h, d):
        """Convert a box at (x,y,z) with size (w,h,d) to voxel indices."""
        x0 = int(np.floor(x / sx))
        y0 = int(np.floor(y / sy))
        z0 = int(np.floor(z / sz))
        # End positions (exclusive in voxel space)
        x1 = int(np.ceil((x + w) / sx))
        y1 = int(np.ceil((y + h) / sy))
        z1 = int(np.ceil((z + d) / sz))
        # Clamp
        x0 = max(0, min(nx - 1, x0))
        y0 = max(0, min(ny - 1, y0))
        z0 = max(0, min(nz - 1, z0))
        x1 = max(x0 + 1, min(nx, x1))
        y1 = max(y0 + 1, min(ny, y1))
        z1 = max(z0 + 1, min(nz, z1))
        return x0, y0, z0, x1, y1, z1

    def _can_place(x, y, z, w, h, d):
        """Check if box fits in free space using the voxel grid."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W + 1e-9 or y + h > H + 1e-9 or z + d > D + 1e-9:
            return False
        x0, y0, z0, x1, y1, z1 = _box_to_vox(x, y, z, w, h, d)
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            return False
        # Check if any voxel in the region is occupied
        region = occ[x0:x1, y0:y1, z0:z1]
        return not np.any(region)

    def _place(x, y, z, w, h, d):
        """Mark the box's voxels as occupied."""
        x0, y0, z0, x1, y1, z1 = _box_to_vox(x, y, z, w, h, d)
        occ[x0:x1, y0:y1, z0:z1] = 1

    def _unplace(x, y, z, w, h, d):
        """Mark the box's voxels as free (for backtracking if needed)."""
        x0, y0, z0, x1, y1, z1 = _box_to_vox(x, y, z, w, h, d)
        occ[x0:x1, y0:y1, z0:z1] = 0

    # Sort boxes: largest first to fill big gaps early
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: (-(t[1][0] * t[1][1] * t[1][2]), -max(t[1])))

    placements = [None] * len(boxes)
    placed_count = 0

    # Track placed box extents to generate better candidate positions
    placed_x_extents = set()
    placed_y_extents = set()
    placed_z_extents = set()

    def _candidate_positions(w, h, d, max_candidates=500):
        """Generate candidate (x,y,z) positions for a box of size (w,h,d)."""
        candidates = set()
        
        # Base positions: 0 and max
        xs = [0.0, max(0.0, W - w)]
        ys = [0.0, max(0.0, H - h)]
        zs = [0.0, max(0.0, D - d)]
        
        # Add positions based on placed box extents
        for ex in placed_x_extents:
            for offset in [0.0, -w]:
                x = ex + offset
                if 0 <= x <= W - w + 1e-9:
                    xs.append(x)
        for ey in placed_y_extents:
            for offset in [0.0, -h]:
                y = ey + offset
                if 0 <= y <= H - h + 1e-9:
                    ys.append(y)
        for ez in placed_z_extents:
            for offset in [0.0, -d]:
                z = ez + offset
                if 0 <= z <= D - d + 1e-9:
                    zs.append(z)
        
        # Add some intermediate positions
        if W > w * 2:
            xs.append(W / 2.0 - w / 2.0)
            xs.append(W / 4.0)
            xs.append(3.0 * W / 4.0 - w / 4.0)
        if H > h * 2:
            ys.append(H / 2.0 - h / 2.0)
            ys.append(H / 4.0)
            ys.append(3.0 * H / 4.0 - h / 4.0)
        if D > d * 2:
            zs.append(D / 2.0 - d / 2.0)
            zs.append(D / 4.0)
            zs.append(3.0 * D / 4.0 - d / 4.0)
        
        # Generate all combinations
        for x in xs:
            for y in ys:
                for z in zs:
                    if 0 <= x <= W - w + 1e-9 and 0 <= y <= H - h + 1e-9 and 0 <= z <= D - d + 1e-9:
                        candidates.add((round(x, 6), round(y, 6), round(z, 6)))
        
        # Limit candidates
        cand_list = list(candidates)
        if len(cand_list) > max_candidates:
            # Sort by z (prefer lower z), then y, then x
            cand_list.sort(key=lambda p: (p[2], p[1], p[0]))
            cand_list = cand_list[:max_candidates]
        return cand_list

    # For each box in sorted order, try to place it
    for idx, (w, h, d) in indexed:
        # All 6 rotations (permutations of dimensions)
        rotations = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w),
        ]
        # Remove duplicates
        seen = set()
        unique_rotations = []
        for r in rotations:
            if r not in seen:
                seen.add(r)
                unique_rotations.append(r)

        best_pos = None
        best_rot = None
        best_score = -1

        for rw, rh, rd in unique_rotations:
            # Try candidate positions
            cands = _candidate_positions(rw, rh, rd, max_candidates=300)
            for (cx, cy, cz) in cands:
                if _can_place(cx, cy, cz, rw, rh, rd):
                    # Score: prefer lower z, then lower y, then lower x (fill from bottom)
                    # Also prefer positions that are more "compact" (closer to origin)
                    score = -(cz * 10000 + cy * 100 + cx)
                    if score > best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)

        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placements[idx] = (0, bx, by, bz, bw, bh, bd)
            _place(bx, by, bz, bw, bh, bd)
            placed_count += 1
            # Update extents
            placed_x_extents.add(bx + bw)
            placed_y_extents.add(by + bh)
            placed_z_extents.add(bz + bd)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
