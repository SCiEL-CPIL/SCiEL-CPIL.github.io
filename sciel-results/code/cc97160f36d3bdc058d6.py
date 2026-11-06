"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box, sorted by preferred orientation."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    # Sort: prefer lower z-extent (last dim) to stack better, then lower y, then lower x
    rots = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    return list(rots)

class _VoxelGrid:
    """
    Voxel-grid based occupancy tracker for O(1) overlap checks.
    Uses a 3D boolean array (flattened) to mark occupied voxels.
    """
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        # Use a flat bytearray for memory efficiency; 1 byte per voxel
        self.grid = bytearray(W * H * D)
        self.occupied = 0  # count of occupied voxels

    def _idx(self, x, y, z):
        return (z * self.H + y) * self.W + x

    def mark(self, x, y, z, w, h, d):
        """Mark voxels as occupied."""
        for iz in range(z, z + d):
            for iy in range(y, y + h):
                base = (iz * self.H + iy) * self.W + x
                for ix in range(w):
                    self.grid[base + ix] = 1
        self.occupied += w * h * d

    def unmark(self, x, y, z, w, h, d):
        """Unmark voxels."""
        for iz in range(z, z + d):
            for iy in range(y, y + h):
                base = (iz * self.H + iy) * self.W + x
                for ix in range(w):
                    self.grid[base + ix] = 0
        self.occupied -= w * h * d

    def check_fits(self, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with size (w,h,d) fits without overlap."""
        if x + w > self.W or y + h > self.H or z + d > self.D:
            return False
        for iz in range(z, z + d):
            for iy in range(y, y + h):
                base = (iz * self.H + iy) * self.W + x
                for ix in range(w):
                    if self.grid[base + ix]:
                        return False
        return True

    def filled_volume(self):
        """Return the total occupied voxel count."""
        return self.occupied


def _generate_candidates(placed, W, H, D, max_candidates=800):
    """
    Generate candidate positions from placed boxes using extreme points.
    Returns a list of candidate (x, y, z) positions.
    """
    candidates = set()
    candidates.add((0, 0, 0))
    
    for px, py, pz, pw, ph, pd in placed:
        # New extreme points from each placed box
        new_pts = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px + pw, py, pz + pd),
            (px, py + ph, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for pt in new_pts:
            if 0 <= pt[0] <= W and 0 <= pt[1] <= H and 0 <= pt[2] <= D:
                candidates.add(pt)
    
    # Convert to list and limit size
    cand_list = list(candidates)
    if len(cand_list) > max_candidates:
        # Sort by priority (closer to origin first)
        cand_list.sort(key=lambda c: c[0] + c[1] + c[2])
        cand_list = cand_list[:max_candidates]
    
    return cand_list

def _try_place(boxes, container, sort_key='volume', seed=0):
    """
    Attempt to place boxes into the container using an Extreme Point heuristic
    with voxel-grid based overlap checking.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Sort boxes based on the chosen key
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    if sort_key == 'volume':
        box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    elif sort_key == 'aspect_ratio':
        def aspect(i):
            w, h, d = boxes[i]
            dims = sorted([w, h, d])
            return dims[2] / max(dims[0], 1)
        box_indices.sort(key=aspect)
    elif sort_key == 'surface_area':
        def surf(i):
            w, h, d = boxes[i]
            return 2 * (w*h + h*d + w*d)
        box_indices.sort(key=surf, reverse=True)
    elif sort_key == 'max_dim':
        def maxd(i):
            w, h, d = boxes[i]
            return max(w, h, d)
        box_indices.sort(key=maxd, reverse=True)
    elif sort_key == 'min_dim':
        def mind(i):
            w, h, d = boxes[i]
            return min(w, h, d)
        box_indices.sort(key=mind, reverse=True)
    elif sort_key == 'random':
        random.seed(seed)
        random.shuffle(box_indices)
    
    # Initialize voxel grid
    grid = _VoxelGrid(W, H, D)
    # Placements: list of (x, y, z, w, h, d) for placed boxes
    placed = []
    # Result: for each original box index, store placement or None
    result = [None] * n
    
    for idx in box_indices:
        w_orig, h_orig, d_orig = boxes[idx]
        rots = box_rotations[idx]
        
        # Generate candidate positions
        candidates = _generate_candidates(placed, W, H, D)
        
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Try each rotation
        for rw, rh, rd in rots:
            # Try each candidate position
            for cx, cy, cz in candidates:
                # Use voxel grid for fast overlap check
                if not grid.check_fits(cx, cy, cz, rw, rh, rd):
                    continue
                
                # Score: prefer positions closer to origin (lower x, y, z)
                score = cx * 1.0 + cy * 1.0 + cz * 1.0
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            grid.mark(bx, by, bz, bw, bh, bd)
            placed.append((bx, by, bz, bw, bh, bd))
            result[idx] = (0, bx, by, bz, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
        
        # Check time
        if time.time() > deadline:
            # Fill remaining with -1
            for i in range(n):
                if result[i] is None:
                    w_i, h_i, d_i = boxes[i]
                    result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
            break
    
    # Fill any remaining None entries
    for i in range(n):
        if result[i] is None:
            w_i, h_i, d_i = boxes[i]
            result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
    
    return result

def _try_place_diverse(boxes, container):
    """
    Try multiple sorting strategies and return the best result.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin
    
    # Define different sorting strategies
    strategies = ['volume', 'aspect_ratio', 'surface_area', 'max_dim', 'min_dim']
    
    best_result = None
    best_filled = 0
    
    for strat_idx, strategy in enumerate(strategies):
        if time.time() > deadline:
            break
        
        # Place boxes using the extreme point heuristic
        result = _try_place(boxes, container, sort_key=strategy)
        
        # Calculate filled volume
        filled = 0
        for r in result:
            if r[0] == 0:
                filled += r[4] * r[5] * r[6]
        
        if filled > best_filled:
            best_filled = filled
            best_result = result
    
    # If we have time remaining, try random shuffles for additional diversity
    remaining_time = deadline - time.time()
    if remaining_time > 15.0 and best_result is not None:
        random.seed(42)
        for trial in range(5):
            if time.time() > deadline:
                break
            result = _try_place(boxes, container, sort_key='random', seed=42 + trial)
            filled = 0
            for r in result:
                if r[0] == 0:
                    filled += r[4] * r[5] * r[6]
            if filled > best_filled:
                best_filled = filled
                best_result = result
    
    if best_result is None:
        # Fallback: all boxes left out
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    return best_result

def _try_place_with_shake(boxes, container, seed=0):
    """
    Try to improve a placement by shaking (removing and re-placing) boxes
    using voxel-grid based overlap checking.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Get initial placement
    result = _try_place(boxes, container, sort_key='volume')
    
    # Calculate initial filled volume
    filled = sum(r[4] * r[5] * r[6] for r in result if r[0] == 0)
    
    # Build placed list from result
    placed = []
    box_to_placed_idx = {}
    for i, r in enumerate(result):
        if r[0] == 0:
            placed.append((r[1], r[2], r[3], r[4], r[5], r[6]))
            box_to_placed_idx[i] = len(placed) - 1
    
    # Initialize voxel grid with current placements
    grid = _VoxelGrid(W, H, D)
    for px, py, pz, pw, ph, pd in placed:
        grid.mark(px, py, pz, pw, ph, pd)
    
    # Precompute rotations
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Try to shake: remove a box and try to re-place it in a different position
    improved = True
    max_iterations = 50
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try each placed box in reverse order (larger boxes first)
        for i in range(n):
            if time.time() > deadline:
                break
            
            if result[i][0] != 0:
                continue
            
            # Remove this box from placed
            placed_idx = box_to_placed_idx.get(i)
            if placed_idx is None:
                continue
            
            bx, by, bz, bw, bh, bd = placed[placed_idx]
            grid.unmark(bx, by, bz, bw, bh, bd)
            placed.pop(placed_idx)
            
            # Update box_to_placed_idx for remaining boxes
            for k in list(box_to_placed_idx.keys()):
                if box_to_placed_idx[k] > placed_idx:
                    box_to_placed_idx[k] -= 1
            del box_to_placed_idx[i]
            
            # Try to re-place the box in a different position
            w_orig, h_orig, d_orig = boxes[i]
            rots = box_rotations[i]
            
            candidates = _generate_candidates(placed, W, H, D)
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for rw, rh, rd in rots:
                for cx, cy, cz in candidates:
                    # Use voxel grid for fast overlap check
                    if not grid.check_fits(cx, cy, cz, rw, rh, rd):
                        continue
                    
                    # Prefer different position than original
                    score = cx * 1.0 + cy * 1.0 + cz * 1.0
                    if (cx, cy, cz, rw, rh, rd) == (bx, by, bz, bw, bh, bd):
                        score += 1000  # Penalize same position
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx2, by2, bz2 = best_pos
                bw2, bh2, bd2 = best_rot
                grid.mark(bx2, by2, bz2, bw2, bh2, bd2)
                placed.append((bx2, by2, bz2, bw2, bh2, bd2))
                box_to_placed_idx[i] = len(placed) - 1
                result[i] = (0, bx2, by2, bz2, bw2, bh2, bd2)
                
                # Check if this led to improvement (we can't easily measure this, so just accept)
                improved = True
            else:
                # Put back in original position
                grid.mark(bx, by, bz, bw, bh, bd)
                placed.append((bx, by, bz, bw, bh, bd))
                box_to_placed_idx[i] = len(placed) - 1
                result[i] = (0, bx, by, bz, bw, bh, bd)
    
    return result

def load(boxes, container):
    """
    Place boxes into container. Returns list of (bin, x, y, z, w, h, d) for each box.
    bin=0 for loaded, bin=-1 for left out.
    """
    W, H, D = container
    
    # Try the diverse strategy which explores multiple sorting orders
    placements = _try_place_diverse(boxes, container)
    
    return placements
# EVOLVE-BLOCK-END
