"""3D Container Loading - Skyline + Shelf hybrid approach."""
import time
import random
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


def _rotations_sorted(w, h, d):
    """Return unique rotations sorted by max-dim ascending (more compact first)."""
    rots = _rotations(w, h, d)
    rots.sort(key=lambda r: (max(r), r[1], r[2]))
    return rots


class Skyline3D:
    """
    3D Skyline representation for fast box placement.
    Maintains a set of 'skyline' points that represent the top surface.
    A box can be placed at (x, y, z) if z >= max height of all skyline points
    in the region [x, x+bw) x [y, y+bh).
    """
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        # Skyline points: list of (x, y, z) where z is the current top height
        # at that (x,y) location. We use a sparse representation.
        self.skyline = []  # list of (x, y, z)
        self.placed_boxes = []  # list of (x, y, z, w, h, d)
    
    def can_place(self, x, y, z, bw, bh, bd):
        """Check if a box of size (bw, bh, bd) fits at (x, y, z)."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > self.W or y + bh > self.H or z + bd > self.D:
            return False
        # Check against all placed boxes for overlap
        for (px, py, pz, pw, ph, pd) in self.placed_boxes:
            if not (x + bw <= px or px + pw <= x):
                if not (y + bh <= py or py + ph <= y):
                    if not (z + bd <= pz or pz + pd <= z):
                        return False
        return True
    
    def place(self, x, y, z, bw, bh, bd):
        """Place a box and update the skyline."""
        self.placed_boxes.append((x, y, z, bw, bh, bd))
        # Update skyline: add new top points
        self.skyline.append((x, y, z + bd))
        self.skyline.append((x + bw, y, z + bd))
        self.skyline.append((x, y + bh, z + bd))
        self.skyline.append((x + bw, y + bh, z + bd))
    
    def get_candidate_positions(self, bw, bh, bd):
        """Generate candidate (x, y, z) positions for a box of given size."""
        candidates = set()
        # Origin
        candidates.add((0, 0, 0))
        
        if not self.placed_boxes:
            return [(0, 0, 0)]
        
        # Generate positions based on existing box boundaries
        xs = {0}
        ys = {0}
        zs = {0}
        
        for (px, py, pz, pw, ph, pd) in self.placed_boxes:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        
        # Filter valid coordinates
        xs = sorted(set(x for x in xs if 0 <= x <= self.W - bw))
        ys = sorted(set(y for y in ys if 0 <= y <= self.H - bh))
        zs = sorted(set(z for z in zs if 0 <= z <= self.D - bd))
        
        # Generate combinations - limit to avoid explosion
        # Use a subset: for each placed box, try positions adjacent to it
        for (px, py, pz, pw, ph, pd) in self.placed_boxes:
            # Right of box
            if px + pw + bw <= self.W:
                candidates.add((px + pw, py, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            # Left of box (if box not at x=0)
            if px >= bw:
                candidates.add((px - bw, py, pz))
                candidates.add((px - bw, py + ph, pz))
            # Above box
            if pz + pd + bd <= self.D:
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            # In front/behind in y
            if py + ph + bh <= self.H:
                candidates.add((px, py + ph, pz))
                candidates.add((px + pw, py + ph, pz))
            if py >= bh:
                candidates.add((px, py - bh, pz))
                candidates.add((px + pw, py - bh, pz))
        
        # Also try some grid-based positions for diversity
        if len(xs) <= 15 and len(ys) <= 15:
            for x in xs[:10]:
                for y in ys[:10]:
                    # Find minimum z that works
                    min_z = 0
                    for (px, py, pz, pw, ph, pd) in self.placed_boxes:
                        if not (x + bw <= px or px + pw <= x):
                            if not (y + bh <= py or py + ph <= y):
                                if pz + pd > min_z:
                                    min_z = pz + pd
                    if min_z + bd <= self.D:
                        candidates.add((x, y, min_z))
        
        # Filter candidates
        valid = []
        for (cx, cy, cz) in candidates:
            if 0 <= cx and cx + bw <= self.W and 0 <= cy and cy + bh <= self.H and 0 <= cz and cz + bd <= self.D:
                valid.append((cx, cy, cz))
        
        return valid
    
    def best_position(self, bw, bh, bd, scoring_fn=None):
        """Find the best position for a box. Returns (x, y, z) or None."""
        candidates = self.get_candidate_positions(bw, bh, bd)
        best = None
        best_score = None
        
        for (cx, cy, cz) in candidates:
            if self.can_place(cx, cy, cz, bw, bh, bd):
                if scoring_fn:
                    s = scoring_fn(cx, cy, cz, bw, bh, bd)
                else:
                    # Default: prefer lower z, then lower y, then lower x
                    s = (cz, cy, cx)
                if best_score is None or s < best_score:
                    best_score = s
                    best = (cx, cy, cz)
        
        return best


class ShelfPacker:
    """
    Shelf-based 3D packing.
    Divides the container into horizontal shelves (layers in z).
    Within each shelf, packs boxes using 2D skyline.
    """
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        self.shelves = []  # list of dicts with 'z_start', 'height', 'boxes'
        self.placed_boxes = []
    
    def _shelf_skyline(self, shelf):
        """Get the 2D skyline for a shelf."""
        # Returns list of (x, y, max_y) representing the top of the shelf
        return shelf.get('skyline', [(0, 0, 0)])
    
    def can_place_in_shelf(self, shelf, x, y, bw, bh):
        """Check if a box fits in a shelf at (x, y)."""
        if x < 0 or y < 0:
            return False
        if x + bw > self.W or y + bh > self.H:
            return False
        for (px, py, pw, ph) in shelf['boxes_2d']:
            if not (x + bw <= px or px + pw <= x):
                if not (y + bh <= py or py + ph <= y):
                    return False
        return True
    
    def place_in_shelf(self, shelf, x, y, bw, bh, z_start, bd):
        """Place a box in a shelf."""
        shelf['boxes_2d'].append((x, y, bw, bh))
        self.placed_boxes.append((x, y, z_start, bw, bh, bd))
    
    def find_best_shelf_position(self, bw, bh, bd):
        """Find the best shelf and position for a box."""
        best_result = None
        best_score = None
        
        for shelf in self.shelves:
            z_start = shelf['z_start']
            shelf_height = shelf['height']
            
            if z_start + bd > self.D:
                continue
            if bh > shelf_height:
                continue
            
            # Try to find position in this shelf
            candidates = set()
            candidates.add((0, 0))
            
            for (px, py, pw, ph) in shelf['boxes_2d']:
                if px + pw + bw <= self.W:
                    candidates.add((px + pw, py))
                    candidates.add((px + pw, py + ph))
                if px >= bw:
                    candidates.add((px - bw, py))
                if py + ph + bh <= self.H:
                    candidates.add((px, py + ph))
                if py >= bh:
                    candidates.add((px, py - bh))
            
            for (cx, cy) in candidates:
                if self.can_place_in_shelf(shelf, cx, cy, bw, bh):
                    # Score: prefer lower z, then more compact
                    s = (z_start + bd, cy, cx)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_result = (shelf, cx, cy, z_start)
        
        # If no existing shelf works, try creating a new shelf
        if best_result is None:
            # Find the lowest z where we can start a new shelf
            max_z = 0
            for shelf in self.shelves:
                if shelf['z_start'] + shelf['height'] > max_z:
                    max_z = shelf['z_start'] + shelf['height']
            
            if max_z + bd <= self.D and bh <= self.H:
                # Create new shelf
                new_shelf = {
                    'z_start': max_z,
                    'height': bh,
                    'boxes_2d': []
                }
                self.shelves.append(new_shelf)
                self.place_in_shelf(new_shelf, 0, 0, bw, bh, max_z, bd)
                best_result = (new_shelf, 0, 0, max_z)
        
        return best_result


def _score_position_skyline(x, y, z, bw, bh, bd, placed_boxes, W, H, D):
    """Score a position: prefer compact, low, and well-connected."""
    # Primary: minimize z (pack from bottom)
    # Secondary: minimize y, then x
    # Tertiary: maximize contact with existing boxes
    contact = 0
    for (px, py, pz, pw, ph, pd) in placed_boxes:
        # Check face contacts
        if abs((x + bw) - px) < 1e-9:
            oy = min(y + bh, py + ph) - max(y, py)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if oy > 0 and oz > 0:
                contact += oy * oz
        if abs((y + bh) - py) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if ox > 0 and oz > 0:
                contact += ox * oz
        if abs((z + bd) - pz) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oy = min(y + bh, py + ph) - max(y, py)
            if ox > 0 and oy > 0:
                contact += ox * oy
    
    return (z, y, x, -contact)


def _greedy_skyline(order, boxes, container, deadline):
    """Greedy placement using skyline approach."""
    W, H, D = container
    n = len(boxes)
    skyline = Skyline3D(W, H, D)
    result = [None] * n
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        for (bw, bh, bd) in rots:
            pos = skyline.best_position(bw, bh, bd, scoring_fn=lambda x, y, z, bw, bh, bd: _score_position_skyline(x, y, z, bw, bh, bd, skyline.placed_boxes, W, H, D))
            if pos is not None:
                cx, cy, cz = pos
                s = _score_position_skyline(cx, cy, cz, bw, bh, bd, skyline.placed_boxes, W, H, D)
                if best_score is None or s < best_score:
                    best_score = s
                    best_pos = pos
                    best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            skyline.place(x, y, z, bw, bh, bd)
            result[idx] = (0, x, y, z, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result


def _greedy_shelf(order, boxes, container, deadline):
    """Greedy placement using shelf-based approach."""
    W, H, D = container
    n = len(boxes)
    packer = ShelfPacker(W, H, D)
    result = [None] * n
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        for (bw, bh, bd) in rots:
            res = packer.find_best_shelf_position(bw, bh, bd)
            if res is not None:
                shelf, cx, cy, z_start = res
                s = (z_start + bd, cy, cx)
                if best_score is None or s < best_score:
                    best_score = s
                    best_pos = (cx, cy, z_start)
                    best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            # Find which shelf
            for shelf in packer.shelves:
                if abs(shelf['z_start'] - z) < 1e-9:
                    packer.place_in_shelf(shelf, x, y, bw, bh, z, bd)
                    break
            result[idx] = (0, x, y, z, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result


def _compute_volume(result):
    """Compute total volume of placed boxes."""
    total = 0
    for r in result:
        if r[0] == 0:
            total += r[4] * r[5] * r[6]
    return total


def _heuristic_orders(boxes, n):
    """Generate multiple heuristic orderings."""
    orders = []
    
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    orders.append(('vol', vol_order))
    
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    orders.append(('maxdim', maxdim_order))
    
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    orders.append(('sa', sa_order))
    
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    orders.append(('mindim', mindim_order))
    
    # Compactness: prefer more cube-like boxes first
    compact_order = sorted(range(n), key=lambda i: max(boxes[i]) / max(1, min(boxes[i])))
    orders.append(('compact', compact_order))
    
    # Volume-weighted by max dimension
    vol_maxdim_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] * max(boxes[i]), reverse=True)
    orders.append(('vol_maxdim', vol_maxdim_order))
    
    # Surface area to volume ratio (prefer lower ratio = more compact)
    svr_order = sorted(range(n), key=lambda i: (2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])) / max(1, boxes[i][0]*boxes[i][1]*boxes[i][2]))
    orders.append(('svr', svr_order))
    
    return orders


def _local_search(result, boxes, container, deadline, max_iters=500):
    """Local search: try to place unplaced boxes and improve positions of placed boxes."""
    W, H, D = container
    n = len(boxes)
    best_result = [tuple(r) for r in result]
    best_vol = _compute_volume(best_result)
    rng = random.Random(42)
    
    def get_placed_list(result):
        placed = []
        for i in range(n):
            if result[i][0] == 0:
                x, y, z, bw, bh, bd = result[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        return placed
    
    def try_place_box(idx, placed):
        """Try to place box idx given current placed boxes."""
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        skyline = Skyline3D(W, H, D)
        for p in placed:
            skyline.place(p[0], p[1], p[2], p[3], p[4], p[5])
        
        best_pos = None
        best_score = None
        best_rot = None
        
        for (bw, bh, bd) in rots:
            pos = skyline.best_position(bw, bh, bd)
            if pos is not None:
                cx, cy, cz = pos
                s = (cz, cy, cx)
                if best_score is None or s < best_score:
                    best_score = s
                    best_pos = pos
                    best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            return True, best_pos, best_rot
        return False, None, None
    
    # Phase 1: Try to place unplaced boxes
    unplaced = [i for i in range(n) if best_result[i][0] == -1]
    attempts = 0
    while unplaced and attempts < max_iters // 3 and time.time() < deadline:
        attempts += 1
        idx = rng.choice(unplaced)
        placed = get_placed_list(best_result)
        success, pos, rot = try_place_box(idx, placed)
        if success:
            x, y, z = pos
            bw, bh, bd = rot
            temp = list(best_result)
            temp[idx] = (0, x, y, z, bw, bh, bd)
            new_vol = _compute_volume(temp)
            if new_vol > best_vol:
                best_result = [tuple(r) for r in temp]
                best_vol = new_vol
                unplaced.remove(idx)
    
    # Phase 2: Move placed boxes to find better positions
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        placed_indices = [i for i in range(n) if best_result[i][0] == 0]
        if not placed_indices:
            break
        
        idx = rng.choice(placed_indices)
        temp = list(best_result)
        temp[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        
        placed = []
        for i in range(n):
            if i != idx and temp[i][0] == 0:
                x, y, z, bw, bh, bd = temp[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        
        success, pos, rot = try_place_box(idx, placed)
        if success:
            x, y, z = pos
            bw, bh, bd = rot
            temp[idx] = (0, x, y, z, bw, bh, bd)
            new_vol = _compute_volume(temp)
            if new_vol > best_vol:
                best_result = [tuple(r) for r in temp]
                best_vol = new_vol
    
    # Phase 3: 2-box swap moves
    placed_indices = [i for i in range(n) if best_result[i][0] == 0]
    for _ in range(max_iters // 3):
        if time.time() > deadline:
            break
        if len(placed_indices) < 2:
            break
        
        i1, i2 = rng.sample(placed_indices, 2)
        temp = list(best_result)
        temp[i1] = (-1, 0, 0, 0, boxes[i1][0], boxes[i1][1], boxes[i1][2])
        temp[i2] = (-1, 0, 0, 0, boxes[i2][0], boxes[i2][1], boxes[i2][2])
        
        placed = []
        for i in range(n):
            if i not in (i1, i2) and temp[i][0] == 0:
                x, y, z, bw, bh, bd = temp[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        
        success1, pos1, rot1 = try_place_box(i1, placed)
        if success1:
            placed.append(pos1 + rot1)
            success2, pos2, rot2 = try_place_box(i2, placed)
            if success2:
                temp[i1] = (0, pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
                temp[i2] = (0, pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2])
                new_vol = _compute_volume(temp)
                if new_vol > best_vol:
                    best_result = [tuple(r) for r in temp]
                    best_vol = new_vol
    
    # Phase 4: Final polish - try harder to fit remaining boxes with more candidate positions
    unplaced = [i for i in range(n) if best_result[i][0] == -1]
    if unplaced and time.time() < deadline - 2:
        placed = get_placed_list(best_result)
        for idx in unplaced[:]:
            if time.time() > deadline - 2:
                break
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            found = False
            for (bw, bh, bd) in rots:
                # Try all extreme points from placed boxes
                candidates = set()
                candidates.add((0, 0, 0))
                for (px, py, pz, pw, ph, pd) in placed:
                    for dx in [0, pw]:
                        for dy in [0, ph]:
                            for dz in [0, pd]:
                                cx = px + dx
                                cy = py + dy
                                cz = pz + dz
                                if 0 <= cx and cx + bw <= W and 0 <= cy and cy + bh <= H and 0 <= cz and cz + bd <= D:
                                    candidates.add((cx, cy, cz))
                    # Also try positions offset by box dimensions
                    if px + pw + bw <= W:
                        candidates.add((px + pw, py, pz))
                        candidates.add((px + pw, py + ph, pz))
                    if py + ph + bh <= H:
                        candidates.add((px, py + ph, pz))
                        candidates.add((px + pw, py + ph, pz))
                    if pz + pd + bd <= D:
                        candidates.add((px, py, pz + pd))
                        candidates.add((px, py + ph, pz + pd))
                
                for (cx, cy, cz) in candidates:
                    if cx < 0 or cy < 0 or cz < 0:
                        continue
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    # Check overlap
                    ok = True
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (cx + bw <= px or px + pw <= cx):
                            if not (cy + bh <= py or py + ph <= cy):
                                if not (cz + bd <= pz or pz + pd <= cz):
                                    ok = False
                                    break
                    if ok:
                        best_result[idx] = (0, cx, cy, cz, bw, bh, bd)
                        placed.append((cx, cy, cz, bw, bh, bd))
                        best_vol = _compute_volume(best_result)
                        found = True
                        break
                if found:
                    break
    
    return best_result


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115  # Safety margin
    
    if n == 0:
        return []
    
    heuristic_orders = _heuristic_orders(boxes, n)
    
    best_result = None
    best_vol = -1
    
    # Run skyline greedy with different orderings
    for name, order in heuristic_orders:
        if time.time() > deadline - 30:
            break
        result = _greedy_skyline(order, boxes, container, deadline)
        vol = _compute_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Run shelf greedy with different orderings
    if time.time() < deadline - 20:
        for name, order in heuristic_orders[:4]:
            if time.time() > deadline - 15:
                break
            result = _greedy_shelf(order, boxes, container, deadline)
            vol = _compute_volume(result)
            if vol > best_vol:
                best_vol = vol
                best_result = result
    
    # Random restarts with skyline
    if time.time() < deadline - 15 and best_result is not None:
        rng_restart = random.Random(123)
        for _ in range(8):
            if time.time() > deadline - 10:
                break
            random_order = list(range(n))
            rng_restart.shuffle(random_order)
            result_rand = _greedy_skyline(random_order, boxes, container, deadline)
            vol_rand = _compute_volume(result_rand)
            if vol_rand > best_vol:
                best_vol = vol_rand
                best_result = result_rand
            # Also try a hybrid order: volume desc but with some randomization
            if time.time() < deadline - 10:
                hybrid_order = list(range(n))
                rng_restart.shuffle(hybrid_order)
                hybrid_order.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2]*0.7 + rng_restart.random()*0.3*max(b[0]*b[1]*b[2] for b in boxes), reverse=True)
                result_hybrid = _greedy_skyline(hybrid_order, boxes, container, deadline)
                vol_hybrid = _compute_volume(result_hybrid)
                if vol_hybrid > best_vol:
                    best_vol = vol_hybrid
                    best_result = result_hybrid
    
    # Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_result is not None:
        max_iters = int(remaining_time * 30)
        max_iters = max(100, min(max_iters, 1500))
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters)
    
    # Final multi-start polish: try a few more greedy passes with different seeds
    if best_result is not None and time.time() < deadline - 5:
        rng_final = random.Random(999)
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            random_order = list(range(n))
            rng_final.shuffle(random_order)
            result_final = _greedy_skyline(random_order, boxes, container, deadline)
            vol_final = _compute_volume(result_final)
            if vol_final > best_vol:
                best_vol = vol_final
                best_result = result_final
                # Run local search on the improved result
                if time.time() < deadline - 3:
                    max_iters2 = max(50, int((deadline - time.time() - 3) * 20))
                    best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters2)
                    best_vol = _compute_volume(best_result)
    
    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return [tuple(r) for r in best_result]

# EVOLVE-BLOCK-END
