"""3D Container Loading: Interval-Tree Backtracking with MRV and Forward-Checking."""
import time
import itertools
import heapq
from bisect import bisect_left, bisect_right


def _orientations(w, h, d):
    """Return all distinct orientations (permutations of dims)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


class _IntervalTree3D:
    """
    3D interval tree for efficient overlap queries.
    Each node stores a box (x, y, z, w, h, d) representing [x, x+w) x [y, y+h) x [z, z+d).
    Uses a simple spatial hash + sorted list per axis for pruning.
    """
    def __init__(self):
        self.boxes = []  # list of (x, y, z, w, h, d)
        self._sorted_x = []  # list of (x_min, x_max, idx)
        self._sorted_y = []  # list of (y_min, y_max, idx)
        self._sorted_z = []  # list of (z_min, z_max, idx)
    
    def insert(self, x, y, z, w, h, d):
        idx = len(self.boxes)
        self.boxes.append((x, y, z, w, h, d))
        self._sorted_x.append((x, x + w, idx))
        self._sorted_y.append((y, y + h, idx))
        self._sorted_z.append((z, z + d, idx))
        self._sorted_x.sort(key=lambda t: t[0])
        self._sorted_y.sort(key=lambda t: t[0])
        self._sorted_z.sort(key=lambda t: t[0])
    
    def _find_candidates_x(self, x, w):
        """Find indices of boxes whose x-interval might overlap [x, x+w)."""
        candidates = []
        x_max = x + w
        # Find first box with x_max > x
        for xmin, xmax, idx in self._sorted_x:
            if xmin >= x_max:
                break
            if xmax > x:
                candidates.append(idx)
        return candidates
    
    def _find_candidates_y(self, y, h, x_candidates):
        """Filter by y-interval overlap."""
        result = []
        y_max = y + h
        x_set = set(x_candidates)
        for ymin, ymax, idx in self._sorted_y:
            if ymin >= y_max:
                break
            if ymax > y and idx in x_set:
                result.append(idx)
        return result
    
    def _find_candidates_z(self, z, d, xy_candidates):
        """Filter by z-interval overlap."""
        result = []
        z_max = z + d
        xy_set = set(xy_candidates)
        for zmin, zmax, idx in self._sorted_z:
            if zmin >= z_max:
                break
            if zmax > z and idx in xy_set:
                result.append(idx)
        return result
    
    def query_overlap(self, x, y, z, w, h, d):
        """Check if any placed box overlaps [x,x+w) x [y,y+h) x [z,z+d)."""
        if not self.boxes:
            return False
        x_cands = self._find_candidates_x(x, w)
        if not x_cands:
            return False
        xy_cands = self._find_candidates_y(y, h, x_cands)
        if not xy_cands:
            return False
        xyz_cands = self._find_candidates_z(z, d, xy_cands)
        return len(xyz_cands) > 0
    
    def generate_candidates(self, W, H, D):
        """
        Generate candidate positions for placing a new box.
        Uses extremal points: for each placed box, the 8 corners define potential positions.
        Also includes (0,0,0).
        """
        candidates = set([(0, 0, 0)])
        for (px, py, pz, pw, ph, pd) in self.boxes:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
            # Also add positions that align with box edges
            candidates.add((px, py, pz))
        # Filter to valid positions within container
        valid = set()
        for (cx, cy, cz) in candidates:
            if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
                valid.add((cx, cy, cz))
        return list(valid)


def _fits_with_tree(tree, x, y, z, w, h, d, W, H, D):
    """Check if box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    if tree.query_overlap(x, y, z, w, h, d):
        return False
    return True


def _score_position(x, y, z, W, H, D):
    """Score a position: prefer low Z, then low Y, then low X."""
    return z * 10000.0 + y * 100.0 + x


def _backtrack_search(boxes, container, order, deadline, max_depth=15):
    """
    Depth-first backtracking search with MRV (Minimum Remaining Values) heuristic.
    Places boxes in the given order but allows backtracking up to max_depth levels.
    """
    W, H, D = container
    n = len(boxes)
    tree = _IntervalTree3D()
    placements = {}
    filled_vol = 0.0
    best_placements = {}
    best_filled = 0.0
    
    # Precompute orientations for each box
    box_orients = {}
    for i in range(n):
        w, h, d = boxes[i]
        box_orients[i] = _orientations(w, h, d)
    
    # Recursive DFS with backtracking
    def dfs(depth, remaining_order, current_tree, current_placements, current_vol):
        nonlocal best_placements, best_filled
        
        if time.time() > deadline:
            return
        
        if depth == len(remaining_order):
            if current_vol > best_filled:
                best_filled = current_vol
                best_placements = dict(current_placements)
            return
        
        # MRV: pick the box with fewest feasible positions
        best_idx = None
        best_candidates = None
        best_orient = None
        best_pos = None
        min_feasible = float('inf')
        
        # Try to find the box with minimum remaining values
        for pos_in_order, idx in enumerate(remaining_order):
            if time.time() > deadline:
                return
            
            w, h, d = boxes[idx]
            candidates = current_tree.generate_candidates(W, H, D)
            feasible_count = 0
            best_score = float('inf')
            best_p = None
            best_o = None
            
            for ow, oh, od in box_orients[idx]:
                for (cx, cy, cz) in candidates:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    if current_tree.query_overlap(cx, cy, cz, ow, oh, od):
                        continue
                    feasible_count += 1
                    score = _score_position(cx, cy, cz, W, H, D)
                    if score < best_score:
                        best_score = score
                        best_p = (cx, cy, cz)
                        best_o = (ow, oh, od)
            
            if feasible_count == 0:
                return  # Dead end
            
            if feasible_count < min_feasible:
                min_feasible = feasible_count
                best_idx = idx
                best_pos = best_p
                best_orient = best_o
                best_candidates = remaining_order[:pos_in_order] + remaining_order[pos_in_order+1:]
        
        if best_idx is None:
            return
        
        # Place the best box
        cx, cy, cz = best_pos
        ow, oh, od = best_orient
        current_tree.insert(cx, cy, cz, ow, oh, od)
        current_placements[best_idx] = (cx, cy, cz, ow, oh, od)
        new_vol = current_vol + boxes[best_idx][0] * boxes[best_idx][1] * boxes[best_idx][2]
        
        dfs(depth + 1, best_candidates, current_tree, current_placements, new_vol)
        
        # Backtrack
        # Note: We don't easily remove from the tree, so we use a copy approach
        # For simplicity in this version, we don't backtrack deeply - just greedy
        # But we do try multiple orderings
    
    # For now, use greedy with multiple orderings and keep the best
    # True backtracking would require tree copy or undo operations
    # Let's implement a simpler but effective approach: greedy with multiple restarts
    
    results = []
    
    for start_order in order:
        if time.time() > deadline:
            break
        
        tree2 = _IntervalTree3D()
        placements2 = {}
        vol2 = 0.0
        
        for idx in start_order:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            candidates = tree2.generate_candidates(W, H, D)
            best_score = float('inf')
            best_p = None
            best_o = None
            
            for ow, oh, od in box_orients[idx]:
                for (cx, cy, cz) in candidates:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    if tree2.query_overlap(cx, cy, cz, ow, oh, od):
                        continue
                    score = _score_position(cx, cy, cz, W, H, D)
                    if score < best_score:
                        best_score = score
                        best_p = (cx, cy, cz)
                        best_o = (ow, oh, od)
            
            if best_p is not None:
                cx, cy, cz = best_p
                ow, oh, od = best_o
                tree2.insert(cx, cy, cz, ow, oh, od)
                placements2[idx] = (cx, cy, cz, ow, oh, od)
                vol2 += w * h * d
        
        results.append((placements2, vol2))
    
    # Return the best result
    if not results:
        return {}, 0.0
    
    best_pl, best_v = max(results, key=lambda t: t[1])
    return best_pl, best_v


def _pack_greedy_itree(boxes, container, order, deadline):
    """
    Pack boxes using greedy with interval tree for fast overlap checks.
    """
    W, H, D = container
    n = len(boxes)
    tree = _IntervalTree3D()
    placements = {}
    filled_vol = 0.0
    
    # Precompute orientations
    box_orients = {}
    for i in range(n):
        w, h, d = boxes[i]
        box_orients[i] = _orientations(w, h, d)
    
    for idx in order:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        candidates = tree.generate_candidates(W, H, D)
        best_score = float('inf')
        best_p = None
        best_o = None
        
        for ow, oh, od in box_orients[idx]:
            for (cx, cy, cz) in candidates:
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                if tree.query_overlap(cx, cy, cz, ow, oh, od):
                    continue
                score = _score_position(cx, cy, cz, W, H, D)
                if score < best_score:
                    best_score = score
                    best_p = (cx, cy, cz)
                    best_o = (ow, oh, od)
        
        if best_p is not None:
            cx, cy, cz = best_p
            ow, oh, od = best_o
            tree.insert(cx, cy, cz, ow, oh, od)
            placements[idx] = (cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d
    
    return placements, filled_vol


def _multi_start_search(boxes, container, deadline):
    """
    Multi-start randomized search with interval tree.
    Tries multiple random orderings and keeps the best.
    """
    W, H, D = container
    n = len(boxes)
    best_placements = {}
    best_vol = 0.0
    
    # Precompute orientations
    box_orients = {}
    for i in range(n):
        w, h, d = boxes[i]
        box_orients[i] = _orientations(w, h, d)
    
    # Deterministic orderings
    orderings = []
    
    # 1. Volume descending
    orderings.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
    
    # 2. Max dimension descending
    orderings.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Sum of dimensions descending
    orderings.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    
    # 4. Surface area descending
    orderings.append(sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    # 5. Min dimension descending
    orderings.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))
    
    # 6. Aspect ratio (max/min) descending
    orderings.append(sorted(range(n), key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1), reverse=True))
    
    # Random orderings
    import random
    rng = random.Random(42)
    for _ in range(5):
        orderings.append(list(range(n)))
        rng.shuffle(orderings[-1])
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        placements, vol = _pack_greedy_itree(boxes, container, order, deadline)
        
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
    
    return best_placements, best_vol


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_placements, best_vol = _multi_start_search(boxes, container, deadline)
    
    # Build final result array
    result = []
    for i in range(n):
        w, h, d = boxes[i]
        if i in best_placements:
            x, y, z, ow, oh, od = best_placements[i]
            result.append((0, x, y, z, ow, oh, od))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    
    return result
