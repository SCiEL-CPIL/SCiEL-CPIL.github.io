"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s kill
    
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    if container_vol <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Precompute orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    orientations_cache = [get_orientations(w, h, d) for w, h, d in boxes]
    
    # Floor-map based packing: maintain a 2D grid of minimum z-heights
    class FloorMap:
        def __init__(self, W, H, D):
            self.W = W
            self.H = H
            self.D = D
            # Use integer grid if dimensions allow, else use list of placed boxes
            self.use_grid = (W <= 200 and H <= 200)
            if self.use_grid:
                self.grid = [[0] * W for _ in range(H)]
            else:
                self.placed = []
        
        def can_place(self, x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > self.W or y + h > self.H or z + d > self.D:
                return False
            if self.use_grid:
                for gy in range(y, y + h):
                    for gx in range(x, x + w):
                        if self.grid[gy][gx] + d > self.D:
                            return False
                        if self.grid[gy][gx] > z:
                            return False
                return True
            else:
                for (px, py, pz, pw, ph, pd) in self.placed:
                    if not (x + w <= px or px + pw <= x or
                            y + h <= py or py + ph <= y or
                            z + d <= pz or pz + pd <= z):
                        return False
                return True
        
        def place(self, x, y, z, w, h, d):
            if self.use_grid:
                for gy in range(y, y + h):
                    for gx in range(x, x + w):
                        self.grid[gy][gx] = max(self.grid[gy][gx], z + d)
            else:
                self.placed.append((x, y, z, w, h, d))
        
        def get_candidates(self):
            """Generate candidate (x,y) positions to try"""
            cands = [(0, 0)]
            seen = set(cands)
            if self.use_grid:
                # Find unique x,y positions from grid boundaries
                for gy in range(self.H):
                    for gx in range(self.W):
                        if self.grid[gy][gx] > 0:
                            # Add adjacent positions
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    nx, ny = gx + dx, gy + dy
                                    if 0 <= nx < self.W and 0 <= ny < self.H:
                                        if (nx, ny) not in seen:
                                            seen.add((nx, ny))
                                            cands.append((nx, ny))
            else:
                for (px, py, pz, pw, ph, pd) in self.placed:
                    for (nx, ny) in [
                        (px + pw, py),
                        (px, py + ph),
                        (px + pw, py + ph),
                    ]:
                        if (nx, ny) not in seen and nx <= self.W and ny <= self.H:
                            seen.add((nx, ny))
                            cands.append((nx, ny))
            return cands
        
        def lowest_z(self, x, y, w, h):
            """Find the lowest valid z for a box at (x,y) with footprint w,h"""
            if x + w > self.W or y + h > self.H:
                return -1
            min_z = 0
            if self.use_grid:
                for gy in range(y, y + h):
                    for gx in range(x, x + w):
                        min_z = max(min_z, self.grid[gy][gx])
            else:
                for (px, py, pz, pw, ph, pd) in self.placed:
                    if not (x + w <= px or px + pw <= x or
                            y + h <= py or py + ph <= y):
                        min_z = max(min_z, pz + pd)
            return min_z
    
    def position_score(x, y, z):
        """Score: prefer lower z, then lower y, then lower x"""
        return z * W * H + y * W + x
    
    def place_one(box_idx, fm):
        """Try to place a single box; return (x,y,z,ow,oh,od) or None"""
        w, h, d = boxes[box_idx]
        best = None
        best_score = float('inf')
        
        for (ow, oh, od) in orientations_cache[box_idx]:
            if ow > W or oh > H or od > D:
                continue
            
            # Get candidate (x,y) positions
            cands = fm.get_candidates()
            
            for (cx, cy) in cands:
                # Find lowest valid z
                z = fm.lowest_z(cx, cy, ow, oh)
                if z < 0:
                    continue
                if z + od > D:
                    continue
                
                # Verify placement is valid
                if fm.can_place(cx, cy, z, ow, oh, od):
                    s = position_score(cx, cy, z)
                    if s < best_score:
                        best_score = s
                        best = (cx, cy, z, ow, oh, od)
        
        return best
    
    def evaluate(result):
        vol = 0
        for p in result:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        return vol / container_vol
    
    def greedy(order):
        fm = FloorMap(W, H, D)
        result = [None] * n
        for idx in order:
            pos = place_one(idx, fm)
            if pos is not None:
                x, y, z, ow, oh, od = pos
                fm.place(x, y, z, ow, oh, od)
                result[idx] = (0, x, y, z, ow, oh, od)
            else:
                w, h, d = boxes[idx]
                result[idx] = (-1, 0, 0, 0, w, h, d)
        return result
    
    # Build various orders
    indices = list(range(n))
    
    def order_by_volume():
        return sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def order_by_maxdim():
        return sorted(indices, key=lambda i: max(boxes[i]), reverse=True)
    
    def order_by_mindim():
        return sorted(indices, key=lambda i: min(boxes[i]), reverse=True)
    
    def order_by_surface():
        return sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1]+boxes[i][1]*boxes[i][2]+boxes[i][2]*boxes[i][0]), reverse=True)
    
    def order_by_aspect():
        def asp(i):
            w,h,d = boxes[i]
            return max(w,h,d)/min(w,h,d) if min(w,h,d)>0 else 0
        return sorted(indices, key=lambda i: asp(i))
    
    def order_random():
        r = list(indices)
        random.shuffle(r)
        return r
    
    best_result = None
    best_fill = 0
    
    # Try deterministic strategies
    strategies = [order_by_volume, order_by_maxdim, order_by_mindim, order_by_surface, order_by_aspect]
    
    for strat in strategies:
        if time.time() > deadline:
            break
        order = strat()
        res = greedy(order)
        fill = evaluate(res)
        if fill > best_fill:
            best_fill = fill
            best_result = res
    
    # Try random orders
    n_random = 40
    for r in range(n_random):
        if time.time() > deadline - 2:
            break
        order = order_random()
        res = greedy(order)
        fill = evaluate(res)
        if fill > best_fill:
            best_fill = fill
            best_result = res
    
    # Final: try to re-insert any remaining left-out boxes into the best result
    if best_result is not None and time.time() < deadline - 1:
        fm = FloorMap(W, H, D)
        # Rebuild floor map from best_result
        for p in best_result:
            if p[0] == 0:
                fm.place(p[1], p[2], p[3], p[4], p[5], p[6])
        
        for idx in range(n):
            if best_result[idx][0] == -1:
                pos = place_one(idx, fm)
                if pos is not None:
                    x, y, z, ow, oh, od = pos
                    fm.place(x, y, z, ow, oh, od)
                    best_result[idx] = (0, x, y, z, ow, oh, od)
    
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result
# EVOLVE-BLOCK-END
