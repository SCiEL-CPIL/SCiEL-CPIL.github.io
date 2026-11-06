import time
import math
import random
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

class _BSPNode:
    __slots__ = ('x', 'y', 'z', 'w', 'h', 'd', 'children')
    def __init__(self, x, y, z, w, h, d):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h
        self.d = d
        self.children = None

def _split_node(node, bx, by, bz, bw, bh, bd):
    """
    Split a node into up to 6 sub-nodes around a placed box.
    The box is placed at (bx, by, bz) with size (bw, bh, bd).
    The node is (x, y, z, w, h, d).
    Returns a list of new _BSPNode objects.
    """
    x, y, z, w, h, d = node.x, node.y, node.z, node.w, node.h, node.d
    
    # Calculate the boundaries of the box relative to the node
    # Box is at (bx, by, bz)
    # Node is (x, y, z) to (x+w, y+h, z+d)
    
    new_nodes = []
    
    # 1. Right: x from bx+bw to x+w, y from y to y+h, z from z to z+d
    if bx + bw < x + w:
        new_nodes.append(_BSPNode(bx + bw, y, z, x + w - (bx + bw), h, d))
        
    # 2. Back-Left: x from x to bx+bw, y from by+bh to y+h, z from z to z+d
    if by + bh < y + h and bx + bw > x:
        # Width is from x to bx+bw
        wl = bx + bw - x
        if wl > 0:
            new_nodes.append(_BSPNode(x, by + bh, z, wl, y + h - (by + bh), d))

    # 3. Back-Right: x from bx+bw to x+w, y from by+bh to y+h, z from z to z+d
    if by + bh < y + h and bx + bw < x + w:
        wr = x + w - (bx + bw)
        if wr > 0:
            new_nodes.append(_BSPNode(bx + bw, by + bh, z, wr, y + h - (by + bh), d))

    # 4. Top-Front-Left: x from x to bx+bw, y from y to by+bh, z from bz+bd to z+d
    if bz + bd < z + d and bx + bw > x:
        wl = bx + bw - x
        if wl > 0:
            new_nodes.append(_BSPNode(x, y, bz + bd, wl, by + bh - y, z + d - (bz + bd)))

    # 5. Top-Front-Right: x from bx+bw to x+w, y from y to by+bh, z from bz+bd to z+d
    if bz + bd < z + d and bx + bw < x + w:
        wr = x + w - (bx + bw)
        if wr > 0:
            new_nodes.append(_BSPNode(bx + bw, y, bz + bd, wr, by + bh - y, z + d - (bz + bd)))

    # 6. Top-Back: x from x to x+w, y from by+bh to y+h, z from bz+bd to z+d
    if bz + bd < z + d and by + bh < y + h:
        new_nodes.append(_BSPNode(x, by + bh, bz + bd, w, y + h - (by + bh), z + d - (bz + bd)))
        
    return new_nodes

def _try_place_bsp(boxes, container, sort_key='volume', seed=0):
    """
    Attempt to place boxes using BSP tree.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute rotations
    box_rotations = []
    for w, h, d in boxes:
        box_rotations.append(_get_rotations(w, h, d))
    
    # Sort boxes
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    if sort_key == 'volume':
        box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    elif sort_key == 'max_dim':
        box_indices.sort(key=lambda i: max(boxes[i]), reverse=True)
    elif sort_key == 'min_dim':
        box_indices.sort(key=lambda i: min(boxes[i]), reverse=True)
    elif sort_key == 'surface_area':
        box_indices.sort(key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    else:
        box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
        
    # Initialize BSP with the container
    root = _BSPNode(0, 0, 0, W, H, D)
    available_leaves = [root]
    
    result = [None] * n
    
    for idx in box_indices:
        if time.time() > deadline:
            break
            
        w_orig, h_orig, d_orig = boxes[idx]
        rots = box_rotations[idx]
        
        best_leaf_idx = -1
        best_pos = None
        best_rot = None
        best_score = None
        
        # Find the best leaf to place the box
        # We iterate through all available leaves
        # To optimize, we could sort leaves, but for 60 boxes, the tree won't be too deep/wide
        for leaf_idx, leaf in enumerate(available_leaves):
            lx, ly, lz, lw, lh, ld = leaf.x, leaf.y, leaf.z, leaf.w, leaf.h, leaf.d
            
            # Try each rotation
            for rw, rh, rd in rots:
                if rw > lw or rh > lh or rd > ld:
                    continue
                    
                # Calculate position: place at the bottom-front-left corner of the leaf
                # This is a greedy choice. We could try other positions, but corner is standard for BSP
                px, py, pz = lx, ly, lz
                
                # Score: Prefer lower z, then lower x, then lower y
                # Also prefer smaller leaf volume (best fit) to preserve large empty spaces
                # Let's use a composite score
                # Primary: z, x, y (compactness)
                # Secondary: leaf volume (best fit)
                
                # If we just use (pz, px, py), we might fill the bottom layer perfectly.
                # Let's stick to (pz, px, py) for consistency with previous heuristics.
                score = (pz, px, py)
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_leaf_idx = leaf_idx
                    best_pos = (px, py, pz)
                    best_rot = (rw, rh, rd)
                    
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            
            # Remove the chosen leaf
            leaf = available_leaves.pop(best_leaf_idx)
            
            # Split the leaf
            new_nodes = _split_node(leaf, bx, by, bz, bw, bh, bd)
            available_leaves.extend(new_nodes)
            
            result[idx] = (0, bx, by, bz, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
            
    # Fill remaining
    for i in range(n):
        if result[i] is None:
            w_i, h_i, d_i = boxes[i]
            result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
            
    return result

def _try_place_diverse_bsp(boxes, container):
    """
    Try multiple sorting strategies with BSP and return the best.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
        
    start_time = time.time()
    deadline = start_time + 145.0
    
    strategies = ['volume', 'max_dim', 'min_dim', 'surface_area']
    
    best_result = None
    best_filled = 0
    
    for strategy in strategies:
        if time.time() > deadline:
            break
            
        result = _try_place_bsp(boxes, container, sort_key=strategy)
        
        filled = sum(r[4] * r[5] * r[6] for r in result if r[0] == 0)
        
        if filled > best_filled:
            best_filled = filled
            best_result = result
            
    if best_result is None:
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
    return best_result

def load(boxes, container):
    """
    Place boxes into container. Returns list of (bin, x, y, z, w, h, d) for each box.
    bin=0 for loaded, bin=-1 for left out.
    """
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 148.0
    
    # Use the BSP-based diverse strategy
    placements = _try_place_diverse_bsp(boxes, container)
    
    return placements
