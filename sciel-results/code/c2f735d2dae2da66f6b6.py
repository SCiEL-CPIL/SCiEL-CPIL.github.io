import numpy as np
import math
import random

N = 11

# Precompute axes for SAT and containment
# Normals to edges of a hexagon with rotation 0 (vertex at 0 deg)
# Vertices at 0, 60, 120, 180, 240, 300
# Edges connect 0-60, 60-120, etc.
# Midpoint of edge 0-60 is at 30 deg. Normal is 30 deg.
AXES = []
for i in range(6):
    angle = math.radians(30 + i * 60)
    AXES.append((math.cos(angle), math.sin(angle)))

def hex_min_max_projections(cx, cy):
    """
    Return min and max projections of a unit hexagon centered at (cx, cy) with rotation 0
    onto the 6 precomputed axes.
    """
    mins = [0.0] * 6
    maxs = [0.0] * 6
    
    # Vertices of unit hexagon at origin
    # v_k = (cos(k*pi/3), sin(k*pi/3))
    # We need to project (cx + vx, cy + vy) onto axis (ax, ay)
    # dot = cx*ax + cy*ay + vx*ax + vy*ay
    
    # Precompute vertex projections on axes for a unit hexagon at origin
    # This is constant!
    # Let's compute it once.
    
    return mins, maxs

# Let's precompute the constant part
VERTEX_PROJ_MINS = [0.0] * 6
VERTEX_PROJ_MAXS = [0.0] * 6
for i in range(6):
    ax, ay = AXES[i]
    min_proj = 0.0
    max_proj = 0.0
    for k in range(6):
        theta = k * math.pi / 3
        vx = math.cos(theta)
        vy = math.sin(theta)
        proj = vx * ax + vy * ay
        if proj < min_proj: min_proj = proj
        if proj > max_proj: max_proj = proj
    VERTEX_PROJ_MINS[i] = min_proj
    VERTEX_PROJ_MAXS[i] = max_proj

def get_hex_extent(cx, cy):
    """
    Return the min and max extent of the hexagon centered at (cx, cy) on each of the 6 axes.
    """
    ext_min = [0.0] * 6
    ext_max = [0.0] * 6
    for i in range(6):
        ax, ay = AXES[i]
        center_proj = cx * ax + cy * ay
        ext_min[i] = center_proj + VERTEX_PROJ_MINS[i]
        ext_max[i] = center_proj + VERTEX_PROJ_MAXS[i]
    return ext_min, ext_max

def check_overlap_and_containment(state):
    """
    state: list of (x, y)
    Returns (max_side, overlap_penalty)
    """
    max_side = 0.0
    overlap_penalty = 0.0
    
    exts = []
    for x, y in state:
        ext_min, ext_max = get_hex_extent(x, y)
        exts.append((ext_min, ext_max))
        
        # Containment check
        # Outer hexagon centered at (0,0), side L.
        # Extent on axis i is +/- L * sqrt(3) / 2
        # So we need ext_max[i] <= L * sqrt(3) / 2 and ext_min[i] >= -L * sqrt(3) / 2
        # => L >= 2 * max(ext_max[i], -ext_min[i]) / sqrt(3)
        for i in range(6):
            val = max(ext_max[i], -ext_min[i])
            req_side = val * 2.0 / math.sqrt(3)
            if req_side > max_side:
                max_side = req_side
                
    # Overlap check
    for i in range(N):
        min_i, max_i = exts[i]
        for j in range(i + 1, N):
            min_j, max_j = exts[j]
            # Check if intervals overlap on all axes
            overlaps = True
            for k in range(6):
                if max_i[k] < min_j[k] or max_j[k] < min_i[k]:
                    overlaps = False
                    break
            if overlaps:
                overlap_penalty += 1000.0
                
    return max_side, overlap_penalty

def construct():
    # Initialization
    positions = []
    # 1 center
    positions.append((0.0, 0.0))
    # 6 neighbors
    dist = math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        positions.append((x, y))
    # 4 more
    for i in range(4):
        angle = math.radians(90 + 60 * i)
        r = 2.5
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        positions.append((x, y))
        
    state = positions[:]
    
    def score(state):
        ms, op = check_overlap_and_containment(state)
        return ms + op
        
    current_score = score(state)
    best_state = state[:]
    best_score = current_score
    
    T = 5.0
    cooling = 0.999
    steps = 20000
    
    for step in range(steps):
        new_state = state[:]
        idx = random.randint(0, N - 1)
        x, y = new_state[idx]
        
        # Random move
        x += random.gauss(0, 0.5)
        y += random.gauss(0, 0.5)
        new_state[idx] = (x, y)
        
        new_score = score(new_state)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            state = new_state
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_state = state[:]
                
        T *= cooling
        if T < 0.001:
            break
            
    inner = []
    for x, y in best_state:
        inner.append([x, y, 0.0])
        
    # Final side length
    ms, _ = check_overlap_and_containment(best_state)
    
    return inner, (0.0, 0.0), ms, 0.0
