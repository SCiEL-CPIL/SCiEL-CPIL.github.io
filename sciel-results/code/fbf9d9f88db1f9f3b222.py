import numpy as np
import math
import random

N = 11

def get_hex_vertices(cx, cy, angle=0.0):
    verts = []
    for k in range(6):
        theta = angle + k * math.pi / 3
        verts.append((cx + math.cos(theta), cy + math.sin(theta)))
    return verts

def calc_required_side_centered(positions):
    # Assumes positions are centered such that the optimal outer hexagon is at (0,0)
    # This is an approximation if not perfectly centered, but we will center them.
    max_proj = 0.0
    normals = []
    for k in range(6):
        a = k * math.pi / 3
        normals.append((math.cos(a), math.sin(a)))
    
    for px, py in positions:
        verts = get_hex_vertices(px, py, 0.0)
        for vx, vy in verts:
            for nx, ny in normals:
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
    
    return (2.0 / math.sqrt(3.0)) * max_proj

def overlap_penalty(positions):
    pen = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d2 = dx * dx + dy * dy
            if d2 < 4.0:
                d = math.sqrt(d2)
                pen += (2.0 - d) ** 2
    return pen

def construct():
    # Init
    hex_coords = [
        (0,0),
        (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
        (2,0), (-2,0), (0,2), (0,-2)
    ]
    init_pos = []
    for q, r in hex_coords:
        x = 2.0 * q + r
        y = math.sqrt(3.0) * r
        init_pos.append((x, y))
        
    positions = init_pos[:]
    
    # Center the initial positions
    cx = sum(p[0] for p in positions) / N
    cy = sum(p[1] for p in positions) / N
    positions = [(x - cx, y - cy) for x, y in positions]
    
    best_pos = positions[:]
    best_S = calc_required_side_centered(best_pos)
    best_score = best_S + 1000 * overlap_penalty(best_pos)
    
    random.seed(42)
    
    for restart in range(50):
        if restart == 0:
            curr_pos = best_pos[:]
        else:
            curr_pos = [ (x + random.uniform(-2, 2), y + random.uniform(-2, 2)) for x, y in best_pos ]
            # Center it
            cx = sum(p[0] for p in curr_pos) / N
            cy = sum(p[1] for p in curr_pos) / N
            curr_pos = [(x - cx, y - cy) for x, y in curr_pos]
            
        curr_score = best_score if restart == 0 else calc_required_side_centered(curr_pos) + 1000 * overlap_penalty(curr_pos)
        
        for it in range(1000):
            idx = random.randint(0, N - 1)
            dx = random.uniform(-0.5, 0.5)
            dy = random.uniform(-0.5, 0.5)
            new_pos = curr_pos[:]
            new_pos[idx] = (curr_pos[idx][0] + dx, curr_pos[idx][1] + dy)
            
            # Re-center
            cx = sum(p[0] for p in new_pos) / N
            cy = sum(p[1] for p in new_pos) / N
            new_pos = [(x - cx, y - cy) for x, y in new_pos]
            
            new_S = calc_required_side_centered(new_pos)
            new_pen = overlap_penalty(new_pos)
            new_score = new_S + 1000 * new_pen
            
            if new_score < curr_score:
                curr_pos = new_pos
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_pos = curr_pos[:]
                    
    # Final centering
    cx = sum(p[0] for p in best_pos) / N
    cy = sum(p[1] for p in best_pos) / N
    final_pos = [(x - cx, y - cy) for x, y in best_pos]
    
    S = calc_required_side_centered(final_pos)
    
    inner = []
    for x, y in final_pos:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
