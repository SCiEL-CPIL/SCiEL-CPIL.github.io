import numpy as np
import time
import math

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = float('inf')
    best_result = None
    sqrt3 = math.sqrt(3)
    
    # 3-4-3 configuration
    inner_343 = []
    # Middle row: 4 hexes at y=0, x = -3, -1, 1, 3
    for x in [-3, -1, 1, 3]:
        inner_343.append([float(x), 0.0, 0.0])
    # Top row: 3 hexes at y=sqrt3, x = -2, 0, 2
    for x in [-2, 0, 2]:
        inner_343.append([float(x), sqrt3, 0.0])
    # Bottom row: 3 hexes at y=-sqrt3, x = -2, 0, 2
    for x in [-2, 0, 2]:
        inner_343.append([float(x), -sqrt3, 0.0])
    
    def check_valid(inner, outer_side):
        """All hexagons angle=0. Check pairwise dist >= 2 and containment."""
        # Pairwise distance
        for i in range(N):
            for j in range(i+1, N):
                dx = inner[j][0] - inner[i][0]
                dy = inner[j][1] - inner[i][1]
                if dx*dx + dy*dy < 4.0 - 1e-9:
                    return False
        
        # Containment in outer hexagon (center origin, side=outer_side, flat-top)
        # Point (x,y) inside iff:
        # |y| <= outer_side * sqrt3/2  (flat-to-flat height)
        # |x| <= outer_side            (point-to-point width)
        # sqrt3*|x| + |y| <= outer_side * sqrt3  (diagonal constraints)
        # Actually, the half-plane conditions for flat-top hexagon side s:
        # Vertices: (s,0), (s/2, s*sqrt3/2), (-s/2, s*sqrt3/2), (-s,0), (-s/2, -s*sqrt3/2), (s/2, -s*sqrt3/2)
        # Edges and inward normals:
        # Edge 0: (s,0) to (s/2, s*sqrt3/2): direction (-s/2, s*sqrt3/2), normal (s*sqrt3/2, s/2) -> normalized (sqrt3, 1)/2
        #   Condition: sqrt3*x + y <= s*sqrt3
        # Edge 1: (s/2, s*sqrt3/2) to (-s/2, s*sqrt3/2): direction (-s, 0), normal (0, s) -> (0,1)
        #   Condition: y <= s*sqrt3/2
        # Edge 2: (-s/2, s*sqrt3/2) to (-s,0): direction (-s/2, -s*sqrt3/2), normal (-s*sqrt3/2, s/2) -> (-sqrt3, 1)/2
        #   Condition: -sqrt3*x + y <= s*sqrt3
        # Edge 3: (-s,0) to (-s/2, -s*sqrt3/2): direction (s/2, -s*sqrt3/2), normal (-s*sqrt3/2, -s/2) -> (-sqrt3, -1)/2
        #   Condition: -sqrt3*x - y <= s*sqrt3
        # Edge 4: (-s/2, -s*sqrt3/2) to (s/2, -s*sqrt3/2): direction (s, 0), normal (0, -s) -> (0,-1)
        #   Condition: -y <= s*sqrt3/2  i.e. y >= -s*sqrt3/2
        # Edge 5: (s/2, -s*sqrt3/2) to (s,0): direction (s/2, s*sqrt3/2), normal (s*sqrt3/2, -s/2) -> (sqrt3, -1)/2
        #   Condition: sqrt3*x - y <= s*sqrt3
        
        s = outer_side
        for i in range(N):
            cx, cy = inner[i][0], inner[i][1]
            # Check all 6 vertices of inner hexagon
            for k in range(6):
                angle = k * math.pi / 3
                vx = cx + math.cos(angle)
                vy = cy + math.sin(angle)
                if sqrt3 * vx + vy > s * sqrt3 + 1e-9:
                    return False
                if vy > s * sqrt3 / 2 + 1e-9:
                    return False
                if -sqrt3 * vx + vy > s * sqrt3 + 1e-9:
                    return False
                if -sqrt3 * vx - vy > s * sqrt3 + 1e-9:
                    return False
                if -vy > s * sqrt3 / 2 + 1e-9:
                    return False
                if sqrt3 * vx - vy > s * sqrt3 + 1e-9:
                    return False
        return True
    
    if check_valid(inner_343, 4.0):
        best_score = 4.0
        best_result = (inner_343, (0.0, 0.0), 4.0, 0.0)
    
    # Optimize
    np.random.seed(42)
    
    def optimize(target_side, n_restarts=5, n_steps=1000):
        nonlocal best_score, best_result
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return
            
            inner = [list(x) for x in inner_343]
            for i in range(N):
                inner[i][0] += np.random.normal(0, 0.2)
                inner[i][1] += np.random.normal(0, 0.2)
            
            for step in range(n_steps):
                if time.time() > deadline:
                    return
                
                if check_valid(inner, target_side):
                    best_score = target_side
                    best_result = (inner, (0.0, 0.0), target_side, 0.0)
                    return
                
                for i in range(N):
                    if np.random.random() < 0.5:
                        inner[i][0] += np.random.normal(0, 0.1)
                    if np.random.random() < 0.5:
                        inner[i][1] += np.random.normal(0, 0.1)
    
    for target in [3.95, 3.9, 3.85, 3.8, 3.75, 3.7, 3.65, 3.6, 3.55, 3.5, 3.45, 3.4]:
        if time.time() > deadline:
            break
        if target >= best_score:
            continue
        optimize(target)
    
    if best_result is None:
        best_result = (inner_343, (0.0, 0.0), 4.0, 0.0)
    
    return best_result[0], best_result[1], best_result[2], best_result[3]
