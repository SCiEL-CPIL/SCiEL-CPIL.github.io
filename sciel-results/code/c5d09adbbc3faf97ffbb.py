import itertools
import math
import time

N = 11
sqrt3 = math.sqrt(3)

# Lattice vectors for flat-top hexagons
v1 = (sqrt3, 0.0)
v2 = (sqrt3/2.0, 1.5)

# Generate points within radius 3 (axial coords)
# Axial to Cartesian: x = sqrt3 * q + sqrt3/2 * r, y = 1.5 * r
points = []
for q in range(-3, 4):
    for r in range(-3, 4):
        dist = (abs(q) + abs(r) + abs(q+r)) / 2.0
        if dist <= 3.0:
            x = sqrt3 * q + sqrt3/2.0 * r
            y = 1.5 * r
            points.append((x, y))

# We have 19 points.
# Pre-compute projections
u1 = (0.0, 1.0)
u2 = (sqrt3/2.0, 0.5)
u3 = (sqrt3/2.0, -0.5)

projections = []
for p in points:
    x, y = p
    p1 = x * u1[0] + y * u1[1]
    p2 = x * u2[0] + y * u2[1]
    p3 = x * u3[0] + y * u3[1]
    projections.append((p1, p2, p3, x, y))

def construct():
    best_score = float('inf')
    best_combo = None
    
    start_time = time.time()
    limit = 100.0 # Safety margin
    
    # Iterate over combinations
    for combo in itertools.combinations(range(len(points)), N):
        if time.time() - start_time > limit:
            break
        
        min_p1 = min(projections[i][0] for i in combo)
        max_p1 = max(projections[i][0] for i in combo)
        span1 = max_p1 - min_p1
        
        min_p2 = min(projections[i][1] for i in combo)
        max_p2 = max(projections[i][1] for i in combo)
        span2 = max_p2 - min_p2
        
        min_p3 = min(projections[i][2] for i in combo)
        max_p3 = max(projections[i][2] for i in combo)
        span3 = max_p3 - min_p3
        
        max_span = max(span1, span2, span3)
        outer_side = max_span / sqrt3 + 1.0
        
        if outer_side < best_score:
            best_score = outer_side
            best_combo = combo

    if best_combo is not None:
        inner = []
        for i in best_combo:
            x, y = points[i]
            inner.append([x, y, 0.0])
        
        m1 = (min(projections[i][0] for i in best_combo) + max(projections[i][0] for i in best_combo)) / 2.0
        m2 = (min(projections[i][1] for i in best_combo) + max(projections[i][1] for i in best_combo)) / 2.0
        
        cy = m1
        cx = (m2 - 0.5 * cy) / (sqrt3 / 2.0)
        
        # Add small epsilon to side length to ensure validity
        return inner, (cx, cy), best_score + 1e-6, 0.0
    else:
        return [[0.0, 0.0, 0.0]], (0.0, 0.0), 1.0, 0.0
