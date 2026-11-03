import math
import time

N = 11

def construct():
    start_time = time.time()
    
    # Find smallest r such that 1 + 3*r*(r+1) >= N
    r = 0
    while 1 + 3 * r * (r + 1) < N:
        r += 1
    
    side = 2 * r + 1
    
    # Generate hexagonal positions (flat-top hexagons)
    positions = []
    for q in range(-r, r + 1):
        qmin = max(-q, -r)
        qmax = min(-q + r, r)
        for q2 in range(qmin, qmax + 1):
            x = 1.5 * q
            y = math.sqrt(3) * (q2 + 0.5 * q)
            positions.append((x, y))
    
    # Take N closest to center
    positions.sort(key=lambda p: p[0]**2 + p[1]**2)
    positions = positions[:N]
    
    # Center the arrangement
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    positions = [(x - cx, y - cy) for x, y in positions]
    
    inner = [[x, y, 0.0] for x, y in positions]
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
