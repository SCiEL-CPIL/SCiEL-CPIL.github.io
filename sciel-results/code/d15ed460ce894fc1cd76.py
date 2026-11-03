import numpy as np
from math import sqrt, cos, sin, radians

N = 11

def construct():
    # Place 11 unit hexagons in a hexagonal arrangement
    d = sqrt(3)  # distance between adjacent aligned hex centers
    
    positions = []
    positions.append((0.0, 0.0))  # central
    for i in range(6):  # 6 surrounding
        angle = radians(60 * i)
        positions.append((d * cos(angle), d * sin(angle)))
    # 4 additional hexagons
    for i in [0, 2, 4, 6]:
        angle = radians(60 * i)
        positions.append((2 * d * cos(angle), 2 * d * sin(angle)))
    
    # Compute minimum enclosing hexagon side length
    S = 0.0
    inradius_factor = sqrt(3) / 2  # inradius = S * inradius_factor
    
    for (px, py) in positions:
        for k in range(6):
            angle = radians(60 * k)
            vx = px + cos(angle)
            vy = py + sin(angle)
            for m in range(6):
                nangle = radians(60 * m)
                nx = cos(nangle)
                ny = sin(nangle)
                val = vx * nx + vy * ny
                required_S = val / inradius_factor
                if required_S > S:
                    S = required_S
    
    inner = [[px, py, 0.0] for (px, py) in positions]
    return inner, (0.0, 0.0), S, 0.0
