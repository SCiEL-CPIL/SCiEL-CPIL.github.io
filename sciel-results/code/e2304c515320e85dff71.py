import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_vertices, angle_deg):
    # Same as before
    pass

def _optimize_outer_angle(vertices, time_limit=5.0):
    # Use minimize_scalar on the result of _min_enclosing_hex_side_lp
    pass

def construct():
    # Generate lattice points
    # Define angle patterns
    # For each pattern, optimize outer angle
    # Return best
    pass
