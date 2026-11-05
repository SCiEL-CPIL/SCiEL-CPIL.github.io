"""Structural change: optimize packing of 11 unit hexagons using grid search
over lattice perturbations and outer hexagon rotation."""
import math
import time
import random
import numpy as np

N = 11
DEADLINE = 140.0
random.seed(42)

def hexagon_vertices(angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    base = [(math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))) for k in range(6)]
    return [(ca * x - sa * y, sa * x + ca * y) for (x, y) in base]

def point_in_polygon(px, py, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygons_overlap(p1, p2):
    for (x, y) in p1:
        if point_in_polygon(x, y, p2):
            return True
    for (x, y) in p2:
        if point_in_polygon(x, y, p1):
            return True
    n1, n2 = len(p1), len(p2)
    for i in range(n1):
        x1a, y1a = p1[i]
        x1b, y1b = p1[(i + 1) % n1]
        for j in range(n2):
            x2a, y2a = p2[j]
            x2b, y2b = p2[(j + 1) % n2]
            d1x, d1y = x1b - x1a, y1b - y1a
            d2x, d2y = x2b - x2a, y2b - y2a
            denom = d1x * d2y - d1y * d2x
            if abs(denom) < 1e-12:
                continue
            t = ((x2a - x1a) * d2y - (y2a - y1a) * d2x) / denom
            u = ((x2a - x1a) * d1y - (y2a - y1a) * d1x) / denom
            if 0 <= t <= 1 and 0 <= u <= 1:
                return True
    return False

def outer_hexagon_side_length(centers, angles, outer_center, outer_angle_deg):
    ocx, ocy = outer_center
    sqrt3 = math.sqrt(3.0)
    min_side = 0.0
    for (cx, cy), ang in zip(centers, angles):
        verts = hexagon_vertices(ang)
        for (vx, vy) in verts:
            px = cx + vx
            py = cy + vy
            for k in range(6):
                nk = math.radians(outer_angle_deg + 30 + 60 * k)
                proj = (px - ocx) * math.cos(nk) + (py - ocy) * math.sin(nk)
                required = proj * 2.0 / sqrt3
                if required > min_side:
                    min_side = required
    return max(0.0, min_side)

def score_configuration(centers, angles, outer_center, outer_angle_deg):
    n = len(centers)
    polys = []
    for (cx, cy), ang in zip(centers, angles):
        verts = [(cx + vx, cy + vy) for (vx, vy) in hexagon_vertices(ang)]
        polys.append(verts)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(polys[i], polys[j]):
                return float('inf')
    return outer_hexagon_side_length(centers, angles, outer_center, outer_angle_deg)

def construct():
    t0 = time.time()
    sqrt3 = math.sqrt(3.0)
    
    # Hexagonal lattice: 11 points closest to origin in hex lattice with spacing 2
    lattice_points = []
    for row in range(-4, 5):
        for col in range(-5, 6):
            x = (col + row / 2.0) * 2.0
            y = row * sqrt3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    initial_centers = lattice_points[:N]
    initial_angles = [0.0] * N
    
    oc = (0.0, 0.0)
    best_score = score_configuration(initial_centers, initial_angles, oc, 0.0)
    best_centers = initial_centers[:]
    best_angles = initial_angles[:]
    best_oa = 0.0
    
    if best_score == float('inf'):
        best_score = 50.0
    
    # Try different outer rotations
    for oa_deg in range(0, 60, 5):
        if time.time() - t0 > DEADLINE:
            break
        s = score_configuration(initial_centers, initial_angles, oc, float(oa_deg))
        if s < best_score:
            best_score = s
            best_oa = float(oa_deg)
    
    # Perturbation search
    for trial in range(500):
        if time.time() - t0 > DEADLINE:
            break
        centers = []
        for (cx, cy) in best_centers:
            dx = (random.random() - 0.5) * 0.3
            dy = (random.random() - 0.5) * 0.3
            centers.append((cx + dx, cy + dy))
        angles = [random.choice([0.0, 15.0, 30.0]) for _ in range(N)]
        s = score_configuration(centers, angles, oc, best_oa)
        if s < best_score:
            best_score = s
            best_centers = centers[:]
            best_angles = angles[:]
    
    inner = [[c[0], c[1], a] for c, a in zip(best_centers, best_angles)]
    return inner, oc, best_score, best_oa
