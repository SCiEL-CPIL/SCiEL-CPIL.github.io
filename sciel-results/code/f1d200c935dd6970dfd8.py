"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses genetic algorithm to evolve contact graph on hexagonal lattice,
combined with smooth optimization for final polishing.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = time.time() + 135.0  # safety margin before 150s kill
random.seed(42)
np.random.seed(42)

# --- Geometry helpers ---

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def hex_vertices_fast(cx, cy, s, angle_rad):
    """Fast vertex computation using precomputed cos/sin."""
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def compute_hexagon_moments(cx, cy, s, angle_rad):
    """Compute center and moments for a hexagon."""
    return (cx, cy, s, angle_rad)

def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon (CCW or CW)."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def point_in_convex_polygon_fast(px, py, poly, normals):
    """Fast point-in-convex-polygon using precomputed edge normals."""
    for i in range(6):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % 6]
        nx, ny = normals[i]
        cross = nx * (py - y1) + ny * (px - x1)
        if cross < -1e-9:
            return False
    return True

def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # normal to edge
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            # project A
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            # project B
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False  # separating axis found
    return True

def polygons_distance_sat(A, B, eps=1e-7):
    """Compute minimum distance between two convex polygons using SAT.
    Returns 0.0 if overlapping, positive value otherwise."""
    min_dist = float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-12:
                continue
            nx, ny = nx/norm, ny/norm
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - eps:
                gap = minB - maxA
                if gap < min_dist:
                    min_dist = gap
            elif maxB < minA - eps:
                gap = minA - maxB
                if gap < min_dist:
                    min_dist = gap
            else:
                return 0.0  # overlapping
    return min_dist if min_dist < float('inf') else 0.0

def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping.
    inner: list of (x, y, angle_deg) with unit side."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
        
    # Precompute bounding boxes for inner polys to speed up overlap check
    bboxes = []
    for p in polys:
        min_x = min(px for px, py in p)
        max_x = max(px for px, py in p)
        min_y = min(py for px, py in p)
        max_y = max(py for px, py in p)
        bboxes.append((min_x, min_y, max_x, max_y))
        
    # inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    # pairwise non-overlap (allow touching)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            # Bounding box check
            if (bboxes[i][2] < bboxes[j][0] - eps or 
                bboxes[j][2] < bboxes[i][0] - eps or 
                bboxes[i][3] < bboxes[j][1] - eps or 
                bboxes[j][3] < bboxes[i][1] - eps):
                continue
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def check_validity_fast(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Fast validity check using precomputed structures."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    # inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    # pairwise non-overlap
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside.
    For a regular hexagon of side s, apothem = s * sqrt(3)/2.
    A point is inside iff for each of the 6 outward normals n_k, (p - c)·n_k <= apothem.
    So S >= (2/sqrt(3)) * max over all vertices and all k of (p-c)·n_k.
    """
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj

def compute_min_outer_side_smooth(inner, outer_center, outer_angle, beta=150.0):
    """Smooth approximation of min outer side using log-sum-exp."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Collect all projections
    projs = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                projs.append(proj)
    
    if not projs:
        return 0.0
    
    # Log-sum-exp for smooth max
    max_p = max(projs)
    
    # Use log-sum-exp: log(sum(exp(beta*p))) / beta approximates max(p)
    exps = [math.exp(beta * (p - max_p)) for p in projs]
    log_sum = math.log(sum(exps))
    smooth_max = max_p + log_sum / beta
    
    return (2.0 / math.sqrt(3.0)) * max(0.0, smooth_max)

def compute_overlap_penalty(inner, outer_center, outer_side, outer_angle):
    """Compute a smooth penalty for overlaps and boundary violations."""
    penalty = 0.0
    polys = [hex_vertices(x, y, 1.0, ang) for x, y, ang in inner]
    # Pairwise overlap penalty
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            dist = polygons_distance_sat(polys[i], polys[j])
            if dist < 0:
                penalty += (-dist) ** 2  # quadratic penalty for penetration
    # Boundary violation penalty
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                # Approximate penetration depth
                penalty += 0.1
    return penalty

# --- Topology builders ---

def build_ring12(scale, inner_rot, outer_rot):
    """Ring of 12 hexagons around center."""
    inner = []
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        inner.append((x, y, inner_rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot):
    """Two concentric rings of 6, no center."""
    inner = []
    for k in range(6):
        theta = math.radians(rot1 + k * 60.0)
        x = scale1 * math.cos(theta)
        y = scale1 * math.sin(theta)
        inner.append((x, y, rot1))
    for k in range(6):
        theta = math.radians(rot2 + k * 60.0)
        x = scale2 * math.cos(theta)
        y = scale2 * math.sin(theta)
        inner.append((x, y, rot2))
    return inner, (0.0, 0.0), outer_rot

def build_rect3x4(scale_x, scale_y, rot, outer_rot):
    """3x4 rectangular grid."""
    inner = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * scale_x
            y = (j - 1.5) * scale_y
            inner.append((x, y, rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex_lattice(rot, outer_rot):
    """Hexagonal lattice: 1 center + 6 ring + 5 more arranged."""
    inner = []
    inner.append((0.0, 0.0, rot))
    for k in range(6):
        theta = math.radians(rot + k * 60.0)
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        inner.append((x, y, rot))
    for k in range(5):
        theta = math.radians(rot + 30.0 + k * 60.0)
        x = 2.0 * math.sqrt(3) * math.cos(theta)
        y = 2.0 * math.sqrt(3) * math.sin(theta)
        inner.append((x, y, rot))
    return inner, (0.0, 0.0), outer_rot

def build_linear_chain(rot, outer_rot):
    """Linear chain of 12 hexagons."""
    inner = []
    for k in range(12):
        x = (k - 5.5) * 2.0
        y = 0.0
        inner.append((x, y, rot))
    return inner, (0.0, 0.0), outer_rot

def build_hexagonal_cluster(rot, outer_rot):
    """Compact hexagonal cluster: 1 center + 6 + 5."""
    inner = []
    inner.append((0.0, 0.0, rot))
    for k in range(6):
        theta = math.radians(rot + k * 60.0)
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        inner.append((x, y, rot))
    positions = [
        (2.0, 2.0 * math.sqrt(3)),
        (-2.0, 2.0 * math.sqrt(3)),
        (4.0, 0.0),
        (-4.0, 0.0),
        (2.0, -2.0 * math.sqrt(3)),
    ]
    for (px, py) in positions:
        inner.append((px, py, rot))
    return inner, (0.0, 0.0), outer_rot

# --- Genetic Algorithm for Contact Graph Evolution ---

# Define hexagonal lattice positions for 12 hexagons
# Using axial coordinates: (q, r) where distance = max(|q|, |r|, |q+r|)
def generate_lattice_positions(n_hex=12, max_ring=3):
    """Generate positions on hexagonal lattice."""
    positions = []
    # Center
    positions.append((0, 0))
    # Ring 1: 6 positions
    for q in range(-1, 2):
        for r in range(-1, 2):
            if max(abs(q), abs(r), abs(q+r)) == 1:
                positions.append((q, r))
    # Ring 2: 12 positions (take first 5 to reach 12 total)
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                ring2.append((q, r))
    positions.extend(ring2[:5])
    return positions[:n_hex]

def axial_to_cartesian(q, r, scale=2.0):
    """Convert axial hex coordinates to cartesian."""
    x = scale * (q + r / 2.0)
    y = scale * (math.sqrt(3) / 2.0) * r
    return x, y

def cartesian_to_axial(x, y, scale=2.0):
    """Convert cartesian to axial hex coordinates (approximate)."""
    r = y / (scale * math.sqrt(3) / 2.0)
    q = (x / scale) - r / 2.0
    return round(q), round(r)

class ContactGraph:
    """Represents a packing configuration as a set of relative positions."""
    
    def __init__(self, positions, rotations=None, scale=2.0, global_rot=0.0):
        """
        positions: list of (q, r) axial coordinates
        rotations: list of rotation angles for each hexagon
        scale: distance between adjacent hex centers
        global_rot: global rotation angle
        """
        self.positions = positions
        self.rotations = rotations if rotations else [0.0] * len(positions)
        self.scale = scale
        self.global_rot = global_rot
    
    def get_cartesian(self):
        """Get cartesian positions with global rotation."""
        inner = []
        gr = math.radians(self.global_rot)
        for i, (q, r) in enumerate(self.positions):
            x, y = axial_to_cartesian(q, r, self.scale)
            # Apply global rotation
            xr = x * math.cos(gr) - y * math.sin(gr)
            yr = x * math.sin(gr) + y * math.cos(gr)
            inner.append((xr, yr, self.rotations[i] + self.global_rot))
        return inner
    
    def evaluate(self, outer_angle=0.0):
        """Evaluate this configuration: return (side_length, is_valid)."""
        inner = self.get_cartesian()
        side = compute_min_outer_side(inner, (0.0, 0.0), outer_angle)
        valid = check_validity(inner, (0.0, 0.0), side, outer_angle)
        return side, valid
    
    def __repr__(self):
        return f"ContactGraph(scale={self.scale:.3f}, rot={self.global_rot:.1f})"

def crossover(parent1, parent2):
    """Crossover two contact graphs."""
    # Blend positions
    child_positions = []
    for i in range(len(parent1.positions)):
        q1, r1 = parent1.positions[i]
        q2, r2 = parent2.positions[i]
        if random.random() < 0.5:
            child_positions.append((q1, r1))
        else:
            child_positions.append((q2, r2))
    
    # Blend scale and rotation
    scale = random.uniform(min(parent1.scale, parent2.scale), max(parent1.scale, parent2.scale))
    global_rot = random.uniform(min(parent1.global_rot, parent2.global_rot), max(parent1.global_rot, parent2.global_rot))
    if random.random() < 0.5:
        global_rot = parent1.global_rot
    else:
        global_rot = parent2.global_rot
    
    # Blend rotations
    child_rotations = []
    for i in range(len(parent1.rotations)):
        if random.random() < 0.5:
            child_rotations.append(parent1.rotations[i])
        else:
            child_rotations.append(parent2.rotations[i])
    
    return ContactGraph(child_positions, child_rotations, scale, global_rot)

def mutate(graph, mutation_rate=0.3):
    """Mutate a contact graph."""
    new_positions = list(graph.positions)
    new_rotations = list(graph.rotations)
    new_scale = graph.scale
    new_global_rot = graph.global_rot
    
    # Mutate positions
    for i in range(len(new_positions)):
        if random.random() < mutation_rate:
            q, r = new_positions[i]
            # Move to adjacent cell
            dq = random.choice([-1, 0, 1])
            dr = random.choice([-1, 0, 1])
            new_positions[i] = (q + dq, r + dr)
    
    # Mutate scale
    if random.random() < mutation_rate:
        new_scale *= (1 + random.uniform(-0.1, 0.1))
        new_scale = max(1.5, min(3.0, new_scale))
    
    # Mutate global rotation
    if random.random() < mutation_rate:
        new_global_rot += random.uniform(-15, 15)
        new_global_rot = (new_global_rot + 180) % 360 - 180
    
    # Mutate individual rotations
    for i in range(len(new_rotations)):
        if random.random() < mutation_rate * 0.5:
            new_rotations[i] += random.uniform(-30, 30)
    
    return ContactGraph(new_positions, new_rotations, new_scale, new_global_rot)

def genetic_algorithm_search(initial_graphs, population_size=20, generations=50, time_budget=30.0):
    """Run genetic algorithm to evolve contact graphs."""
    start_time = time.time()
    
    # Initialize population
    population = list(initial_graphs)
    while len(population) < population_size:
        parent = random.choice(population)
        population.append(mutate(parent))
    
    best_graph = None
    best_side = float('inf')
    
    for gen in range(generations):
        if time.time() - start_time > time_budget:
            break
        
        # Evaluate population
        scored = []
        for graph in population:
            side, valid = graph.evaluate()
            if valid:
                penalty = 0
            else:
                penalty = 5.0
            scored.append((side + penalty, graph, valid))
        
        # Find best
        scored.sort(key=lambda x: x[0])
        if scored[0][1] and scored[0][2]:
            if scored[0][0] < best_side:
                best_side = scored[0][0]
                best_graph = scored[0][1]
        
        # Selection: keep top half
        elite_count = max(2, population_size // 4)
        new_population = [g for _, g, _ in scored[:elite_count]]
        
        # Fill rest with crossover + mutation
        while len(new_population) < population_size:
            p1, p2 = random.sample(population, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    return best_graph, best_side

# --- Optimization ---

def optimize_topology(topology, x0, bounds, outer_rot, time_budget):
    """Optimize a single topology using Nelder-Mead."""
    start_time = time.time()
    
    def objective(params):
        if topology == 'ring12':
            scale, inner_rot = params
            if scale < 0.5 or scale > 3.0:
                return 1000.0
            inner, center, orot = build_ring12(scale, inner_rot, outer_rot)
        elif topology == 'hex6+6':
            scale1, scale2, rot1, rot2 = params
            if scale1 < 0.5 or scale2 < 0.5 or scale2 < scale1 + 0.3:
                return 1000.0
            inner, center, orot = build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot)
        elif topology == 'rect3x4':
            sx, sy, rot = params
            if sx < 1.5 or sy < 1.5:
                return 1000.0
            inner, center, orot = build_rect3x4(sx, sy, rot, outer_rot)
        elif topology == 'hex_lattice':
            rot = params[0]
            inner, center, orot = build_hex_lattice(rot, outer_rot)
        elif topology == 'linear_chain':
            rot = params[0]
            inner, center, orot = build_linear_chain(rot, outer_rot)
        elif topology == 'hexagonal_cluster':
            rot = params[0]
            inner, center, orot = build_hexagonal_cluster(rot, outer_rot)
        else:
            return 1000.0
        
        side = compute_min_outer_side(inner, center, orot)
        if not check_validity(inner, center, side, orot):
            return side + 10.0
        return side
    
    best_x = x0
    best_f = objective(x0)
    
    # Coarse grid search
    n_grid = 6
    if len(x0) == 2:
        grid1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        grid2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
        for g1 in grid1:
            for g2 in grid2:
                f = objective([g1, g2])
                if f < best_f:
                    best_f = f
                    best_x = [g1, g2]
                if time.time() - start_time > time_budget:
                    return best_x, best_f
    elif len(x0) == 3:
        grid1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        grid2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
        grid3 = np.linspace(bounds[2][0], bounds[2][1], n_grid)
        for g1 in grid1:
            for g2 in grid2:
                for g3 in grid3:
                    f = objective([g1, g2, g3])
                    if f < best_f:
                        best_f = f
                        best_x = [g1, g2, g3]
                    if time.time() - start_time > time_budget:
                        return best_x, best_f
    elif len(x0) == 4:
        grid1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        grid2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
        grid3 = np.linspace(bounds[2][0], bounds[2][1], n_grid)
        grid4 = np.linspace(bounds[3][0], bounds[3][1], n_grid)
        for g1 in grid1:
            for g2 in grid2:
                if g2 < g1 + 0.3:
                    continue
                for g3 in grid3:
                    for g4 in grid4:
                        f = objective([g1, g2, g3, g4])
                        if f < best_f:
                            best_f = f
                            best_x = [g1, g2, g3, g4]
                        if time.time() - start_time > time_budget:
                            return best_x, best_f
    elif len(x0) == 1:
        grid1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        for g1 in grid1:
            f = objective([g1])
            if f < best_f:
                best_f = f
                best_x = [g1]
            if time.time() - start_time > time_budget:
                return best_x, best_f
    
    # Nelder-Mead refinement
    try:
        result = minimize(
            objective,
            best_x,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'maxfev': 5000,
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
        )
        if result.fun < best_f:
            best_f = result.fun
            best_x = result.x
    except Exception:
        pass
    
    return best_x, best_f

def polish_configuration(inner, center, outer_angle, time_budget=10.0):
    """High-precision polish using SLSQP on all parameters."""
    start_time = time.time()
    
    # Parameters: x, y, angle for each hexagon + outer center x, y + outer angle
    # Total: 12*3 + 2 + 1 = 39 parameters
    n_params = len(inner) * 3 + 3
    
    def params_to_inner(params):
        inner_out = []
        for i in range(len(inner)):
            x = params[i*3]
            y = params[i*3+1]
            ang = params[i*3+2]
            inner_out.append((x, y, ang))
        outer_cx = params[-3]
        outer_cy = params[-2]
        outer_ang = params[-1]
        return inner_out, (outer_cx, outer_cy), outer_ang
    
    def objective(params):
        if time.time() - start_time > time_budget:
            return float('inf')
        
        inner_out, outer_c, outer_a = params_to_inner(params)
        
        side = compute_min_outer_side_smooth(inner_out, outer_c, outer_a, beta=150.0)
        
        # Check validity
        if not check_validity(inner_out, outer_c, side, outer_a):
            penalty = compute_overlap_penalty(inner_out, outer_c, side, outer_a)
            return side + 5.0 * penalty
        
        return side
    
    # Initial parameters
    x0 = []
    for x, y, ang in inner:
        x0.extend([x, y, ang])
    x0.extend([center[0], center[1], outer_angle])
    
    # Bounds
    bounds = []
    for _ in range(len(inner)):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((-180, 180))
    bounds.append((-10, 10))
    bounds.append((-10, 10))
    bounds.append((-180, 180))
    
    # Compute initial objective value once
    init_val = objective(x0)
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'maxls': 50}
        )
        if result.fun < init_val:
            inner_out, outer_c, outer_a = params_to_inner(result.x)
            # Verify with exact computation
            exact_side = compute_min_outer_side(inner_out, outer_c, outer_a)
            if check_validity(inner_out, outer_c, exact_side, outer_a):
                return inner_out, outer_c, outer_a, exact_side
            else:
                # Try the smooth side value
                if check_validity(inner_out, outer_c, result.fun, outer_a):
                    return inner_out, outer_c, outer_a, result.fun
    except Exception:
        pass
    
    return inner, center, outer_angle, init_val

def final_polish(inner, center, outer_angle, time_budget=8.0):
    """Final high-precision Nelder-Mead polish on the best configuration."""
    start_time = time.time()
    
    def params_to_inner(params):
        inner_out = []
        for i in range(len(inner)):
            x = params[i*3]
            y = params[i*3+1]
            ang = params[i*3+2]
            inner_out.append((x, y, ang))
        outer_cx = params[-3]
        outer_cy = params[-2]
        outer_ang = params[-1]
        return inner_out, (outer_cx, outer_cy), outer_ang
    
    def objective(params):
        if time.time() - start_time > time_budget:
            return float('inf')
        inner_out, outer_c, outer_a = params_to_inner(params)
        side = compute_min_outer_side(inner_out, outer_c, outer_a)
        if not check_validity(inner_out, outer_c, side, outer_a):
            return side + 10.0
        return side
    
    x0 = []
    for x, y, ang in inner:
        x0.extend([x, y, ang])
    x0.extend([center[0], center[1], outer_angle])
    
    init_val = objective(x0)
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 10000,
                'maxfev': 25000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        if result.fun < init_val:
            inner_out, outer_c, outer_a = params_to_inner(result.x)
            exact_side = compute_min_outer_side(inner_out, outer_c, outer_a)
            if check_validity(inner_out, outer_c, exact_side, outer_a):
                return inner_out, outer_c, outer_a, exact_side
    except Exception:
        pass
    
    return inner, center, outer_angle, init_val

def construct():
    start_time = time.time()
    
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_side = float('inf')
    
    # Phase 1: Try predefined topologies with optimization
    topologies = [
        ('ring12', [2.0, 0.0], [(0.8, 2.5), (0, 360)]),
        ('hex6+6', [1.5, 3.0, 0.0, 30.0], [(0.8, 2.5), (1.5, 4.0), (0, 360), (0, 360)]),
        ('rect3x4', [2.0, 2.0, 0.0], [(1.5, 3.0), (1.5, 3.0), (0, 360)]),
        ('hex_lattice', [0.0], [(0, 360)]),
        ('linear_chain', [0.0], [(0, 360)]),
        ('hexagonal_cluster', [0.0], [(0, 360)]),
    ]
    
    for topology, x0, bounds in topologies:
        if time.time() > DEADLINE - 30:
            break
        
        remaining = DEADLINE - time.time()
        time_budget = min(10.0, remaining * 0.2)
        
        x_opt, f_opt = optimize_topology(topology, x0, bounds, 0.0, time_budget)
        
        if f_opt < best_side:
            best_side = f_opt
            if topology == 'ring12':
                scale, inner_rot = x_opt
                inner, center, outer_angle = build_ring12(scale, inner_rot, 0.0)
            elif topology == 'hex6+6':
                scale1, scale2, rot1, rot2 = x_opt
                inner, center, outer_angle = build_hex6_plus6(scale1, scale2, rot1, rot2, 0.0)
            elif topology == 'rect3x4':
                sx, sy, rot = x_opt
                inner, center, outer_angle = build_rect3x4(sx, sy, rot, 0.0)
            elif topology == 'hex_lattice':
                rot = x_opt[0]
                inner, center, outer_angle = build_hex_lattice(rot, 0.0)
            elif topology == 'linear_chain':
                rot = x_opt[0]
                inner, center, outer_angle = build_linear_chain(rot, 0.0)
            elif topology == 'hexagonal_cluster':
                rot = x_opt[0]
                inner, center, outer_angle = build_hexagonal_cluster(rot, 0.0)
            else:
                continue
            
            if check_validity(inner, center, best_side, outer_angle):
                best_inner = inner
                best_center = center
                best_outer_angle = outer_angle
    
    # Phase 2: Genetic algorithm on hexagonal lattice
    if time.time() < DEADLINE - 20:
        # Generate initial population from lattice positions
        lattice_positions = generate_lattice_positions(12, 3)
        
        initial_graphs = []
        # Start with standard lattice
        initial_graphs.append(ContactGraph(lattice_positions, scale=2.0, global_rot=0.0))
        # Try different scales
        for scale in [1.8, 2.0, 2.2, 2.4]:
            initial_graphs.append(ContactGraph(lattice_positions, scale=scale, global_rot=0.0))
        # Try different rotations
        for rot in [0, 30, 60, 90, 150]:
            initial_graphs.append(ContactGraph(lattice_positions, scale=2.0, global_rot=rot))
        
        ga_time = min(25.0, DEADLINE - time.time() - 15)
        if ga_time > 5:
            ga_best, ga_side = genetic_algorithm_search(
                initial_graphs, 
                population_size=15, 
                generations=40, 
                time_budget=ga_time
            )
            
            if ga_best and ga_side < best_side:
                inner = ga_best.get_cartesian()
                side = compute_min_outer_side(inner, (0.0, 0.0), 0.0)
                if check_validity(inner, (0.0, 0.0), side, 0.0):
                    best_inner = inner
                    best_center = (0.0, 0.0)
                    best_outer_angle = 0.0
                    best_side = side
    
    # Phase 3: Polish the best configuration
    if best_inner is not None and time.time() < DEADLINE - 15:
        polish_time = min(12.0, DEADLINE - time.time() - 8)
        if polish_time > 3:
            polished_inner, polished_center, polished_angle, polished_side = polish_configuration(
                best_inner, best_center, best_outer_angle, polish_time
            )
            if polished_side < best_side:
                if check_validity(polished_inner, polished_center, polished_side, polished_angle):
                    best_inner = polished_inner
                    best_center = polished_center
                    best_outer_angle = polished_angle
                    best_side = polished_side
    
    # Phase 4: Final high-precision Nelder-Mead polish
    if best_inner is not None and time.time() < DEADLINE - 5:
        fm_time = min(8.0, DEADLINE - time.time() - 2)
        if fm_time > 2:
            fm_inner, fm_center, fm_angle, fm_side = final_polish(
                best_inner, best_center, best_outer_angle, fm_time
            )
            if fm_side < best_side:
                if check_validity(fm_inner, fm_center, fm_side, fm_angle):
                    best_inner = fm_inner
                    best_center = fm_center
                    best_outer_angle = fm_angle
                    best_side = fm_side
    
    # Final fallback if nothing worked
    if best_inner is None:
        # Fallback: 3x4 grid
        grid_xs = [-2.0, 0.0, 2.0]
        grid_ys = [0.0, math.sqrt(3), 2*math.sqrt(3), 3*math.sqrt(3)]
        positions = []
        for y in grid_ys:
            for x in grid_xs:
                positions.append((x, y, 0.0))
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        best_inner = [(px - cx, py - cy, angle) for px, py, angle in positions]
        best_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_min_outer_side(best_inner, best_center, best_outer_angle)
    
    # Compute final side and verify
    final_side = compute_min_outer_side(best_inner, best_center, best_outer_angle)
    
    if not check_validity(best_inner, best_center, final_side, best_outer_angle):
        # Last resort fallback
        grid_xs = [-2.0, 0.0, 2.0]
        grid_ys = [0.0, math.sqrt(3), 2*math.sqrt(3), 3*math.sqrt(3)]
        positions = []
        for y in grid_ys:
            for x in grid_xs:
                positions.append((x, y, 0.0))
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        best_inner = [(px - cx, py - cy, angle) for px, py, angle in positions]
        best_center = (0.0, 0.0)
        best_outer_angle = 0.0
        final_side = compute_min_outer_side(best_inner, best_center, best_outer_angle)
    
    inner_list = []
    for x, y, angle in best_inner:
        inner_list.append([float(x), float(y), float(angle)])
    
    return inner_list, (float(best_center[0]), float(best_center[1])), float(final_side), float(best_outer_angle)
