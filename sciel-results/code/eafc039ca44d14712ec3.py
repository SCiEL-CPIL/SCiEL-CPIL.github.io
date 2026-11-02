"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 12
DEADLINE = 115.0  # safety margin before 120s (and 150s kill)

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    best = _initial_guess()
    best_side = _required_side(best)
    
    # Try a few local optimizations with different initial configurations
    configs = [
        _hex_grid_4x3(),
        _hex_grid_3x4(),
        _ring_config(),
        _compact_config(),
    ]
    
    for cfg in configs:
        if time.time() - t0 > DEADLINE - 5:
            break
        cfg, side = _local_optimize(cfg, t0)
        if side < best_side:
            best, best_side = cfg, side
    
    if time.time() - t0 > DEADLINE - 3:
        return best, (0.0, 0.0), best_side, 0.0
    
    # Final polish
    best, best_side = _polish(best, t0)
    
    return best, (0.0, 0.0), best_side, 0.0


def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts


def _hex_support(cx, cy, side, angle_deg, theta):
    """Support function of a hexagon in direction theta."""
    a = math.radians(angle_deg)
    # Support function of a regular hexagon with side s centered at origin, rotated by a:
    # h(theta) = s * max over vertices of cos(theta - vertex_angle)
    # For a regular hexagon, the support function is periodic with period pi/3
    # h(theta) = s / cos((theta - a) mod pi/3 - pi/6)  ... actually let's compute directly
    # The apothem is s * sqrt(3)/2, and the support function of a regular hexagon
    # with "flat top" orientation (vertex at angle 0) is:
    # h(t) = s * sqrt(3) / (2 * cos((t mod pi/3) - pi/6))
    # But this is for the case where vertices are at multiples of pi/3.
    # Let me just compute from vertices to be safe and simple.
    best = -1e18
    for i in range(6):
        theta_i = a + i * math.pi / 3.0
        val = side * math.cos(theta_i) * math.cos(theta - theta_i) + side * math.sin(theta_i) * math.sin(theta - theta_i)
        # Actually simpler: dot product of vertex direction with direction theta
        # vertex at angle theta_i has position (s*cos(theta_i), s*sin(theta_i))
        # support = s * cos(theta - theta_i)
        val = side * math.cos(theta - theta_i)
        if val > best:
            best = val
    return cx * math.cos(theta) + cy * math.sin(theta) + best


def _required_side(cfg):
    """Compute the minimum outer hexagon side length needed to contain all inner hexagons.
    
    cfg: list of (x, y, angle_deg) for N hexagons of side 1.
    We find the minimum side S of a regular hexagon (centered at origin, any rotation)
    that contains all inner hexagons.
    
    For a regular hexagon with side S and rotation phi, the support function is:
    H(theta) = S * f(theta - phi) where f is the support function of a unit hexagon.
    
    We need: for all theta, S * f(theta - phi) >= h_inner(theta)
    So S >= max_theta h_inner(theta) / f(theta - phi)
    
    We minimize over phi.
    """
    # Sample directions densely
    n_theta = 720
    thetas = [i * 2 * math.pi / n_theta for i in range(n_theta)]
    
    # Compute inner support function h_inner(theta) for each theta
    h_inner = []
    for theta in thetas:
        h = 0.0
        for (cx, cy, ang) in cfg:
            h = max(h, _hex_support(cx, cy, 1.0, ang, theta))
        h_inner.append(h)
    
    # For each candidate outer rotation phi, compute required S
    best_S = 1e18
    n_phi = 360
    for j in range(n_phi):
        phi = j * 2 * math.pi / n_phi
        max_ratio = 0.0
        for i in range(n_theta):
            theta = thetas[i]
            # Support function of unit hexagon rotated by phi in direction theta
            h_unit = _hex_support(0.0, 0.0, 1.0, math.degrees(phi), theta)
            if h_unit <= 1e-12:
                continue
            ratio = h_inner[i] / h_unit
            if ratio > max_ratio:
                max_ratio = ratio
        if max_ratio < best_S:
            best_S = max_ratio
    
    return best_S


def _local_optimize(cfg, t0, max_iters=5000, step=0.1):
    """Simple gradient-free local optimization using coordinate descent."""
    cfg = [list(c) for c in cfg]
    current_side = _required_side(cfg)
    
    # Use a simple perturb-and-check approach
    for it in range(max_iters):
        if time.time() - t0 > DEADLINE - 5:
            break
        improved = False
        for i in range(N):
            for coord in range(3):  # x, y, angle
                for delta in [step, -step, step/2, -step/2]:
                    cfg[i][coord] += delta
                    new_side = _required_side(cfg)
                    if new_side < current_side - 1e-8:
                        current_side = new_side
                        improved = True
                        break
                    else:
                        cfg[i][coord] -= delta
                if improved:
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    return [tuple(c) for c in cfg], current_side


def _polish(cfg, t0):
    """High-precision polish."""
    cfg = [list(c) for c in cfg]
    current_side = _required_side(cfg)
    step = 0.01
    
    for it in range(2000):
        if time.time() - t0 > DEADLINE - 1:
            break
        improved = False
        for i in range(N):
            for coord in range(3):
                for delta in [step, -step]:
                    cfg[i][coord] += delta
                    new_side = _required_side(cfg)
                    if new_side < current_side - 1e-10:
                        current_side = new_side
                        improved = True
                        break
                    else:
                        cfg[i][coord] -= delta
                if improved:
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
            if step < 1e-8:
                break
    
    return [tuple(c) for c in cfg], current_side


def _initial_guess():
    """Initial guess: hexagonal grid arrangement."""
    # Place hexagons in a compact hexagonal pattern
    # For 12 hexagons, a 4-3-4 or 3-4-3 pattern
    cfg = []
    # Center hexagon
    cfg.append((0.0, 0.0, 0.0))
    # 6 around center
    for i in range(6):
        angle = i * math.pi / 3.0
        x = 2.0 * math.cos(angle)
        y = 2.0 * math.sin(angle)
        cfg.append((x, y, 0.0))
    # 5 more in second ring
    for i in range(5):
        angle = (i + 0.5) * math.pi / 3.0
        x = 4.0 * math.cos(angle)
        y = 4.0 * math.sin(angle)
        cfg.append((x, y, 0.0))
    return cfg


def _hex_grid_4x3():
    """4x3 grid of hexagons."""
    cfg = []
    dx = 3.0  # horizontal spacing for flat-top hexagons
    dy = math.sqrt(3) * 1.5  # vertical spacing
    for row in range(3):
        for col in range(4):
            x = col * dx - 1.5 * dx
            y = row * dy - dy
            if row % 2 == 1:
                x += dx / 2
            cfg.append((x, y, 0.0))
    return cfg


def _hex_grid_3x4():
    """3x4 grid of hexagons."""
    cfg = []
    dx = 3.0
    dy = math.sqrt(3) * 1.5
    for row in range(4):
        for col in range(3):
            x = col * dx - dx
            y = row * dy - 1.5 * dy
            if row % 2 == 1:
                x += dx / 2
            cfg.append((x, y, 0.0))
    return cfg


def _ring_config():
    """One center + 11 in a ring."""
    cfg = [(0.0, 0.0, 0.0)]
    for i in range(11):
        angle = i * 2 * math.pi / 11.0
        x = 3.0 * math.cos(angle)
        y = 3.0 * math.sin(angle)
        cfg.append((x, y, 0.0))
    return cfg


def _compact_config():
    """Tight compact arrangement."""
    cfg = []
    # Two rows of 6, offset
    for i in range(6):
        x = i * 3.0 - 7.5
        cfg.append((x, 0.0, 0.0))
    for i in range(6):
        x = i * 3.0 - 6.0
        cfg.append((x, math.sqrt(3), 0.0))
    return cfg
# EVOLVE-BLOCK-END
