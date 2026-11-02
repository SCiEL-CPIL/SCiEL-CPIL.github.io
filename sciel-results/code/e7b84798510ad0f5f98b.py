"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Physics-based discrete-event simulation (billiard balls) with radius growth.
Instead of gradient ascent, we simulate circles as billiard balls that collide
elastically, bounce off walls, and gradually grow. The stochastic nature of
initial velocities and the momentum-based collision resolution allows escaping
local minima that trap gradient methods. We use multiple restarts with different
initial conditions and a final Delaunay-based polish.
"""
import time
import numpy as np
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9
DEADLINE = 140.0  # Safety margin for 150s limit


def _elapsed():
    return time.time() - _t_start


def _remaining():
    return DEADLINE - _elapsed()


def _max_radius(pos, r):
    """Compute maximal feasible radius for each circle given positions."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                dmin = min(dmin, d - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _validate(pos, rad):
    """Check if all circles are valid (in square, non-overlapping)."""
    for i in range(len(rad)):
        if rad[i] < 0:
            return False
        x, y = pos[i]
        if x < rad[i] + 1e-8 or x > 1 - rad[i] - 1e-8:
            return False
        if y < rad[i] + 1e-8 or y > 1 - rad[i] - 1e-8:
            return False
    for i in range(len(rad)):
        for j in range(i + 1, len(rad)):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < rad[i] + rad[j] - 1e-8:
                return False
    return True


def _score(pos, rad):
    """Score: sum of radii if valid, else -inf."""
    if not _validate(pos, rad):
        return -np.inf
    return float(np.sum(rad))


def _initial_grid(side):
    """Generate grid initialization."""
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side))
    cells = cells[:N]
    pos = np.array(cells, dtype=float)
    r = np.full(N, 0.5 / side - 0.01)
    for _ in range(20):
        r = _max_radius(pos, r)
    return pos, r


def _initial_hex():
    """Generate hexagonal initialization."""
    side = int(np.ceil(np.sqrt(N)))
    pos_hex = []
    for i in range(side):
        for j in range(side):
            if len(pos_hex) >= N:
                break
            x = (j + 0.5 * (i % 2)) / side
            y = (i + 0.5) / side
            pos_hex.append((x, y))
    pos_hex = np.array(pos_hex[:N], dtype=float)
    pos_hex = np.clip(pos_hex, 0.05, 0.95)
    r_hex = np.full(N, 0.03)
    for _ in range(30):
        r_hex = _max_radius(pos_hex, r_hex)
    return pos_hex, r_hex


def _initial_random(rng):
    """Generate random initialization."""
    pos = rng.random((N, 2)) * 0.8 + 0.1
    r = np.full(N, 0.02)
    for _ in range(20):
        r = _max_radius(pos, r)
    return pos, r


def _physics_simulate(pos0, r0, dt=0.005, n_steps=5000, rng=None, growth_rate=0.0002):
    """
    Physics-based simulation:
    - Circles have positions, velocities, and radii
    - Each step: grow radii slightly, resolve collisions (elastic), bounce off walls
    - This is a discrete-event style simulation with fixed timestep
    """
    if rng is None:
        rng = np.random.default_rng()
    
    pos = pos0.copy().astype(float)
    rad = r0.copy().astype(float)
    vel = rng.random((N, 2)) * 0.1  # Small random initial velocities
    
    best_pos = pos.copy()
    best_rad = rad.copy()
    best_score = _score(pos, rad)
    
    for step in range(n_steps):
        if _remaining() < 10:
            break
        
        # Grow radii
        rad += growth_rate
        # Cap radii to stay in bounds
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if rad[i] > max_r:
                rad[i] = max_r
        
        # Update positions
        pos += vel * dt
        
        # Wall collisions
        for i in range(N):
            if pos[i, 0] < rad[i]:
                pos[i, 0] = rad[i]
                vel[i, 0] = abs(vel[i, 0])
            elif pos[i, 0] > 1 - rad[i]:
                pos[i, 0] = 1 - rad[i]
                vel[i, 0] = -abs(vel[i, 0])
            if pos[i, 1] < rad[i]:
                pos[i, 1] = rad[i]
                vel[i, 1] = abs(vel[i, 1])
            elif pos[i, 1] > 1 - rad[i]:
                pos[i, 1] = 1 - rad[i]
                vel[i, 1] = -abs(vel[i, 1])
        
        # Resolve collisions (iterative, a few passes)
        for _pass in range(5):
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    d = np.hypot(dx, dy)
                    min_d = rad[i] + rad[j]
                    if d < min_d and d > 1e-12:
                        # Push apart
                        overlap = min_d - d
                        ux = dx / d
                        uy = dy / d
                        # Equal mass assumption
                        pos[i, 0] -= 0.5 * overlap * ux
                        pos[i, 1] -= 0.5 * overlap * uy
                        pos[j, 0] += 0.5 * overlap * ux
                        pos[j, 1] += 0.5 * overlap * uy
                        # Elastic collision: swap velocity components along normal
                        v_rel = (vel[j, 0] - vel[i, 0]) * ux + (vel[j, 1] - vel[i, 1]) * uy
                        if v_rel < 0:  # Only if approaching
                            vel[i, 0] += v_rel * ux
                            vel[i, 1] += v_rel * uy
                            vel[j, 0] -= v_rel * ux
                            vel[j, 1] -= v_rel * uy
        
        # Clamp to square
        pos = np.clip(pos, 0.0, 1.0)
        
        # Evaluate score periodically
        if step % 50 == 0:
            # Try to maximize radii at current positions
            rad_check = _max_radius(pos, rad)
            s = _score(pos, rad_check)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
                best_rad = rad_check.copy()
    
    # Final: maximize radii
    rad_final = _max_radius(best_pos, best_rad)
    s_final = _score(best_pos, rad_final)
    if s_final > best_score:
        best_pos = best_pos
        best_rad = rad_final
        best_score = s_final
    
    return best_pos, best_rad, best_score


def _delaunay_polish(pos, rad, max_iters=100):
    """
    Polish using Delaunay-based local optimization.
    For each circle, try to move it slightly to increase its max radius.
    """
    pos = pos.copy().astype(float)
    rad = rad.copy().astype(float)
    
    for it in range(max_iters):
        if _remaining() < 10:
            break
        
        improved = False
        for i in range(N):
            # Compute current max radius
            mr_i = _max_radius(pos, rad)[i]
            
            # Try small perturbations
            best_pert = None
            best_mr = mr_i
            
            # Check boundaries and neighbors
            for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = pos.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    # Clamp
                    new_pos[i, 0] = np.clip(new_pos[i, 0], 0, 1)
                    new_pos[i, 1] = np.clip(new_pos[i, 1], 0, 1)
                    
                    # Check if still valid
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        d = np.hypot(new_pos[i, 0] - new_pos[j, 0], new_pos[i, 1] - new_pos[j, 1])
                        if d < rad[i] + rad[j] - 1e-8:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Compute new max radius for i
                    dmin = min(new_pos[i, 0], 1.0 - new_pos[i, 0], new_pos[i, 1], 1.0 - new_pos[i, 1])
                    for j in range(N):
                        if i == j:
                            continue
                        d = np.hypot(new_pos[i, 0] - new_pos[j, 0], new_pos[i, 1] - new_pos[j, 1])
                        dmin = min(dmin, d - rad[j])
                    new_mr = max(0.0, dmin)
                    
                    if new_mr > best_mr + 1e-10:
                        best_mr = new_mr
                        best_pert = (dx, dy)
            
            if best_pert is not None:
                pos[i, 0] += best_pert[0]
                pos[i, 1] += best_pert[1]
                improved = True
        
        if not improved:
            break
    
    # Final radius maximization
    rad = _max_radius(pos, rad)
    return pos, rad


def construct():
    global _t_start
    _t_start = time.time()
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_rad = None
    best_score = -np.inf
    
    def consider(pos, rad):
        nonlocal best_pos, best_rad, best_score
        rad_check = _max_radius(pos, rad)
        s = _score(pos, rad_check)
        if s > best_score:
            best_score = s
            best_pos = pos.copy()
            best_rad = rad_check.copy()
    
    # Generate diverse initializations
    inits = []
    
    # Grids
    for side in [5, 6, 7]:
        if _remaining() < 30:
            break
        pos, r = _initial_grid(side)
        inits.append((pos, r))
    
    # Hex
    if _remaining() < 30:
        pass
    else:
        pos, r = _initial_hex()
        inits.append((pos, r))
    
    # Randoms
    for k in range(8):
        if _remaining() < 30:
            break
        pos, r = _initial_random(rng)
        inits.append((pos, r))
    
    # Run physics simulation on each initialization
    for idx, (pos0, r0) in enumerate(inits):
        if _remaining() < 20:
            break
        
        try:
            pos_sim, rad_sim, score_sim = _physics_simulate(
                pos0, r0, 
                dt=0.005, 
                n_steps=3000, 
                rng=rng,
                growth_rate=0.0003
            )
            consider(pos_sim, rad_sim)
        except Exception:
            pass
    
    # Polish the best solution
    if best_pos is not None:
        if _remaining() > 15:
            pos_pol, rad_pol = _delaunay_polish(best_pos, best_rad, max_iters=80)
            consider(pos_pol, rad_pol)
    
    # Final polish: maximize radii
    if best_pos is not None:
        rad_final = _max_radius(best_pos, best_rad)
        consider(best_pos, rad_final)
    
    if best_pos is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.column_stack([best_pos, best_rad])


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
