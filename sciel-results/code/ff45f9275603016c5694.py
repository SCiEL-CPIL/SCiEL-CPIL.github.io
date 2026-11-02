"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Simulated Annealing with adaptive radii and multi-start.
"""
import time
import numpy as np
import random

N = 26
EPS = 1e-9
DEADLINE = 145.0  # Safety margin from 150s limit


def _max_radius(pos, r, iters=20):
    """Maximize radii given fixed positions via Jacobi iterations."""
    n = len(r)
    for _ in range(iters):
        for i in range(n):
            dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    dmin = min(dmin, d - r[j])
            r[i] = max(0.0, dmin)
    return r


def _validity(circles):
    """Check if circles are valid (inside square, non-overlapping). Returns (is_valid, sum_radii)."""
    n = len(circles)
    s = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False, 0.0
        if x < r + EPS or x > 1 - r - EPS or y < r + EPS or y > 1 - r - EPS:
            return False, 0.0
        s += r
        for j in range(i + 1, n):
            x2, y2, r2 = circles[j]
            d2 = (x - x2) ** 2 + (y - y2) ** 2
            if d2 < (r + r2) ** 2 - 1e-9:
                return False, 0.0
    return True, s


def construct():
    t_start = time.time()
    best_score = -1.0
    best_circles = None
    
    # Helper to update best
    def update_best(circles):
        nonlocal best_score, best_circles
        valid, s = _validity(circles)
        if valid and s > best_score:
            best_score = s
            best_circles = [tuple(c) for c in circles]

    # Generate diverse initial configurations
    def get_initializations():
        inits = []
        # 1. Grid
        for side in [5, 6, 7]:
            cells = []
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.9))
                    idx += 1
            if len(cells) >= N:
                inits.append(cells[:N])
        
        # 2. Hexagonal packing approximation
        # Try to fit as many as possible in a hex grid
        hex_inits = []
        for offset in [0.0, 0.1, 0.2]:
            pts = []
            y = 0.5 / 5
            row = 0
            while len(pts) < N and y < 1.0:
                x = 0.5 / 5 + (row % 2) * (0.5 / 5) * 0.866
                while x < 1.0 and len(pts) < N:
                    pts.append((x + offset, y + offset, 0.5 / 5 * 0.9))
                    x += 0.5 / 5
                y += 0.5 / 5 * 0.866
                row += 1
            if len(pts) >= N:
                hex_inits.append(pts[:N])
        inits.extend(hex_inits)
        
        # 3. Random
        rng = np.random.default_rng(42)
        for _ in range(5):
            pts = []
            for _ in range(N):
                x = rng.random()
                y = rng.random()
                r = 0.02 + 0.05 * rng.random()
                pts.append((x, y, r))
            inits.append(pts)
            
        return inits

    inits = get_initializations()
    
    # Simulated Annealing parameters
    T_start = 0.1
    T_end = 0.001
    alpha = 0.995  # Cooling factor per step
    
    for init in inits:
        if time.time() > DEADLINE:
            break
            
        # Initialize positions and radii
        pos = np.array([(p[0], p[1]) for p in init], dtype=float)
        r = np.array([p[2] for p in init], dtype=float)
        
        # Maximize radii initially
        r = _max_radius(pos, r, iters=50)
        circles = list(zip(pos[:, 0], pos[:, 1], r))
        update_best(circles)
        
        current_score = best_score if best_circles and all(a==b for a,b in zip(best_circles, circles)) else _validity(circles)[1]
        if not _validity(circles)[0]:
             # If initial is invalid, try to fix or skip? 
             # _max_radius should make it valid if positions allow.
             # If not, it means positions are too close.
             # Let's force validity by shrinking radii drastically if needed?
             # Actually _max_radius sets r to max possible. If dmin < 0, r becomes 0.
             # So it is always valid (r>=0).
             pass
             
        T = T_start
        rng = random.Random(123 + len(inits))
        
        # Annealing loop
        steps_per_init = 2000
        for step in range(steps_per_init):
            if time.time() > DEADLINE:
                break
                
            # Random move: pick a circle and jitter its position
            i = rng.randint(0, N - 1)
            dx = rng.gauss(0, 0.02)
            dy = rng.gauss(0, 0.02)
            
            old_x, old_y = pos[i, 0], pos[i, 1]
            new_x = np.clip(old_x + dx, 0.0, 1.0)
            new_y = np.clip(old_y + dy, 0.0, 1.0)
            
            # If no change, skip
            if new_x == old_x and new_y == old_y:
                continue
                
            pos[i, 0] = new_x
            pos[i, 1] = new_y
            
            # Recalculate radii (fast)
            new_r = _max_radius(pos, r, iters=5)
            new_circles = list(zip(pos[:, 0], pos[:, 1], new_r))
            
            valid, new_score = _validity(new_circles)
            if not valid:
                # Should not happen with _max_radius, but safety check
                pos[i, 0] = old_x
                pos[i, 1] = old_y
                continue
            
            # Acceptance criterion
            if new_score > current_score:
                # Always accept improvement
                current_score = new_score
                r = new_r
                update_best(new_circles)
            else:
                # Accept with probability exp(-(delta)/T)
                delta = current_score - new_score
                if delta < 0: # Should be covered above, but safe
                     accept = True
                else:
                     # If T is very small, reject
                     if T < 1e-6:
                         accept = False
                     else:
                         # Use log to avoid underflow
                         # log(prob) = -delta / T
                         # If -delta/T < log(random), accept
                         # random in (0,1), log in (-inf, 0)
                         # We want P(accept) = exp(-delta/T)
                         # Accept if log(U) < -delta/T  => U < exp(-delta/T)
                         # Since U is uniform [0,1], log(U) is uniform-ish in (-inf, 0) but biased.
                         # Better: accept if rng.random() < exp(-delta/T)
                         # But exp can underflow.
                         # If delta/T > 700, exp is 0.
                         if delta / T > 700:
                             accept = False
                         else:
                             accept = rng.random() < np.exp(-delta / T)
                
                if accept:
                    current_score = new_score
                    r = new_r
                    # Don't update best if score is worse, but keep state
                else:
                    # Reject move
                    pos[i, 0] = old_x
                    pos[i, 1] = old_y
                    # r remains old r? 
                    # Wait, r was updated in place? No, new_r was calculated.
                    # We need to restore r if we reject.
                    # But r is the array used for next _max_radius.
                    # If we reject, we should revert pos, and r should be consistent with old pos.
                    # Since we didn't update r in the 'accept' branch yet, r is still old.
                    # But we calculated new_r based on new pos.
                    # So if we reject, pos is old, r is old. Correct.
                    pass
            
            # Cool down
            T *= alpha
            
            # Occasionally re-maximize radii more thoroughly to escape local radius optima
            if step % 100 == 0:
                 r = _max_radius(pos, r, iters=20)
                 circles = list(zip(pos[:, 0], pos[:, 1], r))
                 valid, s = _validity(circles)
                 if valid:
                     if s > current_score:
                         current_score = s
                     update_best(circles)

    if best_circles is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.array(best_circles)


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
