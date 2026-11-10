# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize


def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _score(centers, sides, tol=1e-7):
    """Score = sum of sides. Higher is better."""
    if not _is_valid(centers, sides, tol):
        return -np.inf
    return np.sum(sides)


def _optimize_topology_shelf(m_shelves, squares_per_shelf):
    """
    Optimize a shelf-based packing.
    m_shelves: number of shelves.
    squares_per_shelf: list of number of squares in each shelf.
    """
    n_squares = sum(squares_per_shelf)
    if n_squares != 10:
        return None, None, -np.inf
        
    # Variables: heights of shelves h_0 ... h_{k-1}
    # We want to maximize sum of sides.
    # For shelf k with m_k squares, max side is min(h_k, 1/m_k).
    # Sum for shelf k is m_k * min(h_k, 1/m_k).
    # Constraint: sum(h_k) <= 1.
    
    def neg_obj(h):
        # h is array of heights
        total_side = 0.0
        for i, m in enumerate(squares_per_shelf):
            h_k = h[i]
            if h_k < 0: return 1e6 # penalty
            max_s = min(h_k, 1.0/m)
            total_side += m * max_s
        return -total_side

    def constraints(h):
        return 1.0 - np.sum(h)

    x0 = np.array([1.0/len(squares_per_shelf)] * len(squares_per_shelf))
    bounds = [(0, 1.0)] * len(squares_per_shelf)
    cons = {'type': 'ineq', 'fun': constraints}
    
    res = minimize(neg_obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    
    if not res.success:
        return None, None, -np.inf
        
    h_opt = res.x
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    y_offset = 0.0
    idx = 0
    for i, m in enumerate(squares_per_shelf):
        h_k = h_opt[i]
        s_k = min(h_k, 1.0/m)
        # Place squares in this shelf
        # They are placed side by side. Total width used is m * s_k.
        # Center them in the shelf? Or left align?
        # To maximize validity, centering is safe.
        x_offset = (1.0 - m * s_k) / 2.0
        for j in range(m):
            cx = x_offset + j * s_k + s_k / 2.0
            cy = y_offset + h_k / 2.0
            centers[idx, 0] = cx
            centers[idx, 1] = cy
            sides[idx] = s_k
            idx += 1
        y_offset += h_k
        
    return centers, sides, np.sum(sides)

def _optimize_one_large_remaining():
    """
    Place one large square in a corner, then pack remaining 9 in the L-shaped region.
    The L-shape consists of a top strip and a right strip.
    We optimize the large square side length, then use shelves for the remaining.
    """
    n_remaining = 9
    best_val = -np.inf
    best_c = None
    best_s = None
    
    # Try different sizes for the large square
    for s_large in np.linspace(0.3, 0.5, 20):
        # Large square in bottom-left corner
        cx_large = s_large / 2.0
        cy_large = s_large / 2.0
        
        # Remaining region: 
        # Top strip: [0,1] x [s_large, 1] with height (1 - s_large)
        # Right strip: [s_large, 1] x [0, s_large] with width (1 - s_large)
        
        # Pack remaining 9 squares using shelves in these two regions
        # Split: some in top strip, some in right strip
        for n_top in range(0, n_remaining + 1):
            n_right = n_remaining - n_top
            if n_top == 0 or n_right == 0:
                continue
            
            # Top strip: height h_top = 1 - s_large, width 1
            # Right strip: width w_right = 1 - s_large, height s_large
            
            # For top strip with n_top squares:
            # If we use a single shelf, max side = min(h_top, 1/n_top)
            # For right strip with n_right squares:
            # We need to orient differently. Use vertical shelves.
            # Max side = min(w_right, s_large / n_right)
            
            h_top = 1.0 - s_large
            w_right = 1.0 - s_large
            h_right = s_large
            
            s_top = min(h_top, 1.0 / n_top) if n_top > 0 else 0
            s_right = min(w_right, h_right / n_right) if n_right > 0 else 0
            
            total = s_large + n_top * s_top + n_right * s_right
            
            # Check if this is better
            if total > best_val:
                # Build the packing
                centers = np.zeros((10, 2))
                sides = np.zeros(10)
                
                idx = 0
                # Large square
                centers[idx, 0] = cx_large
                centers[idx, 1] = cy_large
                sides[idx] = s_large
                idx += 1
                
                # Top strip squares
                if n_top > 0:
                    x_off = (1.0 - n_top * s_top) / 2.0
                    for j in range(n_top):
                        centers[idx, 0] = x_off + j * s_top + s_top / 2.0
                        centers[idx, 1] = s_large + h_top / 2.0
                        sides[idx] = s_top
                        idx += 1
                
                # Right strip squares (stacked vertically)
                if n_right > 0:
                    y_off = (s_large - n_right * s_right) / 2.0
                    for j in range(n_right):
                        centers[idx, 0] = s_large + w_right / 2.0
                        centers[idx, 1] = y_off + j * s_right + s_right / 2.0
                        sides[idx] = s_right
                        idx += 1
                
                if _is_valid(centers, sides, 1e-7):
                    best_val = total
                    best_c = centers.copy()
                    best_s = sides.copy()
    
    if best_c is not None:
        return best_c, best_s, best_val
    return None, None, -np.inf

def _polish_local(centers, sides, tol=1e-7, max_iter=10000, step_size=0.01):
    """
    Local search to improve the packing by small perturbations.
    """
    n = len(sides)
    best_val = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    for it in range(max_iter):
        # Try small random perturbations
        improved = False
        for _ in range(20):
            c_new = best_c.copy()
            s_new = best_s.copy()
            
            # Perturb positions
            mask = np.random.random(n) < 0.5
            c_new[mask, 0] += np.random.uniform(-step_size, step_size, mask.sum())
            c_new[mask, 1] += np.random.uniform(-step_size, step_size, mask.sum())
            
            # Clamp to bounds
            for i in range(n):
                s_i = s_new[i]
                c_new[i, 0] = np.clip(c_new[i, 0], s_i/2, 1 - s_i/2)
                c_new[i, 1] = np.clip(c_new[i, 1], s_i/2, 1 - s_i/2)
            
            if _is_valid(c_new, s_new, tol):
                val = np.sum(s_new)
                if val > best_val + 1e-10:
                    best_val = val
                    best_c = c_new
                    best_s = s_new
                    improved = True
        
        if not improved:
            step_size *= 0.95
            if step_size < 1e-5:
                break
    
    return best_c, best_s, best_val

def _generate_shelf_partitions(n=10):
    """Generate diverse shelf partitions."""
    partitions = []
    # 1 shelf
    partitions.append([10])
    # 2 shelves
    partitions.append([5, 5])
    partitions.append([6, 4])
    partitions.append([7, 3])
    partitions.append([8, 2])
    partitions.append([9, 1])
    # 3 shelves
    partitions.append([4, 3, 3])
    partitions.append([3, 3, 4])
    partitions.append([5, 3, 2])
    partitions.append([2, 3, 5])
    partitions.append([4, 4, 2])
    partitions.append([2, 4, 4])
    partitions.append([6, 2, 2])
    partitions.append([2, 6, 2])
    partitions.append([3, 5, 2])
    partitions.append([5, 2, 3])
    # 4 shelves
    partitions.append([3, 3, 2, 2])
    partitions.append([2, 2, 3, 3])
    partitions.append([4, 2, 2, 2])
    partitions.append([2, 4, 2, 2])
    partitions.append([5, 2, 2, 1])
    partitions.append([2, 2, 5, 1])
    # 5 shelves
    partitions.append([2, 2, 2, 2, 2])
    partitions.append([3, 2, 2, 2, 1])
    partitions.append([2, 3, 2, 2, 1])
    # 6+ shelves
    partitions.append([2, 2, 2, 2, 1, 1])
    partitions.append([2, 2, 2, 1, 2, 1])
    return partitions

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 600  # seconds, with safety margin
    
    best_centers = None
    best_sides = None
    best_value = -np.inf
    
    # Try all shelf partitions
    partitions = _generate_shelf_partitions(n)
    
    for part in partitions:
        if time.time() - start_time > deadline - 30:
            break
        c, s, val = _optimize_topology_shelf(len(part), part)
        if c is not None and _is_valid(c, s, tol):
            if val > best_value:
                best_value = val
                best_centers = c.copy()
                best_sides = s.copy()
    
    # Try one-large-remaining topology
    if time.time() - start_time < deadline - 60:
        c, s, val = _optimize_one_large_remaining()
        if c is not None and _is_valid(c, s, tol):
            if val > best_value:
                best_value = val
                best_centers = c.copy()
                best_sides = s.copy()
    
    # Fallback: grid if no valid solution found
    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
        best_value = np.sum(best_sides)
    
    # Polish with local search if time remains
    if time.time() - start_time < deadline - 10:
        c_pol, s_pol, val_pol = _polish_local(best_centers, best_sides, tol, max_iter=5000, step_size=0.005)
        if val_pol > best_value:
            best_centers = c_pol
            best_sides = s_pol
            best_value = val_pol
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
