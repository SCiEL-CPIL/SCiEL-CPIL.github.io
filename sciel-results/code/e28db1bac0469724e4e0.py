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


def _compute_slicing_tree(x, topology):
    """
    Compute centers and sides from parameters x for a given slicing tree topology.
    x contains side lengths for leaves and split ratios for internal nodes.
    topology is a list of operations.
    We use a recursive construction.
    """
    # This is a placeholder for the actual implementation below
    pass

# We will implement a specific set of topologies directly in the optimizer function for efficiency.

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

def _optimize_topology_one_big():
    """
    Topology: One large square in a corner, rest in the L-shape.
    This is harder to parameterize directly with SLSQP without complex constraints.
    We will skip this specific complex topology for now and rely on shelves and balanced trees.
    """
    return None, None, -np.inf

def _optimize_topology_balanced():
    """
    Balanced binary slicing tree.
    This requires more complex parameterization.
    Given time, we will stick to shelf variations which are robust and often near-optimal for small N.
    """
    return None, None, -np.inf

def _generate_shelf_partitions(n=10):
    """Generate diverse shelf partitions."""
    partitions = []
    # 1 shelf
    partitions.append([10])
    # 2 shelves
    partitions.append([5, 5])
    partitions.append([6, 4])
    partitions.append([7, 3])
    # 3 shelves
    partitions.append([4, 3, 3])
    partitions.append([3, 3, 4])
    partitions.append([5, 3, 2])
    partitions.append([2, 3, 5])
    partitions.append([4, 4, 2])
    partitions.append([2, 4, 4])
    # 4 shelves
    partitions.append([3, 3, 2, 2])
    partitions.append([2, 2, 3, 3])
    partitions.append([4, 2, 2, 2])
    # 5 shelves
    partitions.append([2, 2, 2, 2, 2])
    partitions.append([3, 2, 2, 2, 1])
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
    deadline = 630  # seconds, with safety margin
    
    best_centers = None
    best_sides = None
    best_value = -np.inf
    
    # Try all shelf partitions
    partitions = _generate_shelf_partitions(n)
    
    for part in partitions:
        if time.time() - start_time > deadline - 10:
            break
        c, s, val = _optimize_topology_shelf(len(part), part)
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
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
