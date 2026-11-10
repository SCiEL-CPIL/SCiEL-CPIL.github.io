# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _is_valid(centers, sides, tol=1e-7):
    """Check validity of a packing."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Boundary check
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    # Overlap check
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _solve_lp(order):
    """
    Solve LP for a given topological ordering.
    order is a list of (i, j, dim) where dim=0 means i is left of j, dim=1 means i is below j.
    Returns (centers, sides) or None if infeasible.
    """
    n = 10
    # Variables: cx_0..cx_9, cy_0..cy_9, s_0..s_9
    # Total variables: 30
    # Objective: maximize sum(s_i) -> minimize -sum(s_i)
    c = np.zeros(3 * n)
    for i in range(n):
        c[2 * n + i] = -1.0

    A_ub = []
    b_ub = []

    # Boundary constraints:
    # cx_i - s_i/2 >= 0  ->  -cx_i + s_i/2 <= 0
    # cx_i + s_i/2 <= 1  ->   cx_i - s_i/2 <= 1
    # cy_i - s_i/2 >= 0  ->  -cy_i + s_i/2 <= 0
    # cy_i + s_i/2 <= 1  ->   cy_i - s_i/2 <= 1
    for i in range(n):
        # cx_i - s_i/2 >= 0
        row = np.zeros(3 * n)
        row[i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)

        # cx_i + s_i/2 <= 1
        row = np.zeros(3 * n)
        row[i] = 1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)

        # cy_i - s_i/2 >= 0
        row = np.zeros(3 * n)
        row[n + i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)

        # cy_i + s_i/2 <= 1
        row = np.zeros(3 * n)
        row[n + i] = 1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)

    # Separation constraints based on order
    for i, j, dim in order:
        if dim == 0:
            # i is left of j: cx_i + s_i/2 <= cx_j - s_j/2
            # -> cx_i + s_i/2 - cx_j + s_j/2 <= 0
            row = np.zeros(3 * n)
            row[i] = 1.0
            row[2 * n + i] = 0.5
            row[j] = -1.0
            row[2 * n + j] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)
        else:
            # i is below j: cy_i + s_i/2 <= cy_j - s_j/2
            # -> cy_i + s_i/2 - cy_j + s_j/2 <= 0
            row = np.zeros(3 * n)
            row[n + i] = 1.0
            row[2 * n + i] = 0.5
            row[n + j] = -1.0
            row[2 * n + j] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)

    if not A_ub:
        A_ub = np.array([])
        b_ub = np.array([])
    else:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

    # Bounds: s_i >= 0, cx_i, cy_i >= 0 (implicitly from constraints, but let's add for safety)
    # Actually, cx and cy can be anything, but s must be positive.
    # We add lower bound 0 for s.
    bounds = [(None, None)] * (2 * n) + [(0, None)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.success:
        x = res.x
        cx = x[:n]
        cy = x[n:2 * n]
        s = x[2 * n:]
        centers = np.stack([cx, cy], axis=1)
        return centers, s
    else:
        return None

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, leave margin

    # Initial baseline: grid
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
    best_sides = np.full(n, side, dtype=np.float64)
    best_score = np.sum(best_sides)

    # Generate all possible topological orderings?
    # For n=10, there are C(10,2) = 45 pairs.
    # For each pair, we need to choose one of 3 states: i left of j, i below j, j left of i?
    # Wait, the separation is either horizontal OR vertical.
    # If horizontal: either i left of j OR j left of i.
    # If vertical: either i below j OR j below i.
    # So for each pair (i, j), there are 4 possible constraints:
    # 1. i left of j (cx_i + s_i/2 <= cx_j - s_j/2)
    # 2. j left of i (cx_j + s_j/2 <= cx_i - s_i/2)
    # 3. i below j (cy_i + s_i/2 <= cy_j - s_j/2)
    # 4. j below i (cy_j + s_j/2 <= cy_i - s_i/2)
    #
    # We need to select one constraint for each pair such that the system is consistent.
    # This is equivalent to finding a valid "separation graph" which is a DAG in terms of constraints?
    # Actually, if we pick constraints, we might create cycles (e.g., i left of j, j left of k, k left of i -> impossible).
    # However, the LP will detect infeasibility.
    # The number of combinations is 4^45, which is huge.
    #
    # Alternative: Use a heuristic to find a good set of constraints.
    # Or, use the fact that for optimal packings, the structure is often simple (e.g., grid-like).
    #
    # Let's try a few specific structures:
    # 1. Grid 3x3 + 1
    # 2. Grid 2x5
    # 3. Randomized search over constraint choices?
    #
    # Given the time limit, let's try to solve a few specific plausible topologies.
    #
    # Topology 1: 3x3 grid in top-left, 1 square in bottom-right?
    # Let's define constraints for a 3x3 grid where the 10th square is placed somewhere.
    #
    # Actually, a smarter approach:
    # The problem is small. We can use a local search on the "constraint matrix".
    # Start with a valid configuration (the grid).
    # Determine the active constraints (which pairs are separated horizontally/vertically and in which direction).
    # Then, try to relax some constraints or swap directions to allow larger squares.
    #
    # Let's extract the topology from the baseline grid.
    # In the 3x3 grid (indices 0-8), squares are arranged:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # Square 9 is in (2, 2) position? No, 3x3 has 9 squares.
    # Baseline code:
    # k=3, side=1/3.
    # idx 0: (0,0) -> cx=1/6, cy=1/6
    # idx 1: (1,0) -> cx=1/2, cy=1/6
    # idx 2: (2,0) -> cx=5/6, cy=1/6
    # idx 3: (0,1) -> cx=1/6, cy=1/2
    # ...
    # idx 8: (2,2) -> cx=5/6, cy=5/6
    # idx 9: (0, 3) -> wait, idx=9, cols=9%3=0, rows=9//3=3.
    # So square 9 is at (0, 3), i.e., row 3, col 0.
    # This is outside the 3x3 grid?
    # The unit square is [0,1]^2.
    # Row 3, col 0: cy = (3+0.5)/3 = 3.5/3 = 1.166...
    # This is outside [0,1]!
    # The baseline code is INVALID?
    # Let's check the baseline code again.
    # k = ceil(sqrt(10)) = 4.
    # Ah, k=4.
    # side = 1/4 = 0.25.
    # idx 0: (0,0) -> cx=1/8, cy=1/8
    # ...
    # idx 9: (1,2) -> cols=9%4=1, rows=9//4=2.
    # cx = (1+0.5)*0.25 = 0.375
    # cy = (2+0.5)*0.25 = 0.625
    # This is inside [0,1].
    # So the baseline is a 4x4 grid, using the first 10 cells.
    #
    # Let's try to improve upon this.
    # A known better packing for 10 squares might be:
    # 5 squares of side 0.2 in a row? No, 5*0.2 = 1.0.
    # Two rows of 5 squares of side 0.2?
    # Sum = 10 * 0.2 = 2.0.
    # Baseline sum = 10 * 0.25 = 2.5.
    # Wait, 10 * 0.25 = 2.5.
    # Is 2.5 the max?
    # Area of 10 squares of side 0.25 is 10 * 0.0625 = 0.625.
    # Area of unit square is 1.
    # So area-wise it's fine.
    #
    # Can we do better than 2.5?
    # If we use 9 squares of side 1/3 and 1 smaller?
    # 9 * (1/3) = 3.0.
    # Plus the 10th square.
    # If we pack 9 squares of side 1/3 in a 3x3 grid, they fill the unit square exactly.
    # There is no room for a 10th square.
    # So we must shrink them.
    #
    # Let's try to solve the LP for a specific topology:
    # 3x3 grid of squares 0-8, and square 9 squeezed in somewhere.
    # This requires the 3x3 grid to be slightly smaller than 1/3.
    #
    # Let's try a topology where squares are arranged in two rows of 5.
    # Constraints:
    # Row 0: 0 left of 1, 1 left of 2, 2 left of 3, 3 left of 4
    # Row 1: 5 left of 6, 6 left of 7, 7 left of 8, 8 left of 9
    # Row 0 above Row 1: 0 above 5, 1 above 6, ...
    #
    # Let's define this topology and solve the LP.

    topologies = []

    # Topology 1: 2 rows of 5
    # 0 1 2 3 4
    # 5 6 7 8 9
    order1 = []
    for i in range(5):
        if i < 4:
            order1.append((i, i + 1, 0))  # i left of i+1
            order1.append((i + 5, i + 6, 0))  # i+5 left of i+6
        order1.append((i, i + 5, 1))  # i above i+5 (i is below i+5? No, i is row 0, i+5 is row 1. So i is above i+5.
                                      # My convention: dim=1 means i below j.
                                      # So (i, i+5, 1) means i below i+5. That's wrong.
                                      # I want i above i+5, so i+5 below i.
                                      # So (i+5, i, 1) means i+5 below i.
    # Let's fix the convention in _solve_lp.
    # dim=1: i below j.
    # So for row 0 (top) and row 1 (bottom):
    # Squares in row 0 are above squares in row 1.
    # So square j in row 1 is below square i in row 0.
    # Constraint: (j, i, 1).

    # Let's restart topology definition.
    # Topology 1: 2x5 grid
    order1 = []
    # Horizontal
    for r in range(2):
        for c in range(4):
            i = r * 5 + c
            j = r * 5 + c + 1
            order1.append((i, j, 0))  # i left of j
    # Vertical
    for c in range(5):
        i = c  # top row
        j = c + 5  # bottom row
        order1.append((j, i, 1))  # j below i

    topologies.append(order1)

    # Topology 2: 5x2 grid (5 rows, 2 cols)
    order2 = []
    for r in range(5):
        i = r * 2
        j = r * 2 + 1
        order2.append((i, j, 0))
    for c in range(2):
        for r in range(4):
            i = r * 2 + c
            j = (r + 1) * 2 + c
            order2.append((j, i, 1))  # j below i

    topologies.append(order2)

    # Topology 3: 3x3 grid + 1
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # 9 is placed in a corner, say bottom-right, but overlapping 8?
    # No, must be disjoint.
    # Let's say 9 is to the right of 2, 5, 8?
    # But then it's a 3x4 grid?
    # Let's try 3 rows:
    # Row 0: 0, 1, 2
    # Row 1: 3, 4, 5
    # Row 2: 6, 7, 8, 9 (4 squares)
    order3 = []
    # Row 0
    order3.append((0, 1, 0))
    order3.append((1, 2, 0))
    # Row 1
    order3.append((3, 4, 0))
    order3.append((4, 5, 0))
    # Row 2
    order3.append((6, 7, 0))
    order3.append((7, 8, 0))
    order3.append((8, 9, 0))
    # Vertical
    for c in range(3):
        order3.append((3 + c, c, 1))  # 3+c below c
        order3.append((6 + c, 3 + c, 1))  # 6+c below 3+c
    # 9 is in row 2, col 3.
    # It should be below 5? (5 is row 1, col 2).
    # And to the right of 8.
    # Is 9 below 5?
    # If 5 is at col 2, and 9 is at col 3, they don't overlap vertically if 9 is to the right.
    # But we need a vertical constraint if they are in the same column? No, different columns.
    # So 9 is only constrained horizontally by 8.
    # But we need to ensure 9 doesn't overlap with 5 vertically?
    # If 9 is to the right of 8, and 8 is to the right of 7, etc.
    # And 5 is above 8.
    # So 5 is at col 2, row 1. 9 is at col 3, row 2.
    # They are diagonally separated. No direct constraint needed if horizontal/vertical separation is sufficient.
    # But wait, if 9 is wide, it might overlap with 5 horizontally?
    # No, 9 is to the right of 8, and 8 is to the right of 7, and 7 is below 4, and 4 is to the right of 3, and 3 is below 0...
    # Actually, 5 is at col 2. 9 is at col 3.
    # So 9 is to the right of 5?
    # We don't have a constraint (5, 9, 0).
    # If 9 is very wide, it could extend left and overlap with 5.
    # So we should add (5, 9, 0) to ensure 5 is left of 9.
    order3.append((5, 9, 0))

    topologies.append(order3)

    # Try solving these topologies
    for order in topologies:
        if time.time() > deadline - 10:
            break
        res = _solve_lp(order)
        if res is not None:
            centers, sides = res
            if _is_valid(centers, sides):
                score = np.sum(sides)
                if score > best_score:
                    best_score = score
                    best_centers = centers
                    best_sides = sides

    # If we have time, try a local search or random topology search
    # For now, return the best found
    return best_centers, best_sides

# EVOLVE-BLOCK-END
