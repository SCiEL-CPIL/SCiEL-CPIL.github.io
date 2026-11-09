"""Program for packing unit regular hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Approach:
    - Each unit hexagon has circumradius 1 (distance from center to vertex).
    - Two unit hexagons (any orientation) have minimum center-to-center distance of 2*sqrt(3)/3 ≈ 1.1547
      when they are in "flat-top" or "pointy-top" orientation relative to each other? 
      Actually, the minimum distance between centers of two non-overlapping unit regular hexagons
      depends on their relative orientation. The worst case (minimum required distance) is when
      they are aligned such that a flat side of one touches a flat side of the other, giving
      distance = 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732.
      
      Wait, let me reconsider. A regular hexagon with side length 1 has:
      - Apothem (distance from center to midpoint of a side) = sqrt(3)/2 ≈ 0.866
      - Circumradius (distance from center to a vertex) = 1
      
      For two unit hexagons to be disjoint, the minimum center-to-center distance depends on
      their relative rotation. The minimum possible distance between centers for non-overlapping
      hexagons is sqrt(3) (when flat sides face each other). The maximum is 2 (when vertices face each other).
      
      To be safe and ensure no overlap for any orientation, we should use the minimum required
      distance which is sqrt(3) ≈ 1.732. But if we control the orientations, we can potentially
      do better.
      
      Actually, for a simpler and safer approach: if we align all hexagons with the same orientation
      (e.g., flat-top), then the minimum center-to-center distance for non-overlapping hexagons
      is sqrt(3) in the "row" direction and 2 in the "column" direction... no, that's not right either.
      
      Let me think more carefully. For two identical regular hexagons with the same orientation,
      the minimum distance between centers for them to be disjoint is the width of the hexagon
      in the direction of the center-to-center vector. The width of a regular hexagon with side 1
      in any direction is between sqrt(3) (width across flats) and 2 (width across vertices).
      
      The minimum width is sqrt(3), so if we want to guarantee no overlap for any direction of
      separation, we need center distance >= sqrt(3). But if we know the direction, we can use
      a smaller distance.
      
      For simplicity and robustness, let's use the circumradius approach: treat each hexagon as
      a circle of radius 1 (circumradius). Then two circles don't overlap if center distance >= 2.
      This is conservative but safe. The outer hexagon containing all these circles would need
      to contain all circles, so its inradius must be at least 1 + max distance from outer center
      to any inner center.
      
      Actually, the task says each inner hexagon may translate and rotate. So we can choose
      orientations. Let's use a more careful approach:
      
      For two unit regular hexagons, the minimum center-to-center distance for non-overlap is
      sqrt(3) ≈ 1.732 (achieved when flat sides are parallel and facing each other). If we
      orient all hexagons the same way, we can use a hexagonal lattice with spacing sqrt(3)
      in the appropriate directions.
      
      Let's use a hexagonal grid packing with all hexagons oriented the same way. The optimal
      packing of regular hexagons in a hexagonal container is known to be the hexagonal lattice
      arrangement.
      
      For N hexagons, we want to find the smallest regular hexagon that can contain them.
      
      Let's use optimization:
      - Variables: centers (x_i, y_i) for each inner hexagon, outer center (cx, cy), outer side length s
      - Objective: minimize s
      - Constraints: 
        * For each pair (i,j): distance between centers >= sqrt(3) (conservative for same orientation)
        * Each inner hexagon must be fully inside the outer hexagon
      
      For the containment constraint, a point (x,y) is inside a regular hexagon centered at (cx,cy)
      with side length s and rotation angle theta if... this is complex. Let's simplify by
      assuming the outer hexagon is axis-aligned (flat-top) and checking containment.
      
      Actually, let's use the circle packing approximation as suggested:
      - Replace each unit hexagon with its circumcircle of radius 1
      - Approximate the outer hexagon with its circumscribed circle
      - The circumscribed circle of a regular hexagon with side s has radius s
      - So we need all inner circles (radius 1) to be inside the outer circle (radius s)
      - This means: distance from outer center to each inner center + 1 <= s
      - So s >= max_i (distance from outer center to inner center_i) + 1
      
      For the inner circles not to overlap: distance between any two inner centers >= 2
      
      This is a standard circle packing problem. Let's solve it.
    """
    # Use circle packing approximation
    # Inner circles: radius 1, need center distance >= 2
    # Outer circle: radius s, must contain all inner circles
    # s >= max_i ||p_i - c|| + 1
    
    # Initial guess: hexagonal lattice arrangement
    # For N points, arrange in a hexagonal pattern
    
    def hexagonal_lattice_positions(n):
        """Generate positions for n points in a hexagonal lattice pattern."""
        positions = []
        # Start with center, then add rings
        # Ring k has 6k points
        remaining = n
        ring = 0
        while remaining > 0:
            if ring == 0:
                positions.append([0.0, 0.0])
                remaining -= 1
            else:
                # Ring has 6*ring points
                count_in_ring = min(6 * ring, remaining)
                # Place points around the ring
                for i in range(count_in_ring):
                    # Evenly spaced around the ring
                    angle = 2 * math.pi * i / (6 * ring)
                    # Hexagonal ring: distance from center is ring * sqrt(3)
                    # Actually, for a hexagonal lattice, ring k has points at distance k*sqrt(3)
                    # from center, but arranged in a hexagonal pattern
                    # Let's use a simpler approach: place on a circle
                    r = ring * math.sqrt(3)
                    x = r * math.cos(angle)
                    y = r * math.sin(angle)
                    positions.append([x, y])
                remaining -= count_in_ring
            ring += 1
        return positions
    
    # Initial positions
    initial_positions = hexagonal_lattice_positions(N)
    
    # Objective and constraints
    def objective_and_constraints(params):
        """
        params: [x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}, cx, cy]
        Returns: (objective, constraints)
        """
        positions = []
        for i in range(N):
            x = params[2*i]
            y = params[2*i+1]
            positions.append([x, y])
        cx = params[2*N]
        cy = params[2*N+1]
        
        # Objective: s = max_i (distance from (cx,cy) to (x_i,y_i)) + 1
        max_dist = 0.0
        for i in range(N):
            dx = positions[i][0] - cx
            dy = positions[i][1] - cy
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > max_dist:
                max_dist = dist
        s = max_dist + 1.0
        
        # Constraints: for each pair (i,j), distance >= 2
        # We'll use inequality constraints: g_i >= 0
        constraints_list = []
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Constraint: dist - 2 >= 0
                constraints_list.append(dist - 2.0)
        
        return s, constraints_list
    
    # Use scipy minimize with constraints
    # Variables: 2*N + 2 (positions + outer center)
    x0 = []
    for pos in initial_positions:
        x0.append(pos[0])
        x0.append(pos[1])
    # Initial outer center at origin
    x0.append(0.0)
    x0.append(0.0)
    
    # Define constraints for scipy
    def make_constraint(i, j):
        def constraint(params):
            xi = params[2*i]
            yi = params[2*i+1]
            xj = params[2*j]
            yj = params[2*j+1]
            dx = xi - xj
            dy = yi - yj
            dist = math.sqrt(dx*dx + dy*dy)
            return dist - 2.0
        return constraint
    
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': make_constraint(i, j)
            })
    
    # Objective function for scipy
    def objective(params):
        positions = []
        for i in range(N):
            positions.append([params[2*i], params[2*i+1]])
        cx = params[2*N]
        cy = params[2*N+1]
        
        max_dist = 0.0
        for i in range(N):
            dx = positions[i][0] - cx
            dy = positions[i][1] - cy
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > max_dist:
                max_dist = dist
        return max_dist + 1.0
    
    # Run optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-6}
    )
    
    # Extract results
    optimized_params = result.x
    inner_positions = []
    for i in range(N):
        x = optimized_params[2*i]
        y = optimized_params[2*i+1]
        inner_positions.append([x, y, 0.0])  # angle 0 for all
    
    cx = optimized_params[2*N]
    cy = optimized_params[2*N+1]
    
    # Calculate side length
    max_dist = 0.0
    for i in range(N):
        dx = inner_positions[i][0] - cx
        dy = inner_positions[i][1] - cy
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > max_dist:
            max_dist = dist
    s = max_dist + 1.0
    
    # Convert to hexagon format
    # The outer hexagon has side length s, centered at (cx, cy), rotation 0
    # Inner hexagons have side length 1, at positions (x, y) with angle 0
    
    return inner_positions, (cx, cy), s, 0.0
# EVOLVE-BLOCK-END
