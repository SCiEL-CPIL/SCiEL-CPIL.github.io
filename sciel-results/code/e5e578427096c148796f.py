import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Find smallest k such that 3*k*(k-1) + 1 >= N
    k = 1
    while 3 * k * (k - 1) + 1 < N:
        k += 1
    
    # Generate hexagonal grid points (axial coordinates)
    # Condition: |q| < k, |r| < k, |q+r| < k
    points = []
    for q in range(-k + 1, k):
        for r in range(-k + 1, k):
            if abs(q + r) < k:
                points.append((q, r))
    
    # We need exactly N points. The loop generates 3*k*(k-1)+1 points.
    # We take the first N.
    # To make it symmetric and compact, we might want to sort them by distance from center?
    # Or just take the first N. The first N in the loop order (top-left to bottom-right)
    # might be biased. Let's sort by distance to center to keep the cluster compact.
    
    def dist2(q, r):
        # Cartesian distance squared for pointy-top
        # x = sqrt(3) * (q + r/2), y = 1.5 * r
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x * x + y * y

    points.sort(key=lambda p: dist2(p[0], p[1]))
    selected_points = points[:N]
    
    # Convert to Cartesian coordinates
    inner = []
    for q, r in selected_points:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Side length k
    # Center (0,0)
    # Rotation 0
    return inner, (0.0, 0.0), float(k), 0.0
# EVOLVE-BLOCK-END
