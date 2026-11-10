import time
import math
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    # Helper to count unit distance pairs
    def count_edges(p_set):
        edges = 0
        for a, b in p_set:
            if (a + 1, b) in p_set:
                edges += 1
            if (a, b + 1) in p_set:
                edges += 1
            if (a + 1, b - 1) in p_set:
                edges += 1
        return edges

    # Generate hexagonal spiral points
    def hex_spiral(n):
        # Generate points in order of increasing distance from center
        # Using axial coordinates where distance is max(|a|, |b|, |a+b|)
        points = [(0, 0)]
        if n == 1:
            return points

        # Generate rings
        for r in range(1, 20):
            # Ring r consists of points where max(|a|, |b|, |a+b|) = r
            # We generate them in a specific order for the spiral
            # Start at (r, 0) and go around
            # The 6 sides of the hexagon ring r:
            # Side 1: (r, 0) to (r, -r) -> a=r, b from 0 to -r
            # Side 2: (r, -r) to (0, -r) -> a from r to 0, b=-r
            # Side 3: (0, -r) to (-r, 0) -> a from 0 to -r, b from -r to 0
            # Side 4: (-r, 0) to (-r, r) -> a=-r, b from 0 to r
            # Side 5: (-r, r) to (0, r) -> a from -r to 0, b=r
            # Side 6: (0, r) to (r, 0) -> a from 0 to r, b from r to 0

            # Actually, a simpler way to generate hex spiral:
            # For each ring r, the points are:
            # (r, 0), (r, -1), ..., (r, -r)
            # (r-1, -r), (r-2, -r), ..., (0, -r)
            # (-1, -r+1), (-2, -r+2), ..., (-r, 0)
            # (-r, 1), (-r, 2), ..., (-r, r)
            # (-r+1, r), (-r+2, r), ..., (0, r)
            # (1, r-1), (2, r-2), ..., (r, 0) -- but (r,0) is already counted

            for b in range(0, -r - 1, -1):
                points.append((r, b))
            for a in range(r - 1, -1, -1):
                points.append((a, -r))
            for a in range(-1, -r - 1, -1):
                points.append((a, -r + a + r))  # b = -r - a
            for b in range(1, r + 1):
                points.append((-r, b))
            for a in range(-r + 1, 1):
                points.append((a, r))
            for a in range(1, r):
                points.append((a, r - a))

            if len(points) >= n:
                break

        return points[:n]

    # Initial guess: Hexagonal spiral
    points = hex_spiral(32)
    p_set = set(points)
    current_edges = count_edges(p_set)

    # Target is 76
    if current_edges < 76:
        # Local search: try swapping a point with a nearby non-selected point
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_points = list(points)
            p_set = set(new_points)
            for i in range(len(new_points)):
                a, b = new_points[i]
                # Try moving this point to a neighbor
                for da, db in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
                               (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2),
                               (1, 1), (-1, -1), (2, 1), (-2, -1), (1, 2), (-1, -2)]:
                    na, nb = a + da, b + db
                    if (na, nb) in p_set:
                        continue
                    # Swap
                    new_points[i] = (na, nb)
                    p_set.remove((a, b))
                    p_set.add((na, nb))
                    new_edges = count_edges(p_set)
                    if new_edges > current_edges:
                        current_edges = new_edges
                        points = new_points
                        p_set = set(points)
                        improved = True
                        break
                    else:
                        # Revert
                        new_points[i] = (a, b)
                        p_set.remove((na, nb))
                        p_set.add((a, b))
                if improved:
                    break

    # Convert to list of lists as required
    return [[a, b] for a, b in points]
