"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin before 330s kill
    
    def count_isosceles(points):
        """Count isosceles triangles in a list of (x,y) points."""
        n = len(points)
        if n < 3:
            return 0
        pts = np.array(points, dtype=np.int64)
        total = 0
        for i in range(n):
            d = np.sum((pts - pts[i]) ** 2, axis=1)
            d = np.delete(d, i)
            if len(d) > 1:
                vals, counts = np.unique(d, return_counts=True)
                for c in counts:
                    if c >= 2:
                        total += c * (c - 1) // 2
        return total
    
    def greedy_build():
        """Build a set greedily, adding points that don't create isosceles triangles."""
        points = []
        # Try all cells in random order
        cells = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(cells)
        
        for (x, y) in cells:
            # Check if adding (x,y) creates any isosceles triangle
            ok = True
            for i in range(len(points)):
                xi, yi = points[i]
                di = (x - xi) ** 2 + (y - yi) ** 2
                for j in range(i + 1, len(points)):
                    xj, yj = points[j]
                    dj = (x - xj) ** 2 + (y - yj) ** 2
                    dij = (xi - xj) ** 2 + (yi - yj) ** 2
                    if di == dj or di == dij or dj == dij:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                points.append((x, y))
        return points
    
    best_points = []
    best_count = 0
    
    # Multiple restarts with greedy + local search
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Greedy initialization
        points = greedy_build()
        if len(points) > best_count:
            best_points = list(points)
            best_count = len(points)
        
        # Local search: try swapping points to improve
        if time.time() > deadline - 10:
            break
        
        current = list(points)
        current_viol = count_isosceles(current)
        T = 5.0
        cooling = 0.999
        
        # Local search with simulated annealing
        local_deadline = min(deadline - 5, time.time() + 30)
        while time.time() < local_deadline and T > 0.1:
            # Random swap
            if len(current) > 0 and len(current) < N * N:
                # Remove a random point
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                # Add a random new point
                while True:
                    nx = random.randint(0, N - 1)
                    ny = random.randint(0, N - 1)
                    if (nx, ny) not in current:
                        break
                new_current = [p for i, p in enumerate(current) if i != idx]
                new_current.append((nx, ny))
                new_viol = count_isosceles(new_current)
                delta = new_viol - current_viol
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current = new_current
                    current_viol = new_viol
                    if current_viol == 0 and len(current) > best_count:
                        best_points = list(current)
                        best_count = len(current)
            T *= cooling
        
        # If we have violations, try to fix by removing points
        if count_isosceles(best_points) > 0:
            bp = list(best_points)
            while count_isosceles(bp) > 0 and len(bp) > 0 and time.time() < deadline - 5:
                # Find point in most isosceles triangles
                max_v = -1
                max_i = -1
                for i in range(len(bp)):
                    vi = 0
                    for j in range(len(bp)):
                        if i == j:
                            continue
                        for k in range(j + 1, len(bp)):
                            if i == k:
                                continue
                            p1, p2, p3 = bp[i], bp[j], bp[k]
                            d12 = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                            d13 = (p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2
                            d23 = (p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2
                            if d12 == d13 or d12 == d23 or d13 == d23:
                                vi += 1
                    if vi > max_v:
                        max_v = vi
                        max_i = i
                if max_i >= 0:
                    bp.pop(max_i)
                else:
                    break
            if len(bp) > best_count:
                best_points = bp
                best_count = len(bp)
    
    # Ensure the returned set is valid
    if count_isosceles(best_points) > 0:
        # Fallback: greedy from scratch
        bp = greedy_build()
        if count_isosceles(bp) == 0:
            best_points = bp
        else:
            # Remove points until valid
            while count_isosceles(best_points) > 0 and len(best_points) > 0:
                best_points.pop()
    
    return best_points
# EVOLVE-BLOCK-END
