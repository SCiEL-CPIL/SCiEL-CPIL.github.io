import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0

    # Distance squared on triangular lattice
    # Points are a + b*omega, omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # |z|^2 = da^2 + db^2 + da*db
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    # Generate candidates in a box large enough for radius 3 or 4
    # Radius 3 hexagon has 37 points. We need 32.
    # Radius 4 has 61 points.
    candidates = []
    for a in range(-6, 7):
        for b in range(-6, 7):
            if hex_dist([a, b]) <= 4: # Include radius 4 just in case, but we'll sort
                candidates.append([a, b])

    # Sort by hex distance, then by angle to create a spiral
    # This ordering places the most central, well-connected points first.
    candidates.sort(key=lambda p: (hex_dist(p), angle(p)))

    # Take first 32
    # This specific ordering (spiral) is known to attain the Harborth bound.
    result = [p[:] for p in candidates[:32]]
    
    # Verify edge count to be sure
    edges = count_edges(result)
    
    # If for some reason the spiral doesn't hit 76 (unlikely, but possible if sorting is slightly off),
    # we can do a quick local search.
    if edges < 76:
        # Local search improvement
        point_set = set((p[0], p[1]) for p in result)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(result)):
                if time.time() > deadline:
                    break
                ax, ay = result[i][0], result[i][1]
                # Try moving to nearby empty spots
                for da in range(-2, 3):
                    for db in range(-2, 3):
                        nx, ny = ax + da, ay + db
                        if (nx, ny) == (ax, ay) or (nx, ny) in point_set:
                            continue
                        if abs(nx) > 64 or abs(ny) > 64:
                            continue
                        
                        # Calculate delta
                        old_p = [ax, ay]
                        new_p = [nx, ny]
                        edges_lost = 0
                        edges_gained = 0
                        for j in range(len(result)):
                            if j == i: continue
                            if dist_sq(old_p, result[j]) == 1:
                                edges_lost += 1
                            if dist_sq(new_p, result[j]) == 1:
                                edges_gained += 1
                            
                        delta = edges_gained - edges_lost
                        if delta > 0:
                            result[i] = [nx, ny]
                            point_set.remove((ax, ay))
                            point_set.add((nx, ny))
                            edges += delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return result
