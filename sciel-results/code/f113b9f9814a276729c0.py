import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + da*db + db*db
    
    def count_edges(points):
        n = len(points)
        edges = 0
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    edges += 1
        return edges

    # Theoretical optimum for n=32 is 76.
    # We construct a hexagonal cluster.
    # Pattern: rows of lengths 4, 6, 7, 7, 6, 2
    # We need to align them correctly.
    
    configs = []
    
    # Config 1: The "hexagon" approximation
    # Rows: 4, 6, 7, 7, 6, 2
    # Let's try centering them.
    # If we place row i at y=i.
    # x offsets:
    # Let's try to make it symmetric around x=2.5?
    # y=0: 1, 2, 3, 4
    # y=1: 0, 1, 2, 3, 4, 5
    # y=2: 0, 1, 2, 3, 4, 5, 6
    # y=3: 0, 1, 2, 3, 4, 5, 6
    # y=4: 0, 1, 2, 3, 4, 5
    # y=5: 2, 3
    # Let's check edges for this.
    
    p1 = [
        [1, 0], [2, 0], [3, 0], [4, 0],
        [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1],
        [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2],
        [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3],
        [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4],
        [2, 5], [3, 5]
    ]
    if len(p1) == 32 and len(set(map(tuple, p1))) == 32:
        configs.append(p1)
        
    # Config 2: Slightly different alignment
    p2 = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [-1, 1], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
        [-1, 2], [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2],
        [-1, 3], [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3],
        [-1, 4], [0, 4], [1, 4], [2, 4], [3, 4], [4, 4],
        [0, 5], [1, 5]
    ]
    if len(p2) == 32 and len(set(map(tuple, p2))) == 32:
        configs.append(p2)

    best_points = None
    best_edges = 0
    
    for cfg in configs:
        e = count_edges(cfg)
        if e > best_edges:
            best_edges = e
            best_points = [list(p) for p in cfg]
            
    # If we haven't reached 76, do a local search
    if best_points and best_edges < 76:
        import random
        random.seed(42)
        current = [list(p) for p in best_points]
        current_edges = best_edges
        
        # Local search: move one point to a random empty neighbor
        # Or try all empty neighbors for each point
        for iter in range(500):
            if time.time() - start_time > DEADLINE - 5:
                break
            improved = False
            for i in range(32):
                if time.time() - start_time > DEADLINE - 5:
                    break
                ox, oy = current[i][0], current[i][1]
                # Try moving to all 6 neighbors
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
                    nx, ny = ox + dx, oy + dy
                    if abs(nx) > 64 or abs(ny) > 64:
                        continue
                    # Check if (nx, ny) is occupied
                    occupied = False
                    for k, p in enumerate(current):
                        if k != i and p[0] == nx and p[1] == ny:
                            occupied = True
                            break
                    if occupied:
                        continue
                    
                    # Check distance to all others
                    valid = True
                    for k in range(32):
                        if k == i: continue
                        if dist_sq([nx, ny], current[k]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Try the move
                    current[i][0] = nx
                    current[i][1] = ny
                    new_edges = count_edges(current)
                    if new_edges > current_edges:
                        current_edges = new_edges
                        improved = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_points = [list(p) for p in current]
                    else:
                        # Revert
                        current[i][0] = ox
                        current[i][1] = oy
            
            if not improved:
                # Try random restarts or perturbations if stuck?
                # For now, if no improvement in a full pass, break or do random jumps.
                # Let's do a random jump to escape local minima
                if random.random() < 0.1:
                    idx = random.randint(0, 31)
                    ox, oy = current[idx][0], current[idx][1]
                    # Move to a random nearby spot
                    for _ in range(10):
                        dx = random.randint(-2, 2)
                        dy = random.randint(-2, 2)
                        nx, ny = ox + dx, oy + dy
                        if abs(nx) > 64 or abs(ny) > 64: continue
                        occupied = any(k != idx and p[0]==nx and p[1]==ny for k, p in enumerate(current))
                        if occupied: continue
                        valid = all(dist_sq([nx, ny], current[k]) >= 1 for k in range(32) if k != idx)
                        if not valid: continue
                        current[idx][0] = nx
                        current[idx][1] = ny
                        new_edges = count_edges(current)
                        if new_edges > current_edges:
                            current_edges = new_edges
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_points = [list(p) for p in current]
                        else:
                            current[idx][0] = ox
                            current[idx][1] = oy
                        break

    # Fallback if something went wrong
    if best_points is None:
        return [[a, b] for a in range(16) for b in range(2)]
        
    return best_points
