import time
import random
from math import gcd

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            lines[key] = lines.get(key, 0) + 1
            
    ordinary = 0
    for key, pair_count in lines.items():
        if pair_count == 1:
            ordinary += 1
            
    return ordinary

def local_search(points, deadline, max_iter=1000):
    """
    Local search to minimize ordinary lines.
    """
    n = len(points)
    best_points = [p[:] for p in points]
    best_score = count_ordinary_lines(best_points)
    
    # Precompute a set of candidate moves
    # Small steps
    small_steps = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6) if (dx, dy) != (0, 0)]
    # Large steps
    large_steps = [(dx, dy) for dx in range(-20, 21, 2) for dy in range(-20, 21, 2) if (dx, dy) != (0, 0)]
    
    iters = 0
    while iters < max_iter and time.time() < deadline:
        iters += 1
        improved = False
        
        # Try to improve by moving one point
        for i in range(n):
            x, y = best_points[i]
            
            # Try small steps first
            for dx, dy in small_steps:
                nx, ny = x + dx, y + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_pt = [nx, ny]
                if any(new_pt == p for p in best_points):
                    continue
                new_points = best_points[:]
                new_points[i] = new_pt
                new_score = count_ordinary_lines(new_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points
                    improved = True
                    break
            if improved:
                break
            
            # If no small step improved, try large steps
            for dx, dy in large_steps:
                nx, ny = x + dx, y + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_pt = [nx, ny]
                if any(new_pt == p for p in best_points):
                    continue
                new_points = best_points[:]
                new_points[i] = new_pt
                new_score = count_ordinary_lines(new_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try a random jump to escape local minima
            if random.random() < 0.5:
                i = random.randint(0, n-1)
                x, y = best_points[i]
                nx = x + random.randint(-50, 50)
                ny = y + random.randint(-50, 50)
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    new_pt = [nx, ny]
                    if not any(new_pt == p for p in best_points):
                        new_points = best_points[:]
                        new_points[i] = new_pt
                        new_score = count_ordinary_lines(new_points)
                        if new_score <= best_score:
                            best_score = new_score
                            best_points = new_points
            else:
                break
                
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Initial configurations
    configs = []
    
    # Config 1: 3x3 grid + 3 extensions
    base33 = [[x, y] for x in range(3) for y in range(3)]
    ext1 = [[3, 0], [0, 3], [3, 3]]
    configs.append(base33 + ext1)
    
    # Config 2: Different extensions
    ext2 = [[-1, 1], [1, -1], [4, 2]]
    configs.append(base33 + ext2)
    
    # Config 3: Another set
    ext3 = [[2, 3], [3, 2], [4, 1]]
    configs.append(base33 + ext3)
    
    # Config 4: Try points that create more collinearities
    ext4 = [[1, 3], [3, 1], [4, 0]]
    configs.append(base33 + ext4)
    
    # Config 5: Use a different base structure
    base2 = [[0,0],[1,0],[2,0],[3,0],
             [0,1],[0,2],[0,3],
             [1,1],[2,2],
             [4,1],[1,4]]
    configs.append(base2)
    
    # Config 6: Symmetric configuration
    base3 = [[-1,-1],[0,-1],[1,-1],
             [-1,0],[0,0],[1,0],
             [-1,1],[0,1],[1,1],
             [2,0],[0,2],[0,-2]]
    configs.append(base3)
    
    # Config 7: Another symmetric
    base4 = [[0,0],[1,0],[2,0],[3,0],
             [0,1],[1,1],[2,1],[3,1],
             [0,2],[3,2],[1,3],[2,-1]]
    configs.append(base4)
    
    best_points = None
    best_score = float('inf')
    
    # Run local search from each config
    for cfg in configs:
        if time.time() > deadline - 5:
            break
        if len(set(map(tuple, cfg))) != 12:
            continue
        pts, score = local_search(cfg, deadline, max_iter=500)
        if score < best_score:
            best_score = score
            best_points = pts
        if best_score == 6:
            break # Can't do better than 6
    
    # Final fallback
    if best_points is None:
        best_points = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
    
    return best_points
