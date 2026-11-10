import random
import math

def construct():
    def get_score(pts):
        if len(set(map(tuple, pts))) != 16:
            return -1000000
        for p in pts:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return -1000000
        
        line_counts = {}
        for i in range(16):
            for j in range(i+1, 16):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                A = y2 - y1
                B = x1 - x2
                C = A * x1 + B * y1
                
                def gcd(a, b):
                    a, b = abs(a), abs(b)
                    while b:
                        a, b = b, a % b
                    return a
                    
                g = gcd(gcd(A, B), C)
                if g == 0:
                    continue
                A_n = A // g
                B_n = B // g
                C_n = C // g
                
                if A_n < 0 or (A_n == 0 and B_n < 0) or (A_n == 0 and B_n == 0 and C_n < 0):
                    A_n = -A_n
                    B_n = -B_n
                    C_n = -C_n
                    
                key = (A_n, B_n, C_n)
                line_counts[key] = line_counts.get(key, 0) + 1
        
        final_score = 0
        for key, pair_count in line_counts.items():
            disc = 1 + 8 * pair_count
            sqrt_disc = int(math.sqrt(disc))
            if sqrt_disc * sqrt_disc != disc:
                continue
            k = (1 + sqrt_disc) // 2
            if k * (k-1) // 2 != pair_count:
                continue
            if k > 4:
                return -1000000
            elif k == 4:
                final_score += 1
        return final_score

    def simulated_annealing(initial_points, iterations=10000):
        current_points = [p[:] for p in initial_points]
        current_score = get_score(current_points)
        best_points = [p[:] for p in current_points]
        best_score = current_score
        
        start_temp = 5.0
        end_temp = 0.01
        
        for i in range(iterations):
            temp = start_temp * (end_temp / start_temp) ** (i / iterations)
            
            # Random move
            move_type = random.randint(0, 2)
            
            if move_type == 0:
                # Move one point
                idx = random.randint(0, 15)
                old_point = current_points[idx][:]
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                new_point = [old_point[0] + dx, old_point[1] + dy]
                
                if abs(new_point[0]) > 1000 or abs(new_point[1]) > 1000:
                    continue
                if any(p == new_point for j, p in enumerate(current_points) if j != idx):
                    continue
                    
                current_points[idx] = new_point
                new_score = get_score(current_points)
                
                if new_score >= current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                else:
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / temp):
                        current_score = new_score
                    else:
                        current_points[idx] = old_point
                        
            elif move_type == 1:
                # Swap two points
                idx1, idx2 = random.sample(range(16), 2)
                p1 = current_points[idx1][:]
                p2 = current_points[idx2][:]
                current_points[idx1] = p2
                current_points[idx2] = p1
                
                new_score = get_score(current_points)
                
                if new_score >= current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                else:
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / temp):
                        current_score = new_score
                    else:
                        current_points[idx1] = p1
                        current_points[idx2] = p2
                        
            else:
                # Reflect one point across origin
                idx = random.randint(0, 15)
                old_point = current_points[idx][:]
                new_point = [-old_point[0], -old_point[1]]
                
                if abs(new_point[0]) > 1000 or abs(new_point[1]) > 1000:
                    continue
                if any(p == new_point for j, p in enumerate(current_points) if j != idx):
                    continue
                    
                current_points[idx] = new_point
                new_score = get_score(current_points)
                
                if new_score >= current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                else:
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / temp):
                        current_score = new_score
                    else:
                        current_points[idx] = old_point
                        
        return best_points, best_score

    # Start with 4x4 grid
    grid_points = [[x, y] for x in range(4) for y in range(4)]
    best_points, best_score = simulated_annealing(grid_points)
    
    # Try a few random restarts with perturbed grids
    for _ in range(5):
        perturbed = [p[:] for p in grid_points]
        for i in range(16):
            perturbed[i][0] += random.randint(-2, 2)
            perturbed[i][1] += random.randint(-2, 2)
        # Ensure distinct
        # If not distinct, skip or fix
        if len(set(map(tuple, perturbed))) != 16:
            continue
        pts, sc = simulated_annealing(perturbed, iterations=5000)
        if sc > best_score:
            best_score = sc
            best_points = pts

    return best_points
