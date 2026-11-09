import time
import random
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin

    def canonical_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return None
        dx //= g
        dy //= g
        # Normalize direction
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        
        # Line equation: ax + by = c
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)

    def score(points):
        """
        Returns (num_4_lines, num_5_plus_lines)
        """
        line_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = canonical_line(points[i], points[j])
                if key is not None:
                    line_counts[key] += 1
        
        num_4 = 0
        num_5_plus = 0
        for key, count in line_counts.items():
            # count is number of pairs
            # k points -> k*(k-1)/2 pairs
            # 4 points -> 6 pairs
            # 5 points -> 10 pairs
            if count == 6:
                num_4 += 1
            elif count >= 10:
                num_5_plus += 1
        return num_4, num_5_plus

    def evaluate(points):
        """
        Score: High is good.
        Penalty for 5+ collinear is severe.
        """
        n4, n5 = score(points)
        # If n5 > 0, it's invalid for the "no 5 collinear" constraint.
        # The task says "no five collinear". So n5 must be 0.
        # Score = n4 - 1000 * n5
        return n4 - 1000 * n5

    def get_initial_configs():
        configs = []
        
        # 1. Standard 4x4 grid
        grid = []
        for i in range(4):
            for j in range(4):
                grid.append([i, j])
        configs.append(grid)
        
        # 2. Perturbed grids with larger variations
        for _ in range(30):
            g = [p[:] for p in grid]
            for _ in range(random.randint(2, 8)):
                idx = random.randint(0, 15)
                # Use larger jumps to escape local optima
                jump = random.randint(1, 10)
                g[idx][0] += random.randint(-jump, jump)
                g[idx][1] += random.randint(-jump, jump)
            # Ensure distinctness and bounds
            valid = True
            for i in range(16):
                if abs(g[i][0]) > 1000 or abs(g[i][1]) > 1000:
                    valid = False
                    break
                for j in range(i+1, 16):
                    if g[i] == g[j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                configs.append(g)
                
        # 3. Random points in a larger box
        for _ in range(20):
            pts = []
            while len(pts) < 16:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if [x, y] not in pts:
                    pts.append([x, y])
            configs.append(pts)
            
        # 4. Staggered grid pattern
        staggered = []
        for i in range(4):
            for j in range(4):
                x = i
                y = j + (i % 2) * 0.5  # Stagger odd rows
                staggered.append([x * 2, int(y * 2)])  # Scale to integers
        configs.append(staggered)
        
        # 5. Diagonal-based configuration
        diag = []
        for i in range(4):
            for j in range(4):
                diag.append([i + j, i - j])
        configs.append(diag)
        
        return configs

    best_points = None
    best_score = -10000

    # Evaluate initial configs
    for config in get_initial_configs():
        s = evaluate(config)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in config]

    # Simulated Annealing with more aggressive moves
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    T = 15.0
    cooling_rate = 0.9995
    
    # Move types: single point move, pair move, triple move
    move_weights = [0.6, 0.25, 0.15]  # Single, pair, triple
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.choices([0, 1, 2], weights=move_weights)[0]
        
        if move_type == 0:
            # Move one point
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            # Use larger jump sizes to explore more
            jump = random.choice([1, 2, 3, 5, 8, 13, 21])
            new_x = old_x + random.randint(-jump, jump)
            new_y = old_y + random.randint(-jump, jump)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            # Check distinctness
            is_distinct = True
            for k, p in enumerate(current_points):
                if k != idx and p[0] == new_x and p[1] == new_y:
                    is_distinct = False
                    break
            
            if not is_distinct:
                continue
                
            # Apply move
            current_points[idx] = [new_x, new_y]
            new_score = evaluate(current_points)
            
            # Acceptance criterion
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                # Probability of accepting worse move
                delta = current_score - new_score
                if T > 0.01 and random.random() < pow(2.718, -delta / T):
                    current_score = new_score
                else:
                    # Revert
                    current_points[idx] = [old_x, old_y]
                    
        elif move_type == 1:
            # Move two points
            i, j = random.sample(range(16), 2)
            old_i = current_points[i][:]
            old_j = current_points[j][:]
            
            jump = random.choice([1, 2, 3, 5])
            new_i = [old_i[0] + random.randint(-jump, jump), old_i[1] + random.randint(-jump, jump)]
            new_j = [old_j[0] + random.randint(-jump, jump), old_j[1] + random.randint(-jump, jump)]
            
            if abs(new_i[0]) > 1000 or abs(new_i[1]) > 1000 or abs(new_j[0]) > 1000 or abs(new_j[1]) > 1000:
                continue
            
            # Check distinctness for new positions
            is_distinct = True
            for k, p in enumerate(current_points):
                if k == i or k == j:
                    continue
                if p == new_i or p == new_j:
                    is_distinct = False
                    break
            if new_i == new_j:
                is_distinct = False
                
            if not is_distinct:
                continue
                
            current_points[i] = new_i
            current_points[j] = new_j
            new_score = evaluate(current_points)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                delta = current_score - new_score
                if T > 0.01 and random.random() < pow(2.718, -delta / T):
                    current_score = new_score
                else:
                    current_points[i] = old_i
                    current_points[j] = old_j

        else:
            # Move three points
            i, j, k = random.sample(range(16), 3)
            old_i = current_points[i][:]
            old_j = current_points[j][:]
            old_k = current_points[k][:]
            
            jump = random.choice([1, 2, 3])
            new_i = [old_i[0] + random.randint(-jump, jump), old_i[1] + random.randint(-jump, jump)]
            new_j = [old_j[0] + random.randint(-jump, jump), old_j[1] + random.randint(-jump, jump)]
            new_k = [old_k[0] + random.randint(-jump, jump), old_k[1] + random.randint(-jump, jump)]
            
            if (abs(new_i[0]) > 1000 or abs(new_i[1]) > 1000 or 
                abs(new_j[0]) > 1000 or abs(new_j[1]) > 1000 or 
                abs(new_k[0]) > 1000 or abs(new_k[1]) > 1000):
                continue
            
            # Check distinctness for new positions
            is_distinct = True
            for idx, p in enumerate(current_points):
                if idx in (i, j, k):
                    continue
                if p == new_i or p == new_j or p == new_k:
                    is_distinct = False
                    break
            if new_i == new_j or new_j == new_k or new_i == new_k:
                is_distinct = False
                
            if not is_distinct:
                continue
                
            current_points[i] = new_i
            current_points[j] = new_j
            current_points[k] = new_k
            new_score = evaluate(current_points)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                delta = current_score - new_score
                if T > 0.01 and random.random() < pow(2.718, -delta / T):
                    current_score = new_score
                else:
                    current_points[i] = old_i
                    current_points[j] = old_j
                    current_points[k] = old_k

        T *= cooling_rate
        
        # If T is very low, restart with a higher T to escape local optima
        if T < 0.5:
            T = 15.0
            # Restart from best or a random perturbation of best
            if random.random() < 0.5:
                current_points = [p[:] for p in best_points]
                current_score = best_score
            else:
                current_points = [p[:] for p in best_points]
                # Perturb more aggressively
                for _ in range(random.randint(2, 5)):
                    idx = random.randint(0, 15)
                    jump = random.randint(3, 10)
                    current_points[idx][0] += random.randint(-jump, jump)
                    current_points[idx][1] += random.randint(-jump, jump)
                # Check validity
                valid = True
                for i in range(16):
                    if abs(current_points[i][0]) > 1000 or abs(current_points[i][1]) > 1000:
                        valid = False
                        break
                    for j in range(i+1, 16):
                        if current_points[i] == current_points[j]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    current_points = [p[:] for p in best_points]
                    current_score = best_score
                else:
                    current_score = evaluate(current_points)

    # Final verification
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(pts)
    print(len(pts))
    # Verify no 5 collinear
    from collections import defaultdict
    def canonical_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0: return (0,0,0)
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)
    
    line_counts = defaultdict(int)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            line_counts[canonical_line(pts[i], pts[j])] += 1
            
    max_collinear = 0
    for k, v in line_counts.items():
        # v is pairs. k points -> k(k-1)/2 pairs
        if v == 10: max_collinear = max(max_collinear, 5)
        elif v >= 15: max_collinear = max(max_collinear, 6)
        elif v == 6: max_collinear = max(max_collinear, 4)
        elif v == 3: max_collinear = max(max_collinear, 3)
        elif v == 1: max_collinear = max(max_collinear, 2)
        
    print("Max collinear:", max_collinear)
    # Count 4-lines
    count_4 = 0
    for k, v in line_counts.items():
        if v == 6:
            count_4 += 1
    print("4-lines:", count_4)
