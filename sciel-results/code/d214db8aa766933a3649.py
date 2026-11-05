"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import math
import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def _line_key(x1, y1, x2, y2):
    """Compute normalized line key from two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    a = dy
    b = -dx
    c = a * x1 + b * y1
    g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _evaluate(points):
    """Return the number of 4-point lines, or -1 if invalid."""
    n = len(points)
    if n != 16:
        return -1
    
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
        if (x, y) in seen:
            return -1
        seen.add((x, y))
        
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            if key is None:
                continue
            count = line_counts.get(key, 0) + 1
            line_counts[key] = count
            if count >= 5:
                return -1

    four_count = 0
    for count in line_counts.values():
        if count == 4:
            four_count += 1
    return four_count

def _affine_plane_seed(rng, scale=1, offset=(0, 0)):
    """Generate an affine plane of order 4 configuration."""
    points = []
    ox, oy = offset
    for x in range(4):
        for y in range(4):
            points.append([x * scale + ox, y * scale + oy])
    return points

def _perturb_structured(points, rng):
    """Perturb points with structured moves that preserve line structure."""
    new_points = [p[:] for p in points]
    move_type = rng.random()
    
    if move_type < 0.4:
        # Move one point slightly
        idx = rng.randint(0, 15)
        new_points[idx][0] += rng.randint(-3, 3)
        new_points[idx][1] += rng.randint(-3, 3)
    elif move_type < 0.6:
        # Move two points
        i, j = rng.sample(range(16), 2)
        new_points[i][0] += rng.randint(-2, 2)
        new_points[i][1] += rng.randint(-2, 2)
        new_points[j][0] += rng.randint(-2, 2)
        new_points[j][1] += rng.randint(-2, 2)
    elif move_type < 0.8:
        # Swap two points' coordinates (partial)
        i, j = rng.sample(range(16), 2)
        if rng.random() < 0.5:
            new_points[i][0], new_points[j][0] = new_points[j][0], new_points[i][0]
        else:
            new_points[i][1], new_points[j][1] = new_points[j][1], new_points[i][1]
    else:
        # Translate a pair
        i, j = rng.sample(range(16), 2)
        dx = rng.randint(-2, 2)
        dy = rng.randint(-2, 2)
        new_points[i][0] += dx
        new_points[i][1] += dy
        new_points[j][0] += dx
        new_points[j][1] += dy
        
    return new_points

def construct():
    """Find 16 integer points maximizing 4-point lines with no five collinear."""
    deadline = time.time() + 110
    rng = random.Random(42)
    
    best_points = None
    best_score = -1
    
    # Try multiple affine plane seeds with different scales and offsets
    seeds = []
    for scale in [1, 2, 3]:
        for ox in range(-2, 3):
            for oy in range(-2, 3):
                pts = _affine_plane_seed(rng, scale, (ox, oy))
                # Check bounds
                valid = True
                for p in pts:
                    if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                        valid = False
                        break
                if valid:
                    seeds.append(pts)
    
    # Also add some random configurations
    for _ in range(20):
        pts = [[rng.randint(-50, 50), rng.randint(-50, 50)] for _ in range(16)]
        seeds.append(pts)
    
    # Evaluate all seeds
    for pts in seeds:
        score = _evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
    
    if best_points is None:
        best_points = _affine_plane_seed(rng, 1, (0, 0))
        best_score = _evaluate(best_points)
    
    # Simulated annealing from best
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    temperature = 10.0
    cooling = 0.999
    min_temperature = 0.1
    
    while time.time() < deadline - 5 and temperature > min_temperature:
        new_points = _perturb_structured(current_points, rng)
        new_score = _evaluate(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        elif new_score >= 0:
            delta = current_score - new_score
            if rng.random() < math.exp(-delta / max(temperature, 0.01)):
                current_points = new_points
                current_score = new_score
        
        temperature *= cooling
    
    # Final polish: try small moves from best
    for _ in range(10000):
        if time.time() > deadline - 2:
            break
        new_points = [p[:] for p in best_points]
        idx = rng.randint(0, 15)
        new_points[idx][0] += rng.randint(-1, 1)
        new_points[idx][1] += rng.randint(-1, 1)
        new_score = _evaluate(new_points)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points
    
    return best_points

# EVOLVE-BLOCK-END
