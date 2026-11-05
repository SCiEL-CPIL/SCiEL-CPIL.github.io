"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import math
import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def _count_four_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the direction vector
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g > 0:
                dx //= g
                dy //= g
            # Ensure canonical form
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            # Compute line key: (a, b, c) for ax + by = c
            a = dy
            b = -dx
            c = a * x1 + b * y1
            # Normalize the line equation
            g2 = math.gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            line_counts[key] += 1
    four_count = 0
    for count in line_counts.values():
        if count == 4:
            four_count += 1
    return four_count

def _has_five_collinear(points):
    """Check if any line contains 5 or more points."""
    n = len(points)
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g2 = math.gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            line_counts[key] += 1
            if line_counts[key] >= 5:
                return True
    return False

def _is_valid(points):
    """Check if points are valid: distinct, within bounds, no five collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if _has_five_collinear(points):
        return False
    return True

def _evaluate(points):
    """Return the number of 4-point lines, or -1 if invalid."""
    if not _is_valid(points):
        return -1
    return _count_four_lines(points)

def _random_point(rng, x_range=(-1000, 1000), y_range=(-1000, 1000)):
    """Generate a random point within bounds."""
    x = rng.randint(x_range[0], x_range[1])
    y = rng.randint(y_range[0], y_range[1])
    return [x, y]

def _perturb(points, rng):
    """Perturb one or two points to create a neighbor."""
    new_points = [p[:] for p in points]
    num_moves = rng.choice([1, 1, 1, 2])
    for _ in range(num_moves):
        idx = rng.randint(0, 15)
        if rng.random() < 0.5:
            # Small move
            new_points[idx][0] += rng.randint(-5, 5)
            new_points[idx][1] += rng.randint(-5, 5)
        else:
            # Larger move
            new_points[idx][0] += rng.randint(-50, 50)
            new_points[idx][1] += rng.randint(-50, 50)
    return new_points

def construct():
    """Find 16 integer points maximizing 4-point lines with no five collinear."""
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with the 4x4 grid (known to have 10 four-point lines)
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _evaluate(best_points)
    
    rng = random.Random(42)
    
    # Simulated annealing
    temperature = 50.0
    cooling = 0.995
    min_temperature = 0.1
    iterations = 0
    
    while time.time() < deadline:
        if temperature < min_temperature:
            break
        
        new_points = _perturb(best_points, rng)
        new_score = _evaluate(new_points)
        
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            temperature = min(temperature, 50.0)  # Reset temperature on improvement
        elif new_score >= 0:
            # Accept worse solution with probability based on temperature
            delta = best_score - new_score
            if rng.random() < math.exp(-delta / max(temperature, 0.01)):
                best_points = new_points
                best_score = new_score
        
        temperature *= cooling
        iterations += 1
        
        # Periodic restart with random configuration if stuck
        if iterations % 5000 == 0 and temperature < 5.0:
            random_points = [_random_point(rng) for _ in range(16)]
            random_score = _evaluate(random_points)
            if random_score > best_score:
                best_points = random_points
                best_score = random_score
            temperature = 20.0
    
    return best_points

# EVOLVE-BLOCK-END
