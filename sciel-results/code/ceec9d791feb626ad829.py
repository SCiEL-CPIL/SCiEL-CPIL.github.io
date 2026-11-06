"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g == 0:
        return None
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _build_line_map(points):
    """Build a map from line key to list of point indices on that line."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = []
            line_map[key].append(i)
            line_map[key].append(j)
    # Deduplicate indices in each line
    for key in line_map:
        line_map[key] = sorted(set(line_map[key]))
    return line_map

def _evaluate(points):
    """
    Evaluate a configuration:
    - Returns (score, valid) where score = number of lines with exactly 4 points,
      valid = True if no line has >= 5 points.
    """
    n = len(points)
    line_map = _build_line_map(points)
    score = 0
    valid = True
    for key, indices in line_map.items():
        if len(indices) >= 5:
            valid = False
        if len(indices) == 4:
            score += 1
    return score, valid

def _count_4_lines(points):
    score, valid = _evaluate(points)
    return score

def _is_valid(points):
    _, valid = _evaluate(points)
    return valid

def _ag24_points():
    """
    Generate 16 points from a structured initial configuration.
    We use a perturbed 4x4 grid to break symmetry.
    """
    # Start with a 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    return points

def _random_initial_config(seed=None):
    """Generate a random initial configuration with no 5 collinear."""
    if seed is not None:
        random.seed(seed)
    attempts = 0
    max_attempts = 1000
    while attempts < max_attempts:
        attempts += 1
        points = []
        for _ in range(16):
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            # Check collision
            collision = False
            for p in points:
                if p[0] == x and p[1] == y:
                    collision = True
                    break
            if collision:
                continue
            points.append([x, y])
            if len(points) == 16:
                break
        if len(points) != 16:
            continue
        # Check validity
        if _is_valid(points):
            return points
    # Fallback to grid
    return _ag24_points()

def _local_search(points, max_time=100.0, deadline=None):
    """
    Local search to improve the configuration.
    Uses line-based move generation for efficiency.
    """
    start_time = time.time()
    if deadline is None:
        deadline = start_time + max_time
    
    best_points = [p[:] for p in points]
    best_score = _count_4_lines(best_points)
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Try moving each point
        for idx in range(16):
            if time.time() > deadline - 10.0:
                break
            
            x, y = points[idx]
            
            # Generate candidate moves: nearby positions
            candidates = []
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                        candidates.append((nx, ny))
            
            # Also try some random positions
            for _ in range(10):
                nx = random.randint(-15, 15)
                ny = random.randint(-15, 15)
                candidates.append((nx, ny))
            
            for nx, ny in candidates:
                if time.time() > deadline - 10.0:
                    break
                
                # Check collision
                collision = False
                for i in range(16):
                    if i == idx:
                        continue
                    if points[i][0] == nx and points[i][1] == ny:
                        collision = True
                        break
                if collision:
                    continue
                
                new_points = [p[:] for p in points]
                new_points[idx] = [nx, ny]
                
                # Check validity
                if not _is_valid(new_points):
                    continue
                
                score = _count_4_lines(new_points)
                if score > best_score:
                    best_score = score
                    best_points = new_points
                    points = new_points
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try a random jump
            if random.random() < 0.3:
                idx = random.randint(0, 15)
                nx = random.randint(-20, 20)
                ny = random.randint(-20, 20)
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    collision = False
                    for i in range(16):
                        if i == idx:
                            continue
                        if points[i][0] == nx and points[i][1] == ny:
                            collision = True
                            break
                    if not collision:
                        new_points = [p[:] for p in points]
                        new_points[idx] = [nx, ny]
                        if _is_valid(new_points):
                            score = _count_4_lines(new_points)
                            if score > best_score:
                                best_score = score
                                best_points = new_points
                                points = new_points
            else:
                break
                
    return best_points, best_score

def _multi_start_search(deadline=None):
    """
    Multi-start local search with different initial configurations.
    """
    if deadline is None:
        deadline = time.time() + 110.0
    
    best_points = None
    best_score = -1
    
    # Try several initial configurations
    initial_configs = [
        _ag24_points(),
        _random_initial_config(seed=1),
        _random_initial_config(seed=2),
        _random_initial_config(seed=3),
        _random_initial_config(seed=4),
    ]
    
    # Add some structured perturbations of the grid
    for _ in range(3):
        pts = _ag24_points()
        for i in range(16):
            if random.random() < 0.3:
                pts[i][0] += random.randint(-2, 2)
                pts[i][1] += random.randint(-2, 2)
        if _is_valid(pts):
            initial_configs.append(pts)
    
    for config in initial_configs:
        if time.time() > deadline - 15.0:
            break
        
        score, valid = _evaluate(config)
        if not valid:
            continue
        
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
        
        # Run local search on this config
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 5.0:
            improved_points, improved_score = _local_search(config, max_time=remaining_time, deadline=deadline)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Run multi-start search
    best_points, best_score = _multi_start_search(deadline=deadline)
    
    # Final verification
    if best_points is None:
        best_points = _ag24_points()
    
    # Ensure validity
    if not _is_valid(best_points):
        # Fallback to a known valid configuration
        best_points = _ag24_points()
    
    return best_points
# EVOLVE-BLOCK-END
