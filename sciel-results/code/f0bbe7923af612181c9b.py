import time
import random
import numpy as np
from collections import defaultdict
from itertools import combinations

# EVOLVE-BLOCK-START

def _doubled_areas_np(pts):
    """Compute doubled areas for all triples. pts is (30,2) int array.
    Returns dict: area -> count (positive areas only)."""
    n = len(pts)
    # Vectorized: for each pair (i,j), compute cross product magnitude with all k
    # d(i,j,k) = |xi*yj - yi*xj + xj*yk - yj*xk + xk*yi - yk*xi|
    # = |(pi - pk) x (pj - pk)|
    # More efficient: fix k, compute all pairs among remaining
    cnt = defaultdict(int)
    # Precompute
    for k in range(n):
        pk = pts[k]
        # vectors from k to all other points
        diffs = pts - pk  # (n, 2)
        diffs[k] = 0
        # For each pair (i,j) with i,j != k, cross = diffs[i,0]*diffs[j,1] - diffs[i,1]*diffs[j,0]
        # Vectorized cross product matrix
        # cross[i,j] = diffs[i,0]*diffs[j,1] - diffs[i,1]*diffs[j,0]
        cross = diffs[:, 0][:, None] * diffs[:, 1][None, :] - diffs[:, 1][:, None] * diffs[:, 0][None, :]
        # Take upper triangle (i < j), exclude k
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mask[k, :] = False
        mask[:, k] = False
        vals = np.abs(cross[mask])
        vals = vals[vals > 0]
        if len(vals) > 0:
            # Count occurrences
            unique, counts = np.unique(vals, return_counts=True)
            for v, c in zip(unique, counts):
                cnt[int(v)] += int(c)
    return cnt


def _score_np(pts):
    """Return max multiplicity of any positive doubled area."""
    areas = _doubled_areas_np(pts)
    if not areas:
        return 0
    return max(areas.values())


def _score_np_fast(pts):
    """Even faster: just return the max count without building full dict if possible.
    Actually let's just use the same but track max."""
    n = len(pts)
    max_count = 0
    # We need to count across all triples, so we do need the full dict
    # But we can use a more efficient approach
    area_counts = defaultdict(int)
    for k in range(n):
        pk = pts[k]
        diffs = pts - pk
        diffs[k] = 0
        cross = diffs[:, 0][:, None] * diffs[:, 1][None, :] - diffs[:, 1][:, None] * diffs[:, 0][None, :]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mask[k, :] = False
        mask[:, k] = False
        vals = np.abs(cross[mask])
        vals = vals[vals > 0]
        if len(vals) > 0:
            unique, counts = np.unique(vals, return_counts=True)
            for v, c in zip(unique, counts):
                area_counts[int(v)] += int(c)
                if area_counts[int(v)] > max_count:
                    max_count = area_counts[int(v)]
    return max_count


def _hill_climb(pts, deadline, max_iters=5000):
    """Hill-climb: try moving each point to nearby positions, accept if score improves."""
    pts = np.array(pts, dtype=np.int64)
    n = len(pts)
    current_score = _score_np_fast(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    # Generate candidate positions in a local neighborhood
    # Points are in [-30, 30], so we can try all positions in a smaller grid
    # But that's 61*61 = 3721 positions per point, times 30 points = too many
    # Instead, try a smaller neighborhood around current position
    
    rng = random.Random(12345)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        # Try each point
        for idx in range(n):
            if time.time() > deadline:
                break
            x, y = pts[idx]
            # Try several candidate moves
            # 1. Random nearby positions
            candidates = []
            # Nearby grid positions
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if not any(pts[i] == (nx, ny) for i in range(n) if i != idx):
                            candidates.append((nx, ny))
            # Also try some random positions
            for _ in range(5):
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                if not any(pts[i] == (nx, ny) for i in range(n) if i != idx):
                    candidates.append((nx, ny))
            
            rng.shuffle(candidates)
            
            for cand in candidates[:15]:  # Limit candidates per point
                if time.time() > deadline:
                    break
                pts[idx] = cand
                new_score = _score_np_fast(pts)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = pts.copy()
                else:
                    pts[idx] = (x, y)
        
        if not improved:
            # Stuck, try a random restart of one point to escape local optimum
            idx = rng.randint(0, n - 1)
            x, y = pts[idx]
            for _ in range(20):
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                if not any(pts[i] == (nx, ny) for i in range(n) if i != idx):
                    pts[idx] = (nx, ny)
                    new_score = _score_np_fast(pts)
                    if new_score >= current_score:
                        if new_score > current_score:
                            current_score = new_score
                            if new_score > best_score:
                                best_score = new_score
                                best_pts = pts.copy()
                        break
                    else:
                        pts[idx] = (x, y)
    
    return best_pts.tolist(), best_score


def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 6x5 grid
    configs.append([(x, y) for x in range(6) for y in range(5)])
    # 5x6 grid
    configs.append([(x, y) for x in range(5) for y in range(6)])
    # 10x3 grid
    configs.append([(x, y) for x in range(10) for y in range(3)])
    # 15x2 grid
    configs.append([(x, y) for x in range(15) for y in range(2)])
    # 30x1 line
    configs.append([(x, 0) for x in range(30)])
    # Centered grids
    for base in range(-2, 3):
        configs.append([(x + base, y + base) for x in range(6) for y in range(5)])
    # Hexagonal-like
    hex_pts = []
    for row in range(6):
        offset = (row % 2) * 2
        for col in range(5):
            hex_pts.append((col * 4 + offset, row * 3))
    if len(hex_pts) == 30:
        configs.append(hex_pts)
    # Diamond/rhombus lattice
    diamond = []
    for i in range(6):
        for j in range(5):
            diamond.append((i + j, i - j))
    if len(diamond) == 30:
        configs.append(diamond)
    # Random configs
    for seed in range(20):
        rng = random.Random(seed)
        pts = []
        while len(pts) < 30:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        configs.append(pts)
    
    return configs


def construct():
    t0 = time.time()
    DEADLINE = t0 + 70.0  # Safety margin before 90s kill
    
    def elapsed():
        return time.time() - t0
    
    # Generate initial configs
    initial_configs = _generate_initial_configs()
    
    best_score = 0
    best_points = initial_configs[0]
    
    # Evaluate all initial configs
    for config in initial_configs:
        if elapsed() > DEADLINE:
            break
        if len(config) != 30:
            continue
        if not all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in config):
            continue
        s = _score_np_fast(np.array(config, dtype=np.int64))
        if s > best_score:
            best_score = s
            best_points = config
    
    # Hill-climb from each initial config (top few)
    # Sort configs by initial score
    scored_configs = []
    for config in initial_configs:
        if len(config) != 30:
            continue
        if not all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in config):
            continue
        s = _score_np_fast(np.array(config, dtype=np.int64))
        scored_configs.append((s, config))
    
    scored_configs.sort(key=lambda x: -x[0])
    
    # Try hill-climbing from top 10 configs
    for s, config in scored_configs[:10]:
        if elapsed() > DEADLINE:
            break
        result_pts, result_score = _hill_climb(config, DEADLINE, max_iters=3000)
        if result_score > best_score:
            best_score = result_score
            best_points = result_pts
    
    # If still have time, do more random restarts
    while elapsed() < DEADLINE - 5.0:
        rng = random.Random(int(elapsed() * 1000))
        pts = []
        while len(pts) < 30:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        
        result_pts, result_score = _hill_climb(pts, DEADLINE, max_iters=2000)
        if result_score > best_score:
            best_score = result_score
            best_points = result_pts
    
    return best_points

# EVOLVE-BLOCK-END
