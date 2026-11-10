import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

# Precompute all triple indices for 30 points
_TRIPLES = list(combinations(range(30), 3))
_N_TRIPLES = len(_TRIPLES)

def _score_fast(pts):
    """Compute max multiplicity of positive doubled area using NumPy vectorization."""
    pts_arr = np.array(pts, dtype=np.int64)
    dets = np.zeros(_N_TRIPLES, dtype=np.int64)
    for t, (i, j, k) in enumerate(_TRIPLES):
        dets[t] = abs(pts_arr[i,0]*(pts_arr[j,1]-pts_arr[k,1]) +
                      pts_arr[j,0]*(pts_arr[k,1]-pts_arr[i,1]) +
                      pts_arr[k,0]*(pts_arr[i,1]-pts_arr[j,1]))
    positive = dets[dets > 0]
    if len(positive) == 0:
        return 0
    # Count frequencies
    vals, counts = np.unique(positive, return_counts=True)
    return int(counts.max())

def _score_incremental(pts, old_dets, moved_idx):
    """Recompute only triples involving moved_idx. Returns (new_score, new_dets)."""
    pts_arr = np.array(pts, dtype=np.int64)
    new_dets = old_dets.copy()
    for t, (i, j, k) in enumerate(_TRIPLES):
        if i == moved_idx or j == moved_idx or k == moved_idx:
            new_dets[t] = abs(pts_arr[i,0]*(pts_arr[j,1]-pts_arr[k,1]) +
                              pts_arr[j,0]*(pts_arr[k,1]-pts_arr[i,1]) +
                              pts_arr[k,0]*(pts_arr[i,1]-pts_arr[j,1]))
    positive = new_dets[new_dets > 0]
    if len(positive) == 0:
        return 0, new_dets
    vals, counts = np.unique(positive, return_counts=True)
    return int(counts.max()), new_dets

def _gen_rect_grid(rows, cols):
    """Generate a centered rectangular grid of rows x cols points."""
    x0 = -(cols - 1) // 2
    y0 = -(rows - 1) // 2
    pts = [(x0 + x, y0 + y) for x in range(cols) for y in range(rows)]
    if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
        return pts
    return None

def _gen_hex_config(n=30):
    """Generate a hexagonal-ish configuration."""
    pts = []
    # Center
    pts.append((0, 0))
    # Rings
    for r in range(1, 8):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if abs(dx) + abs(dy) == r and len(pts) < n:
                    pts.append((dx, dy))
    # Fill remaining
    if len(pts) < n:
        for x in range(-5, 6):
            for y in range(-5, 6):
                if (x, y) not in pts and len(pts) < n:
                    pts.append((x, y))
    return pts[:n]

def _gen_lattice(a, b, c, d, rows=6, cols=5):
    """Generate a parallelogram lattice configuration."""
    if abs(a*d - b*c) == 0:
        return None
    pts = []
    for i in range(rows):
        for j in range(cols):
            x = i * a + j * c
            y = i * b + j * d
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    if len(pts) == 30 and len(set(pts)) == 30:
        return pts
    return None

def _gen_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # Rectangular grids
    for (rows, cols) in [(6,5), (5,6), (10,3), (3,10), (15,2), (2,15), (30,1), (1,30), (5,4), (4,5)]:
        if rows * cols != 30:
            continue
        pts = _gen_rect_grid(rows, cols)
        if pts:
            configs.append(pts)
    
    # Hexagonal
    hex_pts = _gen_hex_config(30)
    if hex_pts and len(set(hex_pts)) == 30:
        configs.append(hex_pts)
    
    # Lattice configurations
    for a in range(-4, 5):
        for b in range(-4, 5):
            for c in range(-4, 5):
                for d in range(-4, 5):
                    pts = _gen_lattice(a, b, c, d)
                    if pts:
                        configs.append(pts)
    
    # Random configurations
    random.seed(12345)
    for _ in range(20):
        pts = []
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        configs.append(pts)
    
    return configs

def _local_search(pts, deadline, max_iters=3000):
    """Local search with incremental scoring and target-area guidance."""
    current = list(pts)
    pts_arr = np.array(current, dtype=np.int64)
    current_dets = np.zeros(_N_TRIPLES, dtype=np.int64)
    for t, (i, j, k) in enumerate(_TRIPLES):
        current_dets[t] = abs(pts_arr[i,0]*(pts_arr[j,1]-pts_arr[k,1]) +
                              pts_arr[j,0]*(pts_arr[k,1]-pts_arr[i,1]) +
                              pts_arr[k,0]*(pts_arr[i,1]-pts_arr[j,1]))
    
    current_score = _score_fast(current)
    best_score = current_score
    best_config = list(current)
    current_set = set(current)
    
    # Find the target area (most frequent positive area)
    positive = current_dets[current_dets > 0]
    if len(positive) > 0:
        vals, counts = np.unique(positive, return_counts=True)
        target_area = int(vals[counts.argmax()])
    else:
        target_area = 0
    
    no_improve = 0
    max_no_improve = 150
    iter_count = 0
    
    # Precompute which triples involve each point
    triples_by_point = [[] for _ in range(30)]
    for t, (i, j, k) in enumerate(_TRIPLES):
        triples_by_point[i].append(t)
        triples_by_point[j].append(t)
        triples_by_point[k].append(t)
    
    while time.time() < deadline and iter_count < max_iters and no_improve < max_no_improve:
        iter_count += 1
        idx = random.randint(0, 29)
        old_point = current[idx]
        ox, oy = old_point
        
        # Generate candidate moves with bias toward target area
        moves = []
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx == 0 and dy == 0:
                    continue
                np_ = (ox + dx, oy + dy)
                if -30 <= np_[0] <= 30 and -30 <= np_[1] <= 30:
                    if np_ not in current_set:
                        moves.append(np_)
        
        if not moves:
            no_improve += 1
            continue
        
        random.shuffle(moves)
        best_move = None
        best_move_score = current_score
        best_move_dets = None
        
        # Try up to 25 moves per iteration
        for np_ in moves[:25]:
            if time.time() > deadline:
                break
            
            # Temporarily move
            current[idx] = np_
            current_set.discard(old_point)
            current_set.add(np_)
            
            # Incremental score
            new_score, new_dets = _score_incremental(current, current_dets, idx)
            
            # Revert
            current[idx] = old_point
            current_set.discard(np_)
            current_set.add(old_point)
            
            if new_score > best_move_score:
                best_move_score = new_score
                best_move = np_
                best_move_dets = new_dets
        
        if best_move is not None:
            current[idx] = best_move
            current_set.discard(old_point)
            current_set.add(best_move)
            current_dets = best_move_dets
            current_score = best_move_score
            if current_score > best_score:
                best_score = current_score
                best_config = list(current)
                # Update target area
                positive = current_dets[current_dets > 0]
                if len(positive) > 0:
                    vals, counts = np.unique(positive, return_counts=True)
                    target_area = int(vals[counts.argmax()])
            no_improve = 0
        else:
            no_improve += 1
    
    return best_config, best_score

def _polish(pts, deadline):
    """Final high-precision polish with tighter search radius."""
    current = list(pts)
    pts_arr = np.array(current, dtype=np.int64)
    current_dets = np.zeros(_N_TRIPLES, dtype=np.int64)
    for t, (i, j, k) in enumerate(_TRIPLES):
        current_dets[t] = abs(pts_arr[i,0]*(pts_arr[j,1]-pts_arr[k,1]) +
                              pts_arr[j,0]*(pts_arr[k,1]-pts_arr[i,1]) +
                              pts_arr[k,0]*(pts_arr[i,1]-pts_arr[j,1]))
    
    current_score = _score_fast(current)
    best_score = current_score
    best_config = list(current)
    current_set = set(current)
    
    # Precompute triples by point
    triples_by_point = [[] for _ in range(30)]
    for t, (i, j, k) in enumerate(_TRIPLES):
        triples_by_point[i].append(t)
        triples_by_point[j].append(t)
        triples_by_point[k].append(t)
    
    # Try all single-point moves with radius 2
    for idx in range(30):
        if time.time() > deadline:
            break
        old_point = current[idx]
        ox, oy = old_point
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                np_ = (ox + dx, oy + dy)
                if -30 <= np_[0] <= 30 and -30 <= np_[1] <= 30:
                    if np_ not in current_set:
                        current[idx] = np_
                        current_set.discard(old_point)
                        current_set.add(np_)
                        
                        new_score, new_dets = _score_incremental(current, current_dets, idx)
                        
                        if new_score > current_score:
                            current_dets = new_dets
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_config = list(current)
                        else:
                            current[idx] = old_point
                            current_set.discard(np_)
                            current_set.add(old_point)
    
    return best_config, best_score

def _try_all_configs(configs, deadline):
    """Evaluate all initial configs and pick the best."""
    best_score = 0
    best_config = configs[0] if configs else [(x, 0) for x in range(30)]
    
    for cfg in configs:
        if time.time() > deadline:
            break
        if len(cfg) != 30 or len(set(cfg)) != 30:
            continue
        if not all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in cfg):
            continue
        s = _score_fast(cfg)
        if s > best_score:
            best_score = s
            best_config = list(cfg)
    
    return best_config, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s limit
    
    # Generate and evaluate initial configs
    configs = _gen_initial_configs()
    best_config, best_score = _try_all_configs(configs, deadline)
    
    # Local search
    if time.time() < deadline - 5:
        best_config, best_score = _local_search(best_config, deadline)
    
    # Final polish
    if time.time() < deadline - 2:
        best_config, best_score = _polish(best_config, deadline)
    
    return best_config
