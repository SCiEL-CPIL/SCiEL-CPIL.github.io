import numpy as np
import time
import random
from itertools import combinations

N = 30

# Precompute triple indices for N=30
_TRIPLE_IDX = np.array(list(combinations(range(N), 3)), dtype=np.int32)
_TRIPLE_I = _TRIPLE_IDX[:, 0]
_TRIPLE_J = _TRIPLE_IDX[:, 1]
_TRIPLE_K = _TRIPLE_IDX[:, 2]

def fast_score(points):
    """
    points: numpy array of shape (N, 2)
    Returns: max multiplicity of a positive doubled area
    """
    if points.shape[0] < 3:
        return 0
    
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]
    
    det = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
          (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    
    valid_dets = np.abs(det)
    valid_dets = valid_dets[valid_dets > 0]
    
    if len(valid_dets) == 0:
        return 0
        
    counts = np.bincount(valid_dets)
    return counts.max()

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Generate Initial Candidates
    candidates = []

    # Rectangular Grids W x H such that W*H = 30
    for W in [1, 2, 3, 5, 6, 10, 15, 30]:
        H = 30 // W
        if W * H != 30:
            continue
        pts = []
        for i in range(W):
            for j in range(H):
                x = i - W // 2
                y = j - H // 2
                pts.append((x, y))

        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            candidates.append(np.array(pts, dtype=int))

    # Hexagonal-ish lattice: points on a triangular grid pattern
    for scale in [1, 2, 3]:
        pts = []
        for i in range(-6, 7):
            for j in range(-6, 7):
                x = i * scale
                y = j * scale + (i % 2) * (scale // 2) if scale >= 2 else j * scale
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        # Take first 30
        if len(pts) >= N:
            random.shuffle(pts)
            candidates.append(np.array(pts[:N], dtype=int))

    # Add a "clustered" start: points close to each other
    for _ in range(3):
        pts = []
        base_x = random.randint(-20, 20)
        base_y = random.randint(-20, 20)
        while len(pts) < N:
            p = (base_x + random.randint(-5, 5), base_y + random.randint(-5, 5))
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))

    # Add random starts
    for _ in range(3):
        pts = []
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))

    # Evaluate candidates and pick the best few for local search
    scored_candidates = []
    for cand in candidates:
        s = fast_score(cand)
        scored_candidates.append((s, cand))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    best_global_score = -1
    best_global_pts = None

    # Run local search on the top 5 initial candidates
    for initial_score, initial_pts in scored_candidates[:5]:
        current_pts = initial_pts.copy()
        current_score = initial_score

        if current_score > best_global_score:
            best_global_score = current_score
            best_global_pts = current_pts.copy()

        # Simulated Annealing Parameters
        T = 2.0
        T_min = 0.01
        alpha = 0.997  # Slower cooling
        no_improve_count = 0
        max_no_improve = 800

        while time.time() < deadline:
            if current_score > best_global_score:
                best_global_score = current_score
                best_global_pts = current_pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

            # Pick a random point to move
            idx = np.random.randint(N)

            # Decide move type: 80% local, 20% global
            if random.random() < 0.8:
                # Local move: shift by -2 to 2 for finer granularity
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                if dx == 0 and dy == 0:
                    continue
                new_p = current_pts[idx] + np.array([dx, dy])
            else:
                # Global move: random point in box
                new_p = np.array([random.randint(-30, 30), random.randint(-30, 30)])

            # Check bounds
            if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
                continue

            # Check distinctness
            is_distinct = True
            for i in range(N):
                if i != idx and np.array_equal(current_pts[i], new_p):
                    is_distinct = False
                    break

            if not is_distinct:
                continue

            # Calculate new score
            new_pts = current_pts.copy()
            new_pts[idx] = new_p

            new_score = fast_score(new_pts)

            # Accept if better
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
            else:
                # Accept if worse with probability exp(-delta / T)
                delta = current_score - new_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_pts = new_pts
                        current_score = new_score
                # If equal, accept (to explore)
                elif random.random() < 0.5:
                    current_pts = new_pts
                    # current_score stays same

            # Cool down
            T = max(T_min, T * alpha)

    return [(int(p[0]), int(p[1])) for p in best_global_pts]
