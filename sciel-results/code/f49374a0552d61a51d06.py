import numpy as np
import time
import random
from math import gcd

def normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def count_incidences(points, lines):
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    vals = A * px + B * py + C
    return int(np.sum(vals == 0))

def get_best_points_for_lines(lines, candidate_points, k=54):
    L = lines.astype(np.int64)
    P = candidate_points.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    top_indices = np.argsort(counts)[::-1][:k]
    best_points = candidate_points[top_indices]
    return best_points, int(counts[top_indices].max())

def generate_candidate_points(size=30):
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()])

def generate_dense_line_pool(max_coeff=12, max_C=40):
    """Generate a dense pool of low-slope integer lines."""
    lines = []
    seen = set()
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_C, max_C + 1):
                nl = normalize_line(A, B, C)
                if nl and nl not in seen:
                    seen.add(nl)
                    lines.append(nl)
    return lines

def greedy_select_lines_from_points(points, n_lines=27, line_pool=None):
    P = points.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    n_pts = len(points)
    
    candidate_lines = []
    seen_lines = set()
    
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]
            A = int(dy)
            B = int(-dx)
            C = int(dx * points[i, 1] - dy * points[i, 0])
            nl = normalize_line(A, B, C)
            if nl and nl not in seen_lines:
                seen_lines.add(nl)
                candidate_lines.append(nl)
    
    if line_pool:
        for nl in line_pool:
            if nl not in seen_lines:
                seen_lines.add(nl)
                candidate_lines.append(nl)
    
    if not candidate_lines:
        for i in range(27):
            candidate_lines.append((1, 0, -i))
        return np.array(candidate_lines[:27], dtype=np.int64)
    
    line_counts = {}
    for nl in candidate_lines:
        A, B, C = nl
        vals = A * px + B * py + C
        line_counts[nl] = int(np.sum(vals == 0))
    
    selected_lines = []
    selected_set = set()
    
    for _ in range(n_lines):
        best_line = None
        best_score = -1
        for nl in candidate_lines:
            if nl in selected_set:
                continue
            score = line_counts[nl]
            if score > best_score:
                best_score = score
                best_line = nl
        if best_line is None:
            break
        selected_lines.append(best_line)
        selected_set.add(best_line)
    
    while len(selected_lines) < n_lines:
        A = random.randint(-10, 10)
        B = random.randint(-10, 10)
        C = random.randint(-30, 30)
        nl = normalize_line(A, B, C)
        if nl and nl not in selected_set:
            selected_lines.append(nl)
            selected_set.add(nl)
    
    return np.array(selected_lines[:n_lines], dtype=np.int64)

def generate_grid_based_lines():
    lines = []
    for c in range(-4, 5):
        lines.append((1, 0, -c))
    for c in range(-4, 5):
        lines.append((0, 1, -c))
    for c in range(-4, 5):
        lines.append((1, -1, -c))
    return np.array(lines, dtype=np.int64)

def generate_grid_points():
    grid_size = 9
    center = grid_size // 2
    xs = np.arange(-center, center + 1)
    ys = np.arange(-center, center + 1)
    XX, YY = np.meshgrid(xs, ys)
    all_points = np.column_stack([XX.ravel(), YY.ravel()])
    distances = np.sqrt(XX.ravel()**2 + YY.ravel()**2)
    order = np.argsort(distances)
    selected = all_points[order[:54]]
    return selected

def joint_optimize(lines, points, candidate_points, line_pool, time_limit, seed=42):
    """Joint point-line optimization with submodular-style greedy selection."""
    random.seed(seed)
    current_lines = lines.copy()
    current_points = points.copy()
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 8.0
    cooling = 0.9998
    min_T = 0.3
    
    no_improve = 0
    max_no_improve = 300
    
    # Precompute point incidence vectors for speed
    def compute_point_incidence(pts, lns):
        P = pts.astype(np.int64)
        L = lns.astype(np.int64)
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        vals = A * px + B * py + C
        return (vals == 0).astype(np.int8)
    
    def compute_line_incidence(pts, lns):
        P = pts.astype(np.int64)
        L = lns.astype(np.int64)
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        vals = A * px + B * py + C
        return (vals == 0).astype(np.int8)
    
    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            T = max(T, 8.0)
            no_improve = 0
            max_no_improve = int(max_no_improve * 1.2)
        
        move_type = random.random()
        
        if move_type < 0.35:
            # Replace a line: pick from point pairs or line pool
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            if random.random() < 0.6:
                p1_idx, p2_idx = random.sample(range(54), 2)
                p1 = current_points[p1_idx]
                p2 = current_points[p2_idx]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                A = int(dy)
                B = int(-dx)
                C = int(dx * p1[1] - dy * p1[0])
            else:
                nl_candidate = random.choice(line_pool)
                A, B, C = nl_candidate
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
            
            current_lines[idx] = new_line
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                no_improve += 1
                
        elif move_type < 0.55:
            # Replace a point: swap with a better candidate
            idx = random.randint(0, 53)
            old_point = current_points[idx].copy()
            
            # Find a candidate point that would improve things
            L = current_lines.astype(np.int64)
            cand = candidate_points
            px = cand[:, 0].reshape(-1, 1)
            py = cand[:, 1].reshape(-1, 1)
            A = L[:, 0].reshape(1, -1)
            B = L[:, 1].reshape(1, -1)
            C = L[:, 2].reshape(1, -1)
            vals = A * px + B * py + C
            cand_counts = np.sum(vals == 0, axis=1)
            
            # Exclude current points
            current_set = set(map(tuple, current_points))
            available = [i for i in range(len(cand)) if tuple(cand[i]) not in current_set]
            
            if not available:
                continue
            
            # Pick top few candidates
            avail_arr = np.array(available)
            avail_counts = cand_counts[avail_arr]
            top_avail = avail_arr[np.argsort(avail_counts)[::-1][:10]]
            
            new_point_idx = random.choice(top_avail)
            new_point = cand[new_point_idx]
            
            # Check if it's already in current points
            if tuple(new_point) in current_set:
                continue
            
            current_points[idx] = new_point
            new_I = count_incidences(current_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
                current_I = new_I
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_points[idx] = old_point
                no_improve += 1
                
        elif move_type < 0.75:
            # Re-optimize points for current lines
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.90:
            # Swap two lines
            i1, i2 = random.sample(range(27), 2)
            old_l1 = current_lines[i1].copy()
            old_l2 = current_lines[i2].copy()
            
            # Generate new lines
            p1_idx, p2_idx = random.sample(range(54), 2)
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A1 = int(dy)
            B1 = int(-dx)
            C1 = int(dx * p1[1] - dy * p1[0])
            nl1 = normalize_line(A1, B1, C1)
            
            nl2 = random.choice(line_pool)
            
            if nl1 is None or tuple(nl1) == tuple(old_l1) or tuple(nl1) == tuple(old_l2) or tuple(nl2) == tuple(old_l1) or tuple(nl2) == tuple(old_l2):
                continue
            
            current_lines[i1] = nl1
            current_lines[i2] = nl2
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or random.random() < np.exp(delta_I / (T * 0.5)):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[i1] = old_l1
                current_lines[i2] = old_l2
                no_improve += 1
                
        else:
            # Rebuild: greedily re-select lines from current points
            new_lines = greedy_select_lines_from_points(current_points, 27, line_pool)
            
            unique_lines = []
            for l in new_lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and tuple(nl) not in [tuple(u) for u in unique_lines]:
                    unique_lines.append(nl)
            
            if len(unique_lines) >= 27:
                new_lines = np.array(unique_lines[:27], dtype=np.int64)
                new_points, _ = get_best_points_for_lines(new_lines, candidate_points)
                new_I = count_incidences(new_points, new_lines)
                
                delta_I = new_I - current_I
                
                if delta_I > 0 or random.random() < np.exp(delta_I / (T * 0.3)):
                    current_I = new_I
                    current_lines = new_lines
                    current_points = new_points
                    if current_I > best_I:
                        best_I = current_I
                        best_lines = current_lines.copy()
                        best_points = current_points.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
        T *= cooling
        T = max(T, min_T)
        
    return best_lines, best_points, best_I

def polish_configuration(points, lines, candidate_points, time_limit, line_pool=None):
    """Final polish: local search focusing on improving worst lines and points."""
    random.seed(99999)
    current_points = points.copy()
    current_lines = lines.copy()
    current_I = count_incidences(current_points, current_lines)
    
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_I = current_I
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        P = current_points.astype(np.int64)
        L = current_lines.astype(np.int64)
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        vals = A * px + B * py + C
        inc_matrix = (vals == 0)
        line_counts = np.sum(inc_matrix, axis=1)
        point_counts = np.sum(inc_matrix, axis=0)
        
        # Try to improve worst line
        worst_line_idx = np.argmin(line_counts)
        improved = False
        
        for _ in range(20):
            # Try a line through two high-incidence points
            top_pts = np.argsort(point_counts)[::-1][:25]
            p1_idx, p2_idx = random.sample(list(top_pts), 2)
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = int(dy)
            B = int(-dx)
            C = int(dx * p1[1] - dy * p1[0])
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
            
            old_line = current_lines[worst_line_idx].copy()
            current_lines[worst_line_idx] = new_line
            new_I = count_incidences(current_points, current_lines)
            
            if new_I > current_I:
                current_I = new_I
                improved = True
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                break
            else:
                current_lines[worst_line_idx] = old_line
        
        # Try to improve worst point
        worst_pt_idx = np.argmin(point_counts)
        if not improved:
            for _ in range(15):
                # Find a better replacement point
                L2 = current_lines.astype(np.int64)
                cand = candidate_points
                px_c = cand[:, 0].reshape(-1, 1)
                py_c = cand[:, 1].reshape(-1, 1)
                A2 = L2[:, 0].reshape(1, -1)
                B2 = L2[:, 1].reshape(1, -1)
                C2 = L2[:, 2].reshape(1, -1)
                vals_c = A2 * px_c + B2 * py_c + C2
                cand_counts = np.sum(vals_c == 0, axis=1)
                
                current_set = set(map(tuple, current_points))
                available = [i for i in range(len(cand)) if tuple(cand[i]) not in current_set]
                
                if not available:
                    break
                
                avail_arr = np.array(available)
                avail_counts = cand_counts[avail_arr]
                top_avail = avail_arr[np.argsort(avail_counts)[::-1][:8]]
                
                new_pt_idx = random.choice(top_avail)
                new_point = cand[new_pt_idx]
                
                old_point = current_points[worst_pt_idx].copy()
                current_points[worst_pt_idx] = new_point
                new_I = count_incidences(current_points, current_lines)
                
                if new_I > current_I:
                    current_I = new_I
                    if current_I > best_I:
                        best_I = current_I
                        best_lines = current_lines.copy()
                        best_points = current_points.copy()
                    improved = True
                    break
                else:
                    current_points[worst_pt_idx] = old_point
        
        if not improved:
            # Try re-optimizing all points
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            if new_I > current_I:
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
            else:
                break
    
    return best_lines, best_points, best_I


def szemeredi_trotter_incidences():
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    # Generate candidate points with moderate grid
    candidate_points = generate_candidate_points(size=30)
    
    # Generate dense line pool
    line_pool = generate_dense_line_pool(max_coeff=10, max_C=35)
    
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    # Strategy 1: Grid-based initialization
    lines_grid = generate_grid_based_lines()
    points_grid, _ = get_best_points_for_lines(lines_grid, candidate_points)
    I_grid = count_incidences(points_grid, lines_grid)
    
    if I_grid > best_I_global:
        best_I_global = I_grid
        best_points_global = points_grid.copy()
        best_lines_global = lines_grid.copy()
    
    # Strategy 2: Pure grid points with greedy lines
    points_pure_grid = generate_grid_points()
    lines_from_points = greedy_select_lines_from_points(points_pure_grid, 27, line_pool)
    I_pure = count_incidences(points_pure_grid, lines_from_points)
    
    if I_pure > best_I_global:
        best_I_global = I_pure
        best_points_global = points_pure_grid.copy()
        best_lines_global = lines_from_points.copy()
    
    # Strategy 3: 4-direction grid (V/H/D1/D2)
    mixed_lines = []
    for c in range(-3, 4):  # 7 vertical
        mixed_lines.append((1, 0, -c))
    for c in range(-3, 4):  # 7 horizontal
        mixed_lines.append((0, 1, -c))
    for c in range(-3, 4):  # 7 diagonal (x-y=c)
        mixed_lines.append((1, -1, -c))
    for c in range(-2, 3):  # 5 anti-diagonal (x+y=c)
        mixed_lines.append((1, 1, -c))
    for c in range(-1, 2):  # 3 more (2x-y=c)
        mixed_lines.append((2, -1, -c))
    
    mixed_lines_arr = np.array(mixed_lines[:27], dtype=np.int64)
    points_mixed, _ = get_best_points_for_lines(mixed_lines_arr, candidate_points)
    I_mixed = count_incidences(points_mixed, mixed_lines_arr)
    
    if I_mixed > best_I_global:
        best_I_global = I_mixed
        best_points_global = points_mixed.copy()
        best_lines_global = mixed_lines_arr.copy()
    
    # Run joint optimization from each starting point
    starts = [
        (lines_grid, points_grid, 100),
        (lines_from_points, points_pure_grid, 200),
        (mixed_lines_arr, points_mixed, 300),
    ]
    
    for start_lines, start_points, seed in starts:
        if time.time() < deadline - 40:
            remaining = min(120, deadline - time.time() - 40)
            if remaining > 30:
                bl, bp, bi = joint_optimize(start_lines, start_points, candidate_points,
                                           line_pool, time_limit=remaining, seed=seed)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Validation and cleanup
    if best_points_global is None or best_lines_global is None:
        lines_fallback = generate_grid_based_lines()
        points_fallback, _ = get_best_points_for_lines(lines_fallback, candidate_points)
        best_points_global = points_fallback
        best_lines_global = lines_fallback
    
    # Ensure 54 unique points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        lines_fallback = generate_grid_based_lines()
        points_fallback, _ = get_best_points_for_lines(lines_fallback, candidate_points)
        best_points_global = points_fallback
        best_lines_global = lines_fallback
    else:
        best_points_global = unique_points[indices]
    
    # Ensure 27 unique lines
    unique_lines = []
    for line in best_lines_global:
        n_line = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
    
    if len(unique_lines) < 27:
        lines_fallback = generate_grid_based_lines()
        points_fallback, _ = get_best_points_for_lines(lines_fallback, candidate_points)
        best_points_global = points_fallback
        best_lines_global = lines_fallback
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)
    
    # Final polish
    if time.time() < deadline - 15:
        polish_time = min(50, deadline - time.time() - 15)
        if polish_time > 10:
            pl, pp, pi = polish_configuration(best_points_global, best_lines_global,
                                              candidate_points, polish_time, line_pool)
            if pi > best_I_global:
                best_I_global = pi
                best_points_global = pp.copy()
                best_lines_global = pl.copy()
    
    # Final validation
    best_points_global = best_points_global.astype(np.int64)
    best_lines_global = best_lines_global.astype(np.int64)
    
    return best_points_global, best_lines_global
