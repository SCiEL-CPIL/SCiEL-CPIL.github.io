import numpy as np
import time

N = 21

def _local_search(W, H, time_budget):
    start_time = time.time()
    
    # Initialization: Hexagonal packing
    points = []
    r0 = 0.05
    dx = 2 * r0
    dy = 2 * r0 * np.sqrt(3) / 2
    
    y = r0
    row = 0
    while y < H - r0:
        x_offset = 0 if row % 2 == 0 else r0
        x = r0 + x_offset
        while x < W - r0:
            points.append((x, y, r0))
            x += dx
        y += dy
        row += 1
        
    if len(points) > N:
        points = points[:N]
    elif len(points) < N:
        while len(points) < N:
            points.append((np.random.uniform(r0, W-r0), np.random.uniform(r0, H-r0), r0))
            
    circles = np.array(points)
    best_score = 0
    best_circles = circles.copy()
    
    # Main loop
    while time.time() - start_time < time_budget:
        # 1. Relaxation
        for _ in range(30):
            diffs = circles[:, np.newaxis, :] - circles[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            sum_radii = circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]
            overlaps = dists < sum_radii
            np.fill_diagonal(overlaps, False)
            
            safe_dists = np.where(dists == 0, 1e-6, dists)
            unit_vecs = diffs / safe_dists[:, :, np.newaxis]
            overlap_depth = np.maximum(0, sum_radii - dists)
            weights = np.where(overlaps, overlap_depth, 0)
            disp = np.sum(weights[:, :, np.newaxis] * unit_vecs, axis=1)
            
            bx = np.where(circles[:, 0] < circles[:, 2], circles[:, 2] - circles[:, 0], 0)
            bx -= np.where(circles[:, 0] > W - circles[:, 2], circles[:, 0] - (W - circles[:, 2]), 0)
            by = np.where(circles[:, 1] < circles[:, 2], circles[:, 2] - circles[:, 1], 0)
            by -= np.where(circles[:, 1] > H - circles[:, 2], circles[:, 1] - (H - circles[:, 2]), 0)
            
            step_size = 0.5
            circles[:, 0] += disp[:, 0] * step_size + bx * step_size
            circles[:, 1] += disp[:, 1] * step_size + by * step_size
            
            circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], W - circles[:, 2])
            circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], H - circles[:, 2])
            
            if np.max(np.abs(disp)) < 1e-5:
                break
        
        # 2. Expand
        diffs = circles[:, np.newaxis, :] - circles[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        sum_radii = circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]
        gaps = dists - sum_radii
        np.fill_diagonal(gaps, np.inf)
        
        gap_left = circles[:, 0] - circles[:, 2]
        gap_right = (W - circles[:, 0]) - circles[:, 2]
        gap_bottom = circles[:, 1] - circles[:, 2]
        gap_top = (H - circles[:, 1]) - circles[:, 2]
        
        min_gap = np.min(np.concatenate([gaps.flatten(), gap_left, gap_right, gap_bottom, gap_top]))
        
        if min_gap > 1e-6:
            # Adaptive step: take half the min gap to be safe
            delta_r = min_gap * 0.5
            circles[:, 2] += delta_r
            
        # 3. Perturbation
        if np.random.rand() < 0.1:
            idx = np.random.randint(0, N)
            circles[idx, 0] += np.random.uniform(-0.05, 0.05)
            circles[idx, 1] += np.random.uniform(-0.05, 0.05)
            
        score = np.sum(circles[:, 2])
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
            
    # Final Polish: Smaller steps, more iterations
    time_remaining = time_budget - (time.time() - start_time)
    if time_remaining > 1:
        for _ in range(int(time_remaining * 10)): # Approx 10s of polish
            # Tighter relaxation
            for _ in range(50):
                diffs = circles[:, np.newaxis, :] - circles[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=2))
                sum_radii = circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]
                overlaps = dists < sum_radii
                np.fill_diagonal(overlaps, False)
                
                safe_dists = np.where(dists == 0, 1e-6, dists)
                unit_vecs = diffs / safe_dists[:, :, np.newaxis]
                overlap_depth = np.maximum(0, sum_radii - dists)
                weights = np.where(overlaps, overlap_depth, 0)
                disp = np.sum(weights[:, :, np.newaxis] * unit_vecs, axis=1)
                
                bx = np.where(circles[:, 0] < circles[:, 2], circles[:, 2] - circles[:, 0], 0)
                bx -= np.where(circles[:, 0] > W - circles[:, 2], circles[:, 0] - (W - circles[:, 2]), 0)
                by = np.where(circles[:, 1] < circles[:, 2], circles[:, 2] - circles[:, 1], 0)
                by -= np.where(circles[:, 1] > H - circles[:, 2], circles[:, 1] - (H - circles[:, 2]), 0)
                
                step_size = 0.1
                circles[:, 0] += disp[:, 0] * step_size + bx * step_size
                circles[:, 1] += disp[:, 1] * step_size + by * step_size
                
                circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], W - circles[:, 2])
                circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], H - circles[:, 2])
                
                if np.max(np.abs(disp)) < 1e-7:
                    break
                    
            # Tiny expansion
            diffs = circles[:, np.newaxis, :] - circles[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            sum_radii = circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]
            gaps = dists - sum_radii
            np.fill_diagonal(gaps, np.inf)
            
            gap_left = circles[:, 0] - circles[:, 2]
            gap_right = (W - circles[:, 0]) - circles[:, 2]
            gap_bottom = circles[:, 1] - circles[:, 2]
            gap_top = (H - circles[:, 1]) - circles[:, 2]
            
            min_gap = np.min(np.concatenate([gaps.flatten(), gap_left, gap_right, gap_bottom, gap_top]))
            
            if min_gap > 1e-8:
                circles[:, 2] += min_gap * 0.5
                
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                
    return best_circles

def construct():
    start_time = time.time()
    total_time = 115.0
    aspects = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15]
    
    best_overall = None
    best_score_overall = 0
    
    for aspect in aspects:
        H = 2.0 / (1.0 + aspect)
        W = 2.0 * aspect / (1.0 + aspect)
        
        time_remaining = total_time - (time.time() - start_time)
        if time_remaining < 15:
            break
            
        budget = min(15.0, time_remaining)
        
        res = _local_search(W, H, budget)
        score = np.sum(res[:, 2])
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = res
            
        if time.time() - start_time > total_time - 5:
            break
            
    if best_overall is None:
        return np.zeros((N, 3))
        
    return best_overall
