import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Initial hexagonal packing
    row_counts = [7, 6, 7, 6, 7, 6, 7, 4]
    d = 1/7
    v = d * np.sqrt(3) / 2
    
    centers = []
    for i, n in enumerate(row_counts):
        y = -0.5 + v/2 + i * v
        x_start = -d * (n-1) / 2
        for j in range(n):
            x = x_start + j * d
            if i % 2 == 1:
                x += d/2
            centers.append([x, y])
    
    centers = np.array(centers, dtype=np.float64)
    
    # Local optimization using gradient of the min distance
    best_r = compute_radius(centers)
    best_centers = centers.copy()
    
    lr = 0.005
    momentum = 0.5
    velocity = np.zeros_like(centers)
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute gradient of the min distance
        grad = np.zeros_like(centers)
        
        # Pairwise distances
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Gradient of dist w.r.t. centers[i] and centers[j]
                grad[i] += diff / dist
                grad[j] -= diff / dist
        
        # Boundary distances
        for i in range(N):
            # Left boundary
            d_left = 0.5 + centers[i, 0]
            grad[i, 0] += 1.0  # push right
            # Right boundary
            d_right = 0.5 - centers[i, 0]
            grad[i, 0] -= 1.0  # push left (if close)
            # Bottom
            d_bottom = 0.5 + centers[i, 1]
            grad[i, 1] += 1.0
            # Top
            d_top = 0.5 - centers[i, 1]
            grad[i, 1] -= 1.0
        
        # This gradient approach is wrong. Let me think again.
        
        # Actually, for the min of distances, the gradient is only from the minimum distance.
        # Let me use a different approach: identify the critical pairs and push them apart.
        
        # Find the minimum pairwise distance
        min_dist = np.inf
        min_i, min_j = -1, -1
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
                    min_i, min_j = i, j
        
        # Find the minimum boundary distance
        min_bdist = np.inf
        min_bi = -1
        min_bside = 0  # 0: left, 1: right, 2: bottom, 3: top
        for i in range(N):
            dl = 0.5 + centers[i, 0]
            dr = 0.5 - centers[i, 0]
            db = 0.5 + centers[i, 1]
            dt = 0.5 - centers[i, 1]
            if dl < min_bdist:
                min_bdist = dl
                min_bi = i
                min_bside = 0
            if dr < min_bdist:
                min_bdist = dr
                min_bi = i
                min_bside = 1
            if db < min_bdist:
                min_bdist = db
                min_bi = i
                min_bside = 2
            if dt < min_bdist:
                min_bdist = dt
                min_bi = i
                min_bside = 3
        
        # Determine which is the binding constraint
        if min_dist / 2 < min_bdist:
            # Pairwise constraint is binding
            diff = centers[min_i] - centers[min_j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                direction = diff / norm
                centers[min_i] += lr * direction
                centers[min_j] -= lr * direction
        else:
            # Boundary constraint is binding
            if min_bside == 0:  # left
                centers[min_bi, 0] += lr
            elif min_bside == 1:  # right
                centers[min_bi, 0] -= lr
            elif min_bside == 2:  # bottom
                centers[min_bi, 1] += lr
            else:  # top
                centers[min_bi, 1] -= lr
        
        # Clamp to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Update best
        r = compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers

def compute_radius(centers):
    min_pair = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            min_pair = min(min_pair, d)
    
    min_boundary = np.min([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1]
    ])
    
    return min(min_pair / 2, min_boundary)
