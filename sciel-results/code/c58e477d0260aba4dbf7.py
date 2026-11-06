import time
import random
import numpy as np

def _fast_score(points):
    """
    Calculate the maximum multiplicity of a positive doubled triangle area.
    points: list of 30 tuples (x, y)
    """
    P = np.array(points, dtype=np.int64)
    # Max possible doubled area is around 5400, allocate 6000
    counts = np.zeros(6000, dtype=np.int64)
    
    for i in range(30):
        pi = P[i]
        rest = P[i+1:]
        n_rest = len(rest)
        if n_rest < 2:
            continue
        # Vectors from i to all other points
        V = rest - pi # Shape (n_rest, 2)
        
        for j in range(n_rest - 1):
            vj = V[j]
            vk = V[j+1:]
            # Cross product: vj[0]*vk[1] - vj[1]*vk[0]
            cross = vj[0] * vk[1] - vj[1] * vk[0]
            areas = np.abs(cross)
            
            # Filter zero areas
            mask = areas > 0
            if np.any(mask):
                valid_areas = areas[mask]
                if len(valid_areas) > 0:
                    max_val = np.max(valid_areas)
                    if max_val < len(counts):
                        # np.bincount returns array of length max_val + 1
                        bc = np.bincount(valid_areas)
                        # Add to global counts
                        counts[:len(bc)] += bc[:len(counts)]
                        
    if len(counts) == 0:
        return 0
    return int(np.max(counts))

def _sa_search(initial_points, time_limit=5.0):
    """
    Simulated Annealing to improve the configuration.
    """
    start_time = time.time()
    points = [tuple(p) for p in initial_points]
    current_score = _fast_score(points)
    best_points = [p for p in points]
    best_score = current_score
    
    T = 10.0
    cooling = 0.995
    min_T = 0.1
    
    occupied = set(points)
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = points[idx]
        
        # Pick a random new location
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        new_p = (new_x, new_y)
        
        if new_p == old_p:
            continue
        if new_p in occupied:
            continue
            
        # Apply move
        points[idx] = new_p
        occupied.remove(old_p)
        occupied.add(new_p)
        
        new_score = _fast_score(points)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p for p in points]
        else:
            # Reject move
            points[idx] = old_p
            occupied.remove(new_p)
            occupied.add(old_p)
            
        T *= cooling
        if T < min_T:
            T = min_T
            
    return best_points, best_score

def construct():
    start_time = time.time()
    hard_deadline = start_time + 55.0 # Leave 5s buffer
    
    random.seed(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. 5x6 Grid centered at 0
    g1 = [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    candidates.append(g1)
    
    # 2. 6x5 Grid
    g2 = [(x, y) for x in range(-3, 3) for y in range(-2, 3)]
    candidates.append(g2)
    
    # 3. 3x10 Grid
    g3 = [(x, y) for x in range(-1, 2) for y in range(-5, 5)]
    candidates.append(g3)
    
    # 4. 10x3 Grid
    g4 = [(x, y) for x in range(-5, 5) for y in range(-1, 2)]
    candidates.append(g4)
    
    # 5. Random points
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    g5 = random.sample(all_coords, 30)
    candidates.append(g5)
    
    # 6. Perturbed Grid (5x6)
    g6_base = [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    g6 = []
    for p in g6_base:
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        new_p = (p[0] + dx, p[1] + dy)
        if -30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30:
            g6.append(new_p)
        else:
            g6.append(p)
    if len(set(g6)) != 30:
        g6 = g1
    candidates.append(g6)
    
    best_points = None
    best_score = -1
    
    # Evaluate initial candidates
    scored_candidates = []
    for c in candidates:
        if len(c) != 30 or len(set(c)) != 30:
            continue
        s = _fast_score(c)
        scored_candidates.append((s, c))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Run SA on top candidates
    # Time allocation:
    # 1st: 20s
    # 2nd: 15s
    # 3rd: 10s
    # 4th: 10s
    # Total 55s
    
    time_alloc = [20.0, 15.0, 10.0, 10.0]
    
    for i in range(min(4, len(scored_candidates))):
        if time.time() > hard_deadline - 2.0:
            break
            
        s, c = scored_candidates[i]
        if s > best_score:
            best_score = s
            best_points = [tuple(p) for p in c]
            
        t_limit = time_alloc[i] if i < len(time_alloc) else 5.0
        # Ensure we don't exceed deadline
        remaining = hard_deadline - time.time() - 1.0
        if remaining < 2.0:
            break
            
        t_limit = min(t_limit, remaining)
        
        if t_limit > 1.0:
            fp, fs = _sa_search(c, time_limit=t_limit)
            if fs > best_score:
                best_score = fs
                best_points = fp
                
    if best_points is None:
        best_points = candidates[0]
        
    return [tuple(p) for p in best_points]
