import time
import math
import random

# Pre-define neighbors for fast edge counting
# Triangular lattice neighbors for (a, b) are:
# (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def _hexdist(a, b):
    """Calculate hexagonal distance from origin (0,0)"""
    return (abs(a) + abs(b) + abs(a + b)) // 2

def _count_edges(points):
    """Count the number of pairs at distance exactly 1"""
    point_set = set(points)
    count = 0
    for p in points:
        a, b = p
        for da, db in _NEIGHBORS:
            if (a + da, b + db) in point_set:
                count += 1
    return count // 2  # Each edge is counted twice

def _generate_hex_spiral(n, offset=0.0):
    """
    Generate optimal hexagonal cluster by adding points in order of
    increasing hex distance, breaking ties by angle with given offset.
    For n=32, this gives 1+6+12+13=32 points.
    """
    max_radius = 5  # Enough for n=32
    candidates = []
    for a in range(-max_radius, max_radius + 1):
        for b in range(-max_radius, max_radius + 1):
            d = _hexdist(a, b)
            if d <= max_radius:
                x = a + 0.5 * b
                y = math.sqrt(3) / 2 * b
                angle = math.atan2(y, x) + offset
                if angle < 0:
                    angle += 2 * math.pi
                candidates.append((d, angle, a, b))
    
    candidates.sort(key=lambda x: (x[0], x[1]))
    result = [(a, b) for _, _, a, b in candidates[:n]]
    return result

def _local_search(points, max_time=5.0):
    """
    Perform local search to maximize edge count using 1-swap moves.
    """
    start_time = time.time()
    current_points = set(points)
    current_edges = _count_edges(current_points)
    best_points = set(current_points)
    best_edges = current_edges
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Bounding box of current set
        min_a = min(p[0] for p in current_points)
        max_a = max(p[0] for p in current_points)
        min_b = min(p[1] for p in current_points)
        max_b = max(p[1] for p in current_points)
        
        # Expand bounding box by 2 to find candidates
        cand_min_a = min_a - 2
        cand_max_a = max_a + 2
        cand_min_b = min_b - 2
        cand_max_b = max_b + 2
        
        candidates = set()
        for a in range(cand_min_a, cand_max_a + 1):
            for b in range(cand_min_b, cand_max_b + 1):
                candidates.add((a, b))
        
        candidates -= current_points
        
        if not candidates:
            break
            
        # Try swapping each point in current_points with each candidate
        for p_out in list(current_points):
            # Quick check: how many edges does p_out have?
            lost = 0
            for da, db in _NEIGHBORS:
                if (p_out[0] + da, p_out[1] + db) in current_points:
                    lost += 1
            
            for p_in in candidates:
                # Quick check: how many edges would p_in have?
                gained = 0
                for da, db in _NEIGHBORS:
                    nb = (p_in[0] + da, p_in[1] + db)
                    if nb in current_points and nb != p_out:
                        gained += 1
                
                delta = gained - lost
                if delta > 0:
                    # Accept the move
                    current_points.remove(p_out)
                    current_points.add(p_in)
                    current_edges += delta
                    improved = True
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_points = set(current_points)
                    
                    if best_edges >= 76:
                        return list(best_points)
                    
                    # Restart candidate generation
                    break
            
            if improved:
                break
        
        if not improved:
            # No single swap improves. Try a random perturbation.
            if random.random() < 0.15:
                p_out = random.choice(list(current_points))
                p_in = random.choice(list(candidates))
                current_points.remove(p_out)
                current_points.add(p_in)
                current_edges = _count_edges(current_points)
            else:
                break
    
    return list(best_points)

def _simulated_annealing(points, max_time=10.0):
    """
    Perform simulated annealing to maximize edge count.
    """
    start_time = time.time()
    rng = random.Random(42)
    current_points = list(points)
    current_set = set(current_points)
    current_edges = _count_edges(current_points)
    best_points = list(current_points)
    best_edges = current_edges
    
    # Build pool of candidate positions
    min_a = min(p[0] for p in current_points)
    max_a = max(p[0] for p in current_points)
    min_b = min(p[1] for p in current_points)
    max_b = max(p[1] for p in current_points)
    
    pool = set()
    for a in range(min_a - 3, max_a + 4):
        for b in range(min_b - 3, max_b + 4):
            pool.add((a, b))
    pool -= current_set
    pool = list(pool)
    
    T = 1.0
    cooling_rate = 0.95
    
    while time.time() - start_time < max_time and T > 0.01:
        for _ in range(500):
            if time.time() - start_time >= max_time:
                break
            
            i = rng.randrange(32)
            p_i = current_points[i]
            
            if not pool:
                break
            
            c_idx = rng.randrange(len(pool))
            p_c = pool[c_idx]
            
            if p_c == p_i or p_c in current_set:
                continue
            
            lost_edges = 0
            for da, db in _NEIGHBORS:
                if (p_i[0] + da, p_i[1] + db) in current_set:
                    lost_edges += 1
            
            gained_edges = 0
            for da, db in _NEIGHBORS:
                nb = (p_c[0] + da, p_c[1] + db)
                if nb in current_set and nb != p_i:
                    gained_edges += 1
            
            delta = gained_edges - lost_edges
            new_edges = current_edges + delta
            
            if new_edges > current_edges:
                current_points[i] = p_c
                current_set.discard(p_i)
                current_set.add(p_c)
                current_edges = new_edges
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = current_points[:]
                    
                if best_edges >= 76:
                    return best_points
            else:
                if delta < 0:
                    prob = math.exp(delta / T)
                    if rng.random() < prob:
                        current_points[i] = p_c
                        current_set.discard(p_i)
                        current_set.add(p_c)
                        current_edges = new_edges
        
        T *= cooling_rate
    
    return best_points

def construct():
    n = 32
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    best_points = []
    best_edges = 0
    
    # Phase 1: Try hex spirals with various angle offsets
    offsets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5]
    
    for offset in offsets:
        if time.time() > deadline - 20:
            break
        
        points = _generate_hex_spiral(n, offset)
        edges = _count_edges(points)
        
        if edges > best_edges:
            best_edges = edges
            best_points = points
        
        if edges >= 76:
            return [[a, b] for a, b in best_points]
    
    # Phase 2: Local search from best spiral
    if best_edges < 76 and time.time() < deadline - 15:
        search_points = _local_search(best_points, max_time=10.0)
        search_edges = _count_edges(search_points)
        
        if search_edges > best_edges:
            best_edges = search_edges
            best_points = search_points
        
        if best_edges >= 76:
            return [[a, b] for a, b in best_points]
    
    # Phase 3: Simulated annealing from best result
    if best_edges < 76 and time.time() < deadline - 5:
        sa_points = _simulated_annealing(best_points, max_time=15.0)
        sa_edges = _count_edges(sa_points)
        
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_points = sa_points
    
    # Phase 4: If still not optimal, try multiple random restarts with local search
    if best_edges < 76 and time.time() < deadline - 5:
        rng = random.Random(123)
        
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Generate a random initial configuration
            pts_set = set()
            attempts = 0
            while len(pts_set) < n and attempts < 1000:
                a = rng.randint(-6, 6)
                b = rng.randint(-6, 6)
                pts_set.add((a, b))
                attempts += 1
            
            if len(pts_set) < n:
                continue
            
            pts_list = list(pts_set)
            edges = _count_edges(pts_list)
            
            if edges > best_edges:
                best_edges = edges
                best_points = pts_list
            
            if best_edges >= 76:
                return [[a, b] for a, b in best_points]
            
            # Run local search from this random start
            if time.time() < deadline - 5:
                search_points = _local_search(pts_list, max_time=3.0)
                search_edges = _count_edges(search_points)
                
                if search_edges > best_edges:
                    best_edges = search_edges
                    best_points = search_points
                
                if best_edges >= 76:
                    return [[a, b] for a, b in best_points]
    
    return [[a, b] for a, b in best_points]
