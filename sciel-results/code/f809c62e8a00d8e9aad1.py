import time
import random
import math

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _dist_sq(p, q):
    """Squared distance between two lattice points a+b*omega."""
    da = p[0] - q[0]
    db = p[1] - q[1]
    # |a + b*omega|^2 = a^2 + ab + b^2 for omega = exp(i*pi/3)
    return da * da + da * db + db * db

def _edge_count(Sset):
    """Count pairs at distance exactly 1."""
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    """Check validity: 32 distinct points, |a|,|b| <= 64, min distance >= 1."""
    if len(Sset) != 32:
        return False
    if len(set(Sset)) != 32:
        return False
    for a, b in Sset:
        if abs(a) > 64 or abs(b) > 64:
            return False
    pts = list(Sset)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if _dist_sq(pts[i], pts[j]) < 1:
                return False
    return True

def _hexagon_config():
    """
    Generate a hexagonal cluster of 32 points.
    Hexagonal numbers: 1, 7, 19, 37. So 32 is between 19 and 37.
    We take a hexagon of radius 2 (19 points) and add 13 more points around it.
    """
    points = set()
    # Center
    points.add((0, 0))
    # Ring 1: 6 points
    for i in range(6):
        a, b = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)][i]
        points.add((a, b))
    # Ring 2: 12 points
    for i in range(12):
        # Hexagonal ring 2
        if i < 2:
            a, b = (2 - i, i)
        elif i < 4:
            a, b = (0, 2) if i == 2 else (-1, 2) if i == 3 else (1, 1)
            # Let me use a different approach
        # Actually let me just enumerate all points with hex distance <= 2
        pass
    
    # Better approach: enumerate all lattice points with hex distance <= 2
    # Hex distance from origin: max(|a|, |b|, |a+b|) in axial coords... 
    # Actually for our basis, distance from origin is sqrt(a^2 + ab + b^2)
    # Hex distance (cubed) is different. Let me just use the standard hex ring.
    
    # Standard hexagonal lattice: point (a,b) has neighbors at the 6 directions.
    # Ring r consists of points at hex-distance r.
    # For our coordinate system, hex-distance from (0,0) to (a,b):
    # We can compute it by BFS or use the formula.
    # Actually, let me just enumerate all points within a bounding box and filter.
    
    points = set()
    for a in range(-4, 5):
        for b in range(-4, 5):
            # Check if this point is "inside" a hexagon of radius 2
            # Using the fact that hex rings grow as 1, 6, 12, 18, ...
            # Ring 0: 1 point
            # Ring 1: 6 points  
            # Ring 2: 12 points
            # Total up to ring 2: 19 points
            # We need 32, so we need ring 3 partially.
            # Let me just use a simpler approach: take all points with 
            # a^2 + ab + b^2 <= some threshold, then add more.
            pass
    
    # Simplest: just generate a compact cluster manually
    # A hexagonal shape with 32 points
    points = set()
    # Row -2: 3 points
    for a in range(-1, 2):
        points.add((a, -2))
    # Row -1: 5 points
    for a in range(-2, 3):
        points.add((a, -1))
    # Row 0: 7 points
    for a in range(-3, 4):
        points.add((a, 0))
    # Row 1: 5 points
    for a in range(-2, 3):
        points.add((a, 1))
    # Row 2: 3 points
    for a in range(-1, 2):
        points.add((a, 2))
    # That's 3+5+7+5+3 = 23 points. Need 9 more.
    # Add row -3: 1 point (center)
    points.add((0, -3))
    # Add row 3: 1 point
    points.add((0, 3))
    # Need 7 more. Add to sides.
    # Let me just use a different configuration.
    
    # Actually, let me use the spiral approach mentioned in the task.
    # Or just use a simple dense cluster.
    
    # Let me generate a hexagonal cluster properly.
    # In a triangular lattice, a hexagonal cluster of "radius" r has:
    # r=0: 1 point
    # r=1: 7 points
    # r=2: 19 points
    # r=3: 37 points
    # So for 32 points, we take r=2 (19 points) and add 13 from r=3.
    
    points = set()
    # All points with hex distance <= 2
    # Hex distance in our coordinates: we need to define it properly.
    # For omega = exp(i*pi/3), the lattice is a triangular lattice.
    # The six neighbors of (0,0) are: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # These form a hexagon.
    
    # Let me just BFS to find all points at hex distance <= 2
    from collections import deque
    visited = set()
    queue = deque([(0, 0)])
    visited.add((0, 0))
    dist = {(0, 0): 0}
    while queue:
        a, b = queue.popleft()
        d = dist[(a, b)]
        if d >= 2:
            continue
        for nb in _neighbors((a, b)):
            if nb not in visited:
                visited.add(nb)
                dist[nb] = d + 1
                queue.append(nb)
    
    # Now we have all points at hex distance <= 2, which should be 19 points
    for p in visited:
        points.add(p)
    
    # We have 19 points. Need 13 more from hex distance 3.
    # Add points at hex distance 3, prioritizing those adjacent to existing points.
    ring3 = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if (a, b) not in points:
                # Check if it's at hex distance 3
                if (a, b) in dist and dist[(a, b)] == 3:
                    ring3.append((a, b))
    
    # Sort by number of neighbors in points (to keep it compact)
    ring3.sort(key=lambda p: -sum(1 for nb in _neighbors(p) if nb in points))
    
    for p in ring3[:13]:
        points.add(p)
    
    return list(points)

def _greedy_construct(seed):
    """
    Greedily construct a set of 32 points.
    At each step, pick the position that maximizes the number of new edges,
    with random tie-breaking.
    """
    rng = random.Random(seed)
    points = set()
    
    # Define a search region for candidates
    # A 15x15 region should be enough for 32 points
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            candidates.append((a, b))
            
    # Shuffle candidates to break symmetry randomly
    rng.shuffle(candidates)
    
    for _ in range(32):
        best_score = -1
        best_pos = None
        # To break ties randomly, we can add a small random noise
        # Or just iterate and pick the first max with a random start offset
        start_idx = rng.randrange(len(candidates))
        
        best_edge_gain = -1
        best_pos = None
        best_tie = 0
        
        for i in range(len(candidates)):
            idx = (start_idx + i) % len(candidates)
            p = candidates[idx]
            if p in points:
                continue
            
            # Check validity: distance >= 1 from all existing points
            valid = True
            for q in points:
                if _dist_sq(p, q) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Count new edges
            gain = 0
            for nb in _neighbors(p):
                if nb in points:
                    gain += 1
            
            # Tie-breaking: random noise to explore different shapes
            # We want to maximize gain, but if gain is same, pick randomly
            # This specific implementation picks the first one with max gain in the shuffled order
            # To make it more robust, we can collect all max gain candidates and pick one
            # But for speed, let's just use the gain and a small random factor
            # Actually, just picking the first max in a shuffled list is a good heuristic
            
            if gain > best_edge_gain:
                best_edge_gain = gain
                best_pos = p
                best_tie = rng.random()
            elif gain == best_edge_gain:
                # If gain is equal, we might want to pick randomly among them
                # But for simplicity and speed, we'll stick to the first one found
                # in the shuffled order, which is effectively random.
                pass
        
        if best_pos is None:
            # Fallback: pick a random valid position
            # This shouldn't happen with a large enough candidate pool
            for p in candidates:
                if p in points:
                    continue
                valid = True
                for q in points:
                    if _dist_sq(p, q) < 1:
                        valid = False
                        break
                if valid:
                    best_pos = p
                    break
        
        if best_pos is not None:
            points.add(best_pos)
        else:
            # Error: no valid position found
            break
            
    return list(points)

def _local_search(Sset, max_iters=2000, deadline=None):
    """
    Local search: try moving points to improve edge count.
    """
    current_edges = _edge_count(Sset)
    best_edges = current_edges
    best_set = set(Sset)
    
    # Candidate positions to try
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            candidates.append((a, b))
            
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = rng.randrange(len(Sset))
        old_pos = Sset[idx]
        
        # Try a few random candidate positions
        # We prioritize neighbors of the current position for locality
        move_candidates = list(_neighbors(old_pos))
        rng.shuffle(move_candidates)
        # Add some random distant candidates
        for _ in range(3):
            move_candidates.append(rng.choice(candidates))
            
        improved = False
        for new_pos in move_candidates:
            if new_pos == old_pos:
                continue
            if new_pos in Sset:
                continue
            
            # Check validity
            valid = True
            for other in Sset:
                if other == old_pos:
                    continue
                if _dist_sq(new_pos, other) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute new edge count
            Sset[idx] = new_pos
            new_edges = _edge_count(Sset)
            
            if new_edges > current_edges:
                current_edges = new_edges
                improved = True
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_set = set(Sset)
                break # Move accepted, next iteration
            else:
                Sset[idx] = old_pos
                # Do not break, try next candidate for this point
        
    return list(best_set)

def _simulated_annealing(Sset, deadline=None, seed=42):
    """
    Simulated annealing to escape local optima.
    """
    rng = random.Random(seed)
    current = list(Sset)
    current_edges = _edge_count(current)
    best = list(current)
    best_edges = current_edges
    
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            candidates.append((a, b))
    
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    while T > T_min:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point and a random new position
        idx = rng.randrange(len(current))
        old_pos = current[idx]
        
        # Try nearby positions first
        if rng.random() < 0.7:
            new_pos = rng.choice(list(_neighbors(old_pos)))
        else:
            new_pos = rng.choice(candidates)
        
        if new_pos == old_pos or new_pos in current:
            T *= cooling
            continue
        
        # Check validity
        valid = True
        for other in current:
            if other == old_pos:
                continue
            if _dist_sq(new_pos, other) < 1:
                valid = False
                break
        if not valid:
            T *= cooling
            continue
        
        # Compute delta
        current[idx] = new_pos
        new_edges = _edge_count(current)
        delta = new_edges - current_edges
        
        if delta > 0 or rng.random() < math.exp(delta / T):
            current_edges = new_edges
            if new_edges > best_edges:
                best_edges = new_edges
                best = list(current)
        else:
            current[idx] = old_pos
        
        T *= cooling
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_result = None
    best_edges = -1
    
    # First try the hexagonal configuration
    hex_config = _hexagon_config()
    if len(hex_config) == 32:
        # Polish with local search
        polished = _local_search(hex_config, max_iters=2000, deadline=deadline - 10)
        edges = _edge_count(polished)
        if edges > best_edges:
            best_edges = edges
            best_result = polished
        
        # Try simulated annealing
        if time.time() < deadline - 20:
            sa_result = _simulated_annealing(hex_config, deadline=deadline - 5, seed=42)
            edges = _edge_count(sa_result)
            if edges > best_edges:
                best_edges = edges
                best_result = sa_result
    
    # Run multiple greedy + local search iterations
    num_iterations = 50
    
    for i in range(num_iterations):
        if time.time() > deadline - 5:
            break
            
        seed = i * 1000 + 42
        initial = _greedy_construct(seed)
        
        if len(initial) != 32:
            continue
            
        # Local search
        result = _local_search(initial, max_iters=500, deadline=deadline - 2)
        
        edges = _edge_count(result)
        
        if edges > best_edges:
            best_edges = edges
            best_result = result
            
        # If we hit the theoretical maximum, we can stop early
        if best_edges >= 76:
            break
            
    if best_result is None or len(best_result) != 32:
        # Fallback: simple grid
        # This is a valid configuration but likely suboptimal
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    # Final verification
    if not _valid(best_result):
        # Try to repair or fallback
        # For safety, return a known valid config
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    return [[a, b] for a, b in best_result]
