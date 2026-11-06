import time
import random
import math
from collections import deque

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _dist_sq(p, q):
    da = p[0] - q[0]
    db = p[1] - q[1]
    return da * da + da * db + db * db

def _edge_count(Sset):
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
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

def _spiral_config():
    """
    Generate a spiral configuration on the triangular lattice.
    The triangular lattice spiral (or hexagonal spiral) places points
    in order of increasing hex distance, going around each ring.
    For n=32, we take the first 32 points in spiral order.
    """
    points = []
    # Center
    points.append((0, 0))
    
    # Generate spiral rings
    # Ring r has 6*r points
    # We need 31 more points: ring 1 (6), ring 2 (12), ring 3 (18) -> total 36, take first 31
    # So we take all of ring 1 and 2 (18 points) and first 13 of ring 3
    
    # Directions for the spiral
    # In a hexagonal spiral, we move in 6 directions per ring
    # Let's use the standard approach: for each ring, traverse the 6 sides
    
    for ring in range(1, 10):
        # Start at (ring, 0)
        # Move in 6 directions
        # Direction vectors for the 6 sides of the hexagon
        dirs = [(0, -1), (-1, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
        # Actually, let me use a cleaner spiral generation
        
        # Standard hex spiral: start at (ring, 0)
        # Side 1: move (0, -1) for 'ring' steps
        # Side 2: move (-1, 0) for 'ring' steps
        # Side 3: move (0, 1) for 'ring' steps
        # Side 4: move (1, 0) for 'ring' steps
        # Side 5: move (0, -1) for 'ring' steps
        # Side 6: move (-1, 0) for 'ring' steps
        
        pos = (ring, 0)
        steps_per_side = ring
        side_dirs = [
            (0, -1),   # down
            (-1, 0),   # left
            (0, 1),    # up
            (1, 0),    # right
            (0, -1),   # down
            (-1, 0),   # left
        ]
        
        for side in range(6):
            for step in range(steps_per_side):
                if len(points) >= 32:
                    break
                da, db = side_dirs[side]
                pos = (pos[0] + da, pos[1] + db)
                if pos not in points:
                    points.append(pos)
            if len(points) >= 32:
                break
        if len(points) >= 32:
            break
    
    return points[:32]

def _hexagonal_cluster():
    """
    Generate a hexagonal cluster of 32 points.
    Take hex distance <= 2 (19 points) and add 13 from hex distance 3.
    """
    points = set()
    
    # BFS to find all points at hex distance <= 2
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
    
    for p in visited:
        points.add(p)
    
    # Find points at hex distance 3
    ring3 = []
    for a in range(-5, 6):
        for b in range(-5, 6):
            if (a, b) not in points:
                # Check if it's at hex distance 3 by BFS
                if (a, b) in dist and dist[(a, b)] == 3:
                    ring3.append((a, b))
    
    # Sort by number of neighbors in points (to keep it compact)
    ring3.sort(key=lambda p: -sum(1 for nb in _neighbors(p) if nb in points))
    
    for p in ring3[:13]:
        points.add(p)
    
    return list(points)

def _edge_count_fast(Sset):
    """Faster edge count using a set for O(1) lookups."""
    S = set(Sset)
    cnt = 0
    for p in S:
        for q in _neighbors(p):
            if q in S and p < q:
                cnt += 1
    return cnt

def _local_search(Sset, max_iters=5000, deadline=None, seed=42):
    """
    Local search: try moving points to improve edge count.
    Uses a list for the set and a dict for O(1) membership.
    """
    S = list(Sset)
    S_set = set(S)
    current_edges = _edge_count_fast(S_set)
    best_edges = current_edges
    best_S = list(S)
    best_S_set = set(S)
    
    # Candidate positions
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            candidates.append((a, b))
    
    rng = random.Random(seed)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = rng.randrange(len(S))
        old_pos = S[idx]
        
        # Try neighbors and some random positions
        move_candidates = list(_neighbors(old_pos))
        rng.shuffle(move_candidates)
        for _ in range(5):
            move_candidates.append(rng.choice(candidates))
        
        improved = False
        for new_pos in move_candidates:
            if new_pos == old_pos or new_pos in S_set:
                continue
            
            # Check validity
            valid = True
            for other in S_set:
                if other == old_pos:
                    continue
                if _dist_sq(new_pos, other) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute delta in edge count
            # Remove old_pos, add new_pos
            old_degree = sum(1 for nb in _neighbors(old_pos) if nb in S_set)
            new_degree = sum(1 for nb in _neighbors(new_pos) if nb in S_set and nb != old_pos)
            
            # The edges lost are those incident to old_pos
            # The edges gained are those incident to new_pos (excluding old_pos if adjacent)
            delta = new_degree - old_degree
            
            if delta > 0 or (delta == 0 and rng.random() < 0.3):
                # Accept the move
                S_set.remove(old_pos)
                S_set.add(new_pos)
                S[idx] = new_pos
                current_edges += delta
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_S = list(S)
                    best_S_set = set(S)
                improved = True
                break
            # else: reject, try next candidate
        
    return list(best_S)

def _simulated_annealing(Sset, deadline=None, seed=42):
    """
    Simulated annealing to escape local optima.
    """
    S = list(Sset)
    S_set = set(S)
    current_edges = _edge_count_fast(S_set)
    best = list(S)
    best_edges = current_edges
    
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            candidates.append((a, b))
    
    rng = random.Random(seed)
    
    T = 2.0
    T_min = 0.05
    cooling = 0.997
    
    while T > T_min:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point and a random new position
        idx = rng.randrange(len(S))
        old_pos = S[idx]
        
        # Try nearby positions first
        if rng.random() < 0.7:
            new_pos = rng.choice(list(_neighbors(old_pos)))
        else:
            new_pos = rng.choice(candidates)
        
        if new_pos == old_pos or new_pos in S_set:
            T *= cooling
            continue
        
        # Check validity
        valid = True
        for other in S_set:
            if other == old_pos:
                continue
            if _dist_sq(new_pos, other) < 1:
                valid = False
                break
        if not valid:
            T *= cooling
            continue
        
        # Compute delta
        old_degree = sum(1 for nb in _neighbors(old_pos) if nb in S_set)
        new_degree = sum(1 for nb in _neighbors(new_pos) if nb in S_set and nb != old_pos)
        delta = new_degree - old_degree
        
        if delta > 0 or rng.random() < math.exp(delta / T):
            S_set.remove(old_pos)
            S_set.add(new_pos)
            S[idx] = new_pos
            current_edges += delta
            
            if current_edges > best_edges:
                best_edges = current_edges
                best = list(S)
        else:
            pass  # Reject
        
        T *= cooling
    
    return best

def _greedy_construct(seed):
    """
    Greedily construct a set of 32 points.
    At each step, pick the position that maximizes the number of new edges.
    """
    rng = random.Random(seed)
    points = set()
    
    # Define a search region for candidates
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            candidates.append((a, b))
            
    # Shuffle candidates to break symmetry randomly
    rng.shuffle(candidates)
    
    for _ in range(32):
        best_edge_gain = -1
        best_pos = None
        
        for p in candidates:
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
            
            if gain > best_edge_gain:
                best_edge_gain = gain
                best_pos = p
        
        if best_pos is None:
            # Fallback: pick a random valid position
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
            break
            
    return list(points)

def _backtracking_search(deadline=None):
    """
    Backtracking search that tries to find a configuration with 76 edges.
    Uses pruning based on maximum possible additional edges.
    """
    best_result = None
    best_edges = -1
    
    # Start with a good initial configuration
    initial = _spiral_config()
    if len(initial) != 32:
        initial = _hexagonal_cluster()
    
    # Polish the initial configuration
    polished = _local_search(initial, max_iters=2000, deadline=deadline, seed=1)
    edges = _edge_count_fast(polished)
    if edges > best_edges:
        best_edges = edges
        best_result = polished
    
    if best_edges >= 76:
        return best_result
    
    # Try simulated annealing from the polished configuration
    if deadline is not None and time.time() < deadline - 10:
        sa_result = _simulated_annealing(polished, deadline=deadline - 5, seed=42)
        edges = _edge_count_fast(sa_result)
        if edges > best_edges:
            best_edges = edges
            best_result = sa_result
    
    if best_edges >= 76:
        return best_result
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_result = None
    best_edges = -1
    
    # Try spiral configuration
    spiral = _spiral_config()
    if len(spiral) == 32:
        polished = _local_search(spiral, max_iters=3000, deadline=deadline - 20, seed=1)
        edges = _edge_count_fast(polished)
        if edges > best_edges:
            best_edges = edges
            best_result = polished
        
        if best_edges >= 76:
            return [[a, b] for a, b in best_result]
        
        # Try SA from spiral
        if time.time() < deadline - 15:
            sa_result = _simulated_annealing(polished, deadline=deadline - 5, seed=42)
            edges = _edge_count_fast(sa_result)
            if edges > best_edges:
                best_edges = edges
                best_result = sa_result
    
    if best_edges >= 76:
        return [[a, b] for a, b in best_result]
    
    # Try hexagonal cluster
    hex_config = _hexagonal_cluster()
    if len(hex_config) == 32:
        polished = _local_search(hex_config, max_iters=3000, deadline=deadline - 20, seed=2)
        edges = _edge_count_fast(polished)
        if edges > best_edges:
            best_edges = edges
            best_result = polished
        
        if best_edges >= 76:
            return [[a, b] for a, b in best_result]
        
        if time.time() < deadline - 15:
            sa_result = _simulated_annealing(polished, deadline=deadline - 5, seed=43)
            edges = _edge_count_fast(sa_result)
            if edges > best_edges:
                best_edges = edges
                best_result = sa_result
    
    if best_edges >= 76:
        return [[a, b] for a, b in best_result]
    
    # Run multiple greedy + local search iterations
    num_iterations = 30
    
    for i in range(num_iterations):
        if time.time() > deadline - 5:
            break
            
        seed = i * 1000 + 42
        initial = _greedy_construct(seed)
        
        if len(initial) != 32:
            continue
            
        # Local search
        result = _local_search(initial, max_iters=1000, deadline=deadline - 2, seed=seed)
        
        edges = _edge_count_fast(result)
        
        if edges > best_edges:
            best_edges = edges
            best_result = result
            
        # If we hit the theoretical maximum, we can stop early
        if best_edges >= 76:
            break
    
    # If we haven't found 76, try more aggressive SA
    if best_edges < 76 and best_result is not None and time.time() < deadline - 10:
        sa_result = _simulated_annealing(best_result, deadline=deadline - 2, seed=99)
        edges = _edge_count_fast(sa_result)
        if edges > best_edges:
            best_edges = edges
            best_result = sa_result
    
    if best_result is None or len(best_result) != 32:
        # Fallback: simple grid
        best_result = [(a, b) for a in range(8) for b in range(4)]
    
    # Final verification
    if not _valid(best_result):
        best_result = [(a, b) for a in range(8) for b in range(4)]
    
    return [[a, b] for a, b in best_result]
