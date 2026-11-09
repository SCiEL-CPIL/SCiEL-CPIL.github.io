import time
import random

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0

def construct():
    start_time = time.time()
    
    # Precompute neighbors and neighbor masks
    neighbors = [[] for _ in range(MAX_VERTS)]
    neighbor_masks = [0] * MAX_VERTS
    for v in range(MAX_VERTS):
        mask = 0
        for b in range(N):
            nb = v ^ (1 << b)
            neighbors[v].append(nb)
            mask |= (1 << nb)
        neighbor_masks[v] = mask
        
    best_path = []
    best_len = 0
    
    def get_valid_moves(used_mask, end_vertex):
        """
        Returns list of valid candidates to extend from end_vertex.
        A candidate v is valid if:
        1. v is adjacent to end_vertex
        2. v is not used
        3. v is not adjacent to any other used vertex (except end_vertex)
        """
        moves = []
        for v in neighbors[end_vertex]:
            if used_mask & (1 << v):
                continue
            # Check if v is adjacent to any used vertex other than end_vertex
            # Used vertices excluding end_vertex is (used_mask ^ (1 << end_vertex))
            if neighbor_masks[v] & (used_mask ^ (1 << end_vertex)):
                continue
            moves.append(v)
        return moves

    def score_move(used_mask, v):
        """
        Score a move. Lower is better (more constrained).
        Primary: number of valid moves from v (Warnsdorff).
        """
        # Simulate adding v
        new_used_mask = used_mask | (1 << v)
        
        # Count valid moves from v
        count = 0
        for u in neighbors[v]:
            if new_used_mask & (1 << u):
                continue
            # Check if u is adjacent to any used vertex other than v
            # Used vertices excluding v is (new_used_mask ^ (1 << v))
            if neighbor_masks[u] & (new_used_mask ^ (1 << v)):
                continue
            count += 1
            
        return count

    def bidirectional_grow(initial_path=None, rng=None):
        if rng is None:
            rng = random
            
        if initial_path is None:
            v0 = rng.randint(0, MAX_VERTS - 1)
            v1 = rng.choice(neighbors[v0])
            path = [v0, v1]
        else:
            path = list(initial_path)
            
        if len(path) < 2:
            return path
            
        used_mask = 0
        for v in path:
            used_mask |= (1 << v)
            
        while True:
            if time.time() - start_time > DEADLINE:
                return path
                
            left_end = path[0]
            right_end = path[-1]
            
            left_moves = get_valid_moves(used_mask, left_end)
            right_moves = get_valid_moves(used_mask, right_end)
            
            if not left_moves and not right_moves:
                return path
            
            # Score all moves
            best_left = None
            best_left_score = float('inf')
            if left_moves:
                min_score = float('inf')
                candidates = []
                for v in left_moves:
                    s = score_move(used_mask, v)
                    if s < min_score:
                        min_score = s
                        candidates = [v]
                    elif s == min_score:
                        candidates.append(v)
                best_left = rng.choice(candidates)
                best_left_score = min_score
                
            best_right = None
            best_right_score = float('inf')
            if right_moves:
                min_score = float('inf')
                candidates = []
                for v in right_moves:
                    s = score_move(used_mask, v)
                    if s < min_score:
                        min_score = s
                        candidates = [v]
                    elif s == min_score:
                        candidates.append(v)
                best_right = rng.choice(candidates)
                best_right_score = min_score
            
            # Choose end: prefer the more constrained side (lower score)
            if not left_moves:
                path.append(best_right)
                used_mask |= (1 << best_right)
            elif not right_moves:
                path.insert(0, best_left)
                used_mask |= (1 << best_left)
            else:
                if best_left_score < best_right_score:
                    path.insert(0, best_left)
                    used_mask |= (1 << best_left)
                elif best_right_score < best_left_score:
                    path.append(best_right)
                    used_mask |= (1 << best_right)
                else:
                    # Tie: pick randomly
                    if rng.random() < 0.5:
                        path.insert(0, best_left)
                        used_mask |= (1 << best_left)
                    else:
                        path.append(best_right)
                        used_mask |= (1 << best_right)

    def generate_initial_greedy(rng=None):
        """
        Generate a short initial path using greedy extension.
        """
        if rng is None:
            rng = random
            
        v0 = rng.randint(0, MAX_VERTS - 1)
        path = [v0]
        used_mask = 1 << v0
        
        for _ in range(20):
            current = path[-1]
            moves = get_valid_moves(used_mask, current)
            if not moves:
                break
            # Pick the one with lowest Warnsdorff score
            min_score = float('inf')
            candidates = []
            for v in moves:
                # Quick score: count unvisited neighbors for speed in init
                s = 0
                for u in neighbors[v]:
                    if not (used_mask & (1 << u)):
                        s += 1
                if s < min_score:
                    min_score = s
                    candidates = [v]
                elif s == min_score:
                    candidates.append(v)
                    
            next_v = rng.choice(candidates)
            path.append(next_v)
            used_mask |= (1 << next_v)
            
        return path

    attempts = 0
    
    # Phase 1: Structured attempts with varied seeds
    for i in range(10):
        if time.time() - start_time > DEADLINE:
            break
        rng = random.Random(i * 7919 + 13)
        initial = generate_initial_greedy(rng)
        if len(initial) < 2:
            continue
        p = bidirectional_grow(initial, rng)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        attempts += 1

    # Phase 2: Random restarts
    restart_rng = random.Random()
    while time.time() - start_time < DEADLINE:
        v0 = restart_rng.randint(0, MAX_VERTS - 1)
        v1 = restart_rng.choice(neighbors[v0])
        initial = [v0, v1]
        
        # Sometimes extend the initial path a bit more before bidirectional growth
        if restart_rng.random() < 0.5:
            used_mask = (1 << v0) | (1 << v1)
            for _ in range(restart_rng.randint(2, 8)):
                current = initial[-1]
                moves = get_valid_moves(used_mask, current)
                if not moves:
                    break
                # Quick greedy step
                min_score = float('inf')
                candidates = []
                for v in moves:
                    s = 0
                    for u in neighbors[v]:
                        if not (used_mask & (1 << u)):
                            s += 1
                    if s < min_score:
                        min_score = s
                        candidates = [v]
                    elif s == min_score:
                        candidates.append(v)
                next_v = restart_rng.choice(candidates)
                initial.append(next_v)
                used_mask |= (1 << next_v)
        
        p = bidirectional_grow(initial, restart_rng)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        
        attempts += 1

    if not best_path:
        best_path = [0, 1]
        
    return best_path
