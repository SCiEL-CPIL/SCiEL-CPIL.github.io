import time
import random

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 117.0

def construct():
    start_time = time.time()
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = []
    best_len = 0
    
    def get_valid_moves(used, end_vertex):
        """
        Returns list of valid unvisited neighbors of end_vertex that are not
        adjacent to any other used vertex.
        """
        moves = []
        for v in neighbors[end_vertex]:
            if used[v]:
                continue
            valid = True
            for u in neighbors[v]:
                if u == end_vertex:
                    continue
                if used[u]:
                    valid = False
                    break
            if valid:
                moves.append(v)
        return moves
    
    def get_scored_moves(used, end_vertex):
        """
        Returns list of (candidate, warnsdorff_score) for valid moves.
        """
        moves = []
        for v in neighbors[end_vertex]:
            if used[v]:
                continue
            valid = True
            for u in neighbors[v]:
                if u == end_vertex:
                    continue
                if used[u]:
                    valid = False
                    break
            if not valid:
                continue
            score = 0
            for u in neighbors[v]:
                if not used[u]:
                    score += 1
            moves.append((v, score))
        return moves
    
    def greedy_grow(path, used, max_steps=200):
        """
        Grow path bidirectionally using Warnsdorff heuristic.
        Returns updated path and used array.
        """
        steps = 0
        while steps < max_steps:
            if time.time() - start_time > DEADLINE - 2.0:
                break
            
            left_end = path[0]
            right_end = path[-1]
            
            left_moves = get_scored_moves(used, left_end)
            right_moves = get_scored_moves(used, right_end)
            
            if not left_moves and not right_moves:
                break
            
            # Pick the most constrained end (lowest min score)
            if not left_moves:
                # Must grow right
                min_score = min(s for _, s in right_moves)
                candidates = [v for v, s in right_moves if s == min_score]
                chosen = random.choice(candidates)
                path.append(chosen)
                used[chosen] = True
            elif not right_moves:
                # Must grow left
                min_score = min(s for _, s in left_moves)
                candidates = [v for v, s in left_moves if s == min_score]
                chosen = random.choice(candidates)
                path.insert(0, chosen)
                used[chosen] = True
            else:
                min_left = min(s for _, s in left_moves)
                min_right = min(s for _, s in right_moves)
                
                if min_left < min_right:
                    candidates = [v for v, s in left_moves if s == min_left]
                    chosen = random.choice(candidates)
                    path.insert(0, chosen)
                    used[chosen] = True
                elif min_right < min_left:
                    candidates = [v for v, s in right_moves if s == min_right]
                    chosen = random.choice(candidates)
                    path.append(chosen)
                    used[chosen] = True
                else:
                    # Tie: random
                    if random.random() < 0.5:
                        candidates = [v for v, s in left_moves if s == min_left]
                        chosen = random.choice(candidates)
                        path.insert(0, chosen)
                        used[chosen] = True
                    else:
                        candidates = [v for v, s in right_moves if s == min_right]
                        chosen = random.choice(candidates)
                        path.append(chosen)
                        used[chosen] = True
            
            steps += 1
        
        return path, used
    
    def try_extend_end(path, used, side, max_steps=50):
        """
        Try to extend the path at the given side ('left' or 'right')
        by greedily growing, but only keep if we actually extended.
        """
        original_len = len(path)
        if side == 'left':
            end = path[0]
        else:
            end = path[-1]
        
        moves = get_valid_moves(used, end)
        if not moves:
            return path, used
        
        # Try each move with greedy continuation
        for mv in moves:
            test_path = list(path)
            test_used = list(used)
            test_used[mv] = True
            if side == 'left':
                test_path.insert(0, mv)
            else:
                test_path.append(mv)
            
            # Greedily grow from both ends for a few steps
            test_path, test_used = greedy_grow(test_path, test_used, max_steps=max_steps)
            
            if len(test_path) > original_len:
                return test_path, test_used
        
        return path, used
    
    def local_refine(path, used, max_iterations=100):
        """
        Local refinement: try to insert vertices at various positions
        and extend ends. Iterate until no improvement or deadline.
        """
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            if time.time() - start_time > DEADLINE - 2.0:
                break
            improved = False
            iteration += 1
            
            # Try extending both ends
            new_path, new_used = try_extend_end(path, used, 'right', max_steps=30)
            if len(new_path) > len(path):
                path, used = new_path, new_used
                improved = True
            
            new_path, new_used = try_extend_end(path, used, 'left', max_steps=30)
            if len(new_path) > len(path):
                path, used = new_path, new_used
                improved = True
            
            # Try segment reconnection: remove a middle vertex and see if we can
            # reconnect or find a longer path
            if len(path) > 4:
                # Pick a random interior position
                pos = random.randint(1, len(path) - 2)
                v = path[pos]
                left_neighbor = path[pos - 1]
                right_neighbor = path[pos + 1]
                
                # Check if left_neighbor and right_neighbor are adjacent
                # (they differ in exactly one bit)
                diff = left_neighbor ^ right_neighbor
                if diff & (diff - 1) == 0:  # power of 2 means one bit diff
                    # They are adjacent! Remove v and reconnect
                    test_path = path[:pos] + path[pos+1:]
                    test_used = list(used)
                    test_used[v] = False
                    
                    # Verify the new path is valid (v was the only connection issue)
                    # Since v is removed, we need to check that left_neighbor and right_neighbor
                    # are not adjacent to any other used vertex except their path neighbors
                    valid = True
                    for u in neighbors[left_neighbor]:
                        if u == path[pos-2] if pos >= 2 else True:
                            if u == right_neighbor:
                                continue
                            if u == (path[pos-2] if pos >= 2 else None):
                                continue
                            if test_used[u] and u not in (path[pos-2] if pos >= 2 else -1, right_neighbor):
                                # Check if u is adjacent to left_neighbor and is used
                                # but not a path neighbor of left_neighbor
                                if u != (path[pos-2] if pos >= 2 else -1) and u != right_neighbor:
                                    valid = False
                                    break
                    
                    if valid:
                        # Also check right_neighbor
                        for u in neighbors[right_neighbor]:
                            if u == (path[pos+2] if pos+2 < len(test_path) else -1):
                                continue
                            if u == left_neighbor:
                                continue
                            if test_used[u]:
                                if u != left_neighbor and u != (path[pos+2] if pos+2 < len(test_path) else -1):
                                    valid = False
                                    break
                    
                    if valid:
                        path, used = test_path, test_used
                        improved = True
                        continue
            
            # Try inserting an unused vertex between two consecutive path vertices
            if len(path) >= 2:
                pos = random.randint(0, len(path) - 2)
                a = path[pos]
                b = path[pos + 1]
                
                # Find common neighbors of a and b that are unused
                common = []
                for u in neighbors[a]:
                    if not used[u] and u != b:
                        if u in neighbors[b]:
                            common.append(u)
                
                if common:
                    for u in common:
                        test_path = path[:pos+1] + [u] + path[pos+1:]
                        test_used = list(used)
                        test_used[u] = True
                        
                        # Verify validity: u must not be adjacent to any other used vertex
                        # except a and b
                        valid = True
                        for w in neighbors[u]:
                            if w == a or w == b:
                                continue
                            if test_used[w]:
                                valid = False
                                break
                        
                        if valid:
                            path, used = test_path, test_used
                            improved = True
                            break
        
        return path, used
    
    def full_solve():
        """
        Full solve: greedy grow + local refinement.
        """
        # Start with a random edge
        v0 = random.randint(0, MAX_VERTS - 1)
        v1 = random.choice(neighbors[v0])
        path = [v0, v1]
        used = [False] * MAX_VERTS
        used[v0] = True
        used[v1] = True
        
        # Phase 1: Greedy growth
        path, used = greedy_grow(path, used, max_steps=300)
        
        # Phase 2: Local refinement
        path, used = local_refine(path, used, max_iterations=80)
        
        return path
    
    # Multiple attempts with refinement
    attempt = 0
    while time.time() - start_time < DEADLINE:
        attempt += 1
        p = full_solve()
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        
        # If we've found a good solution, do more refinement on it
        if best_len >= 120:
            # Additional refinement pass on the best path
            used = [False] * MAX_VERTS
            for v in best_path:
                used[v] = True
            refined, _ = local_refine(best_path, used, max_iterations=50)
            if len(refined) > best_len:
                best_len = len(refined)
                best_path = refined
    
    if not best_path:
        best_path = [0, 1]
    
    return best_path
