import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 115.0  # Safety margin for 150s limit (evaluator kills at 150s)

# Precompute neighbors for all vertices
neighbors = [[] for _ in range(MAX_VERTS)]
for v in range(MAX_VERTS):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

# Precompute adjacency masks for fast conflict checking
# adj_mask[v] has bit i set iff v and i are adjacent (Hamming distance 1)
adj_mask = [0] * MAX_VERTS
for v in range(MAX_VERTS):
    m = 0
    for nb in neighbors[v]:
        m |= (1 << nb)
    adj_mask[v] = m


def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Bidirectional growth with Warnsdorff's rule and bitmask-based validity checks.
    Combines efficient bitmask operations with multi-start randomization and backtracking.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    neighbors_local = neighbors
    adj_mask_local = adj_mask
    max_verts = MAX_VERTS
    
    def get_valid_extensions(end_vertex, used_mask, other_end_vertex):
        """
        Get list of valid candidates to extend from end_vertex.
        A candidate c is valid if:
        1. c not in used_mask
        2. c not adjacent to any vertex in path except end_vertex
           (i.e., adj_mask[c] & (used_mask ^ (1 << end_vertex)) == 0)
        """
        valid = []
        for c in neighbors_local[end_vertex]:
            if used_mask & (1 << c):
                continue
            # c must not be adjacent to any other vertex in the path except end_vertex
            other_used = used_mask ^ (1 << end_vertex)
            if adj_mask_local[c] & other_used:
                continue
            valid.append(c)
        return valid
    
    def count_onward_moves(candidate, used_mask, other_end_vertex):
        """
        Count valid moves from candidate if it were added to the path.
        This is used for Warnsdorff's rule heuristic.
        """
        new_used = used_mask | (1 << candidate)
        count = 0
        for w in neighbors_local[candidate]:
            if new_used & (1 << w):
                continue
            # w must not be adjacent to any vertex in the path except candidate
            # The path would be: [other_end, ..., end_vertex, candidate]
            # So w must not be adjacent to any vertex in path[:-1] = [other_end, ..., end_vertex]
            # which corresponds to used_mask (before adding candidate)
            if adj_mask_local[w] & used_mask:
                continue
            count += 1
        return count
    
    def greedy_grow_bidirectional(path, used_mask, max_iters=500, random_factor=0.15):
        """
        Grow path greedily from both ends using Warnsdorff's rule.
        Returns (path, used_mask, length).
        """
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            left_end = path[0]
            right_end = path[-1]
            
            valid_left = get_valid_extensions(left_end, used_mask, right_end)
            valid_right = get_valid_extensions(right_end, used_mask, left_end)
            
            if not valid_left and not valid_right:
                break
            
            # Choose which end to extend using Warnsdorff's rule
            if not valid_left:
                side = 'right'
            elif not valid_right:
                side = 'left'
            else:
                # Score both ends: pick the side with fewer onward moves (more constrained)
                min_left = min(count_onward_moves(v, used_mask, right_end) for v in valid_left)
                min_right = min(count_onward_moves(v, used_mask, left_end) for v in valid_right)
                
                if min_left < min_right:
                    side = 'left'
                elif min_right < min_left:
                    side = 'right'
                else:
                    side = 'left' if random.random() < 0.5 else 'right'
            
            # Get valid moves for chosen side
            if side == 'left':
                valid = valid_left
                other_end = right_end
            else:
                valid = valid_right
                other_end = left_end
            
            if not valid:
                break
            
            # Score candidates using Warnsdorff's rule
            scored = [(count_onward_moves(v, used_mask, other_end), v) for v in valid]
            scored.sort(key=lambda x: x[0])
            
            # Pick with some randomization to explore different regions
            if random.random() < random_factor:
                top_half = max(1, len(scored) // 2)
                idx = random.randint(0, top_half - 1)
            else:
                idx = 0
            
            next_v = scored[idx][1]
            new_used = used_mask | (1 << next_v)
            
            if side == 'left':
                path.insert(0, next_v)
            else:
                path.append(next_v)
            
            used_mask = new_used
        
        return path, used_mask, len(path)
    
    def backtrack_and_retry(path, used_mask, max_depth=8):
        """
        When stuck, try backtracking to a previous branching point and try a different move.
        Returns (new_path, new_used_mask, success).
        """
        if len(path) < 2:
            return path, used_mask, False
        
        # Try backtracking from both ends
        for side in ['right', 'left']:
            for i in range(min(max_depth, len(path) - 1)):
                # Remove last i+1 vertices from the specified side
                if side == 'right':
                    remove_count = i + 1
                    body = path[:-remove_count] if remove_count < len(path) else []
                    if not body:
                        continue
                    removed_vertices = path[-remove_count:]
                    end_vertex = body[-1]
                    other_end = body[0]
                else:
                    remove_count = i + 1
                    body = path[remove_count:] if remove_count < len(path) else []
                    if not body:
                        continue
                    removed_vertices = path[:remove_count]
                    end_vertex = body[0]
                    other_end = body[-1]
                
                body_used = 0
                for v in body:
                    body_used |= (1 << v)
                
                valid = []
                for c in neighbors_local[end_vertex]:
                    if body_used & (1 << c):
                        continue
                    other_used = body_used ^ (1 << end_vertex)
                    if adj_mask_local[c] & other_used:
                        continue
                    # Also ensure c is not in removed_vertices (should be handled by body_used check)
                    valid.append(c)
                
                if valid:
                    # Pick a random different move
                    random.shuffle(valid)
                    next_v = valid[0]
                    new_used = body_used | (1 << next_v)
                    
                    if side == 'right':
                        new_path = body + [next_v]
                    else:
                        new_path = [next_v] + body
                    
                    # Try to regrow from this new state
                    new_path, new_used, new_len = greedy_grow_bidirectional(
                        new_path, new_used, max_iters=30, random_factor=0.3
                    )
                    return new_path, new_used, True
        
        return path, used_mask, False
    
    # Multi-start with backtracking
    num_starts = 0
    
    while time.time() - start_time < DEADLINE:
        num_starts += 1
        
        # Initialize with a random starting edge
        v0 = random.randint(0, max_verts - 1)
        path = [v0]
        used = 1 << v0
        
        v1 = random.choice(neighbors_local[v0])
        path.append(v1)
        used |= (1 << v1)
        
        # Greedy growth with backtracking
        stuck_count = 0
        
        for _ in range(1000):  # Max iterations per start
            if time.time() - start_time > DEADLINE:
                break
            
            prev_len = len(path)
            path, used, current_len = greedy_grow_bidirectional(path, used, max_iters=200)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            if len(path) == prev_len:
                # Stuck, try backtracking
                stuck_count += 1
                if stuck_count > 3:
                    # Give up on this start
                    break
                
                new_path, new_used, success = backtrack_and_retry(path, used)
                if success:
                    path = new_path
                    used = new_used
                    stuck_count = 0
                else:
                    break
            else:
                stuck_count = 0
            
            # If path is very long, we might be near optimal for this start
            if len(path) > 180:
                break
        
        # Check time before next start
        if time.time() - start_time > DEADLINE:
            break
    
    if best_path is None:
        # Fallback: simple greedy from 0
        path = [0]
        used = 1 << 0
        while True:
            cur = path[-1]
            valid = get_valid_extensions(cur, used, path[0] if len(path) > 1 else -1)
            if not valid:
                break
            # Pick the one with fewest onward moves
            scored = [(count_onward_moves(v, used, path[0] if len(path) > 1 else -1), v) for v in valid]
            scored.sort(key=lambda x: x[0])
            next_v = scored[0][1]
            path.append(next_v)
            used |= (1 << next_v)
        return path
    
    return best_path
