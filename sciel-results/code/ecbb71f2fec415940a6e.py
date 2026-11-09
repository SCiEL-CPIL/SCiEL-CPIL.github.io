import time
import random
import sys
import numpy as np
import scipy.sparse as sp

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 115.0  # Safety margin for 150s limit

# Precompute neighbors
neighbors = [[] for _ in range(MAX_VERTS)]
for v in range(MAX_VERTS):
    for b in range(N):
        nb = v ^ (1 << b)
        neighbors[v].append(nb)

# Precompute adjacency masks for conflict checking
adj_mask = [0] * MAX_VERTS
for v in range(MAX_VERTS):
    m = 0
    for nb in neighbors[v]:
        m |= (1 << nb)
    adj_mask[v] = m

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Simulated Annealing on path representations with moves:
    - Extend at either end
    - Pivot at either end (remove last, add different neighbor)
    - Shift (remove from one end, add to other)
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    max_verts = MAX_VERTS
    neighbors_local = neighbors
    adj_mask_local = adj_mask
    
    def is_valid_extension(c, end_vertex, used_mask):
        """Check if vertex c can be added to the path at end_vertex."""
        if used_mask & (1 << c):
            return False
        # c must not be adjacent to any other vertex in the path
        other_used = used_mask ^ (1 << end_vertex)
        if adj_mask_local[c] & other_used:
            return False
        return True
    
    def count_onward_moves(candidate, end_vertex, used_mask):
        """Count valid moves from candidate if it were added."""
        new_used = used_mask | (1 << candidate)
        count = 0
        for w in neighbors_local[candidate]:
            if new_used & (1 << w):
                continue
            if adj_mask_local[w] & used_mask:
                continue
            count += 1
        return count
    
    def get_valid_moves(end_vertex, used_mask):
        """Get list of valid candidates to extend from end_vertex."""
        valid = []
        for c in neighbors_local[end_vertex]:
            if is_valid_extension(c, end_vertex, used_mask):
                valid.append(c)
        return valid
    
    def path_to_used_mask(path):
        """Convert path to bitmask of used vertices."""
        used = 0
        for v in path:
            used |= (1 << v)
        return used
    
    def try_extend(path, used_mask, end, random_seed=None):
        """Try to extend path at given end ('left' or 'right').
        Returns (new_path, new_used, success)."""
        if end == 'left':
            end_vertex = path[0]
        else:
            end_vertex = path[-1]
        
        valid = get_valid_moves(end_vertex, used_mask)
        if not valid:
            return path, used_mask, False
        
        # Use Warnsdorff's rule: pick candidate with fewest onward moves
        scored = [(count_onward_moves(v, end_vertex, used_mask), v) for v in valid]
        scored.sort(key=lambda x: x[0])
        
        # With some probability, pick randomly among top candidates
        if random_seed is not None and random.random() < 0.3:
            # Pick from top half
            top_half = len(scored) // 2 + 1
            idx = random.randint(0, min(top_half, len(scored)) - 1)
        else:
            idx = 0
        
        next_v = scored[idx][1]
        new_used = used_mask | (1 << next_v)
        
        if end == 'left':
            new_path = [next_v] + path
        else:
            new_path = path + [next_v]
        
        return new_path, new_used, True
    
    def try_pivot(path, used_mask, end):
        """Try to pivot at given end: remove last vertex, add different neighbor.
        Returns (new_path, new_used, success)."""
        if len(path) < 2:
            return path, used_mask, False
        
        if end == 'left':
            removed = path[0]
            end_vertex = path[1]
            path_body = path[1:]
        else:
            removed = path[-1]
            end_vertex = path[-2]
            path_body = path[:-1]
        
        body_used = 0
        for v in path_body:
            body_used |= (1 << v)
        
        # Find valid neighbors of end_vertex that aren't 'removed'
        valid = []
        for c in neighbors_local[end_vertex]:
            if c == removed:
                continue
            if body_used & (1 << c):
                continue
            if adj_mask_local[c] & (body_used ^ (1 << end_vertex)):
                continue
            valid.append(c)
        
        if not valid:
            return path, used_mask, False
        
        next_v = random.choice(valid)
        new_used = body_used | (1 << next_v)
        
        if end == 'left':
            new_path = [next_v] + path_body
        else:
            new_path = path_body + [next_v]
        
        return new_path, new_used, True
    
    # Multi-start with simulated annealing
    temperature = 50.0
    cooling_rate = 0.995
    
    while time.time() - start_time < DEADLINE:
        # Initialize with a greedy bidirectional growth
        v0 = random.randint(0, MAX_VERTS - 1)
        path = [v0]
        used = 1 << v0
        
        # Pick first neighbor
        v1 = random.choice(neighbors_local[v0])
        path.append(v1)
        used |= (1 << v1)
        
        # Greedy growth
        for _ in range(200):  # Max iterations for initial growth
            left_end = path[0]
            right_end = path[-1]
            
            valid_left = get_valid_moves(left_end, used)
            valid_right = get_valid_moves(right_end, used)
            
            if not valid_left and not valid_right:
                break
            
            # Choose side and move
            if not valid_left:
                scored = [(count_onward_moves(v, right_end, used), v) for v in valid_right]
                scored.sort(key=lambda x: x[0])
                next_v = scored[0][1]
                path.append(next_v)
                used |= (1 << next_v)
            elif not valid_right:
                scored = [(count_onward_moves(v, left_end, used), used) for v in valid_left]
                scored = [(count_onward_moves(v, left_end, used), v) for v in valid_left]
                scored.sort(key=lambda x: x[0])
                next_v = scored[0][1]
                path.insert(0, next_v)
                used |= (1 << next_v)
            else:
                # Pick side with more options or better score
                score_left = min(count_onward_moves(v, left_end, used) for v in valid_left)
                score_right = min(count_onward_moves(v, right_end, used) for v in valid_right)
                
                if score_left < score_right:
                    # Extend left
                    scored = [(count_onward_moves(v, left_end, used), v) for v in valid_left]
                    scored.sort(key=lambda x: x[0])
                    next_v = scored[0][1]
                    path.insert(0, next_v)
                    used |= (1 << next_v)
                elif score_right < score_left:
                    # Extend right
                    scored = [(count_onward_moves(v, right_end, used), v) for v in valid_right]
                    scored.sort(key=lambda x: x[0])
                    next_v = scored[0][1]
                    path.append(next_v)
                    used |= (1 << next_v)
                else:
                    if random.random() < 0.5:
                        scored = [(count_onward_moves(v, left_end, used), v) for v in valid_left]
                        scored.sort(key=lambda x: x[0])
                        next_v = scored[0][1]
                        path.insert(0, next_v)
                        used |= (1 << next_v)
                    else:
                        scored = [(count_onward_moves(v, right_end, used), v) for v in valid_right]
                        scored.sort(key=lambda x: x[0])
                        next_v = scored[0][1]
                        path.append(next_v)
                        used |= (1 << next_v)
        
        # Now apply simulated annealing to improve
        current_len = len(path)
        
        for sa_iter in range(500):  # Max SA iterations per start
            if time.time() - start_time > DEADLINE:
                break
            
            # Choose a move type
            move_type = random.choice(['extend', 'pivot', 'shift'])
            
            if move_type == 'extend':
                end = random.choice(['left', 'right'])
                new_path, new_used, success = try_extend(path, used, end)
                if success:
                    new_len = len(new_path)
                    delta = new_len - current_len
                    # Accept if improves or with probability based on temperature
                    if delta >= 0 or random.random() < np.exp(delta / temperature):
                        path = new_path
                        used = new_used
                        current_len = new_len
                        if current_len > best_len:
                            best_len = current_len
                            best_path = path.copy()
            
            elif move_type == 'pivot':
                end = random.choice(['left', 'right'])
                new_path, new_used, success = try_pivot(path, used, end)
                if success:
                    new_len = len(new_path)
                    delta = new_len - current_len
                    if delta >= 0 or random.random() < np.exp(delta / temperature):
                        path = new_path
                        used = new_used
                        current_len = new_len
                        if current_len > best_len:
                            best_len = current_len
                            best_path = path.copy()
            
            else:  # shift
                # Remove from one end, add to other
                if len(path) < 2:
                    continue
                
                remove_end = random.choice(['left', 'right'])
                add_end = 'right' if remove_end == 'left' else 'left'
                
                if remove_end == 'left':
                    removed = path[0]
                    new_body = path[1:]
                else:
                    removed = path[-1]
                    new_body = path[:-1]
                
                body_used = 0
                for v in new_body:
                    body_used |= (1 << v)
                
                # Find valid extension at add_end
                if add_end == 'left':
                    end_vertex = new_body[0]
                else:
                    end_vertex = new_body[-1]
                
                valid = []
                for c in neighbors_local[end_vertex]:
                    if body_used & (1 << c):
                        continue
                    if adj_mask_local[c] & (body_used ^ (1 << end_vertex)):
                        continue
                    valid.append(c)
                
                if valid:
                    next_v = random.choice(valid)
                    new_used = body_used | (1 << next_v)
                    
                    if add_end == 'left':
                        new_path = [next_v] + new_body
                    else:
                        new_path = new_body + [next_v]
                    
                    new_len = len(new_path)
                    delta = new_len - current_len
                    if delta >= 0 or random.random() < np.exp(delta / temperature):
                        path = new_path
                        used = new_used
                        current_len = new_len
                        if current_len > best_len:
                            best_len = current_len
                            best_path = path.copy()
            
            # Cool down
            temperature *= cooling_rate
            if temperature < 0.1:
                break
        
        if best_path is None:
            # Fallback
            return [0]
        
        return best_path

# EVOLVE-BLOCK-END
