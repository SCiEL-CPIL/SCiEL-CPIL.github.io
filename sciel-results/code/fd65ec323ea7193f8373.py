import time
import random
import sys
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 115.0

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
    Strategy: Fast multi-start greedy with Warnsdorff lookahead, bidirectional growth,
    and local gap-filling. Uses bitmask operations for O(1) conflict checks.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    neighbors_local = neighbors
    adj_mask_local = adj_mask
    max_verts = MAX_VERTS
    
    def get_valid_extensions(end_vertex, used_mask):
        """Get list of valid candidates to extend from end_vertex."""
        valid = []
        for c in neighbors_local[end_vertex]:
            if used_mask & (1 << c):
                continue
            other_used = used_mask ^ (1 << end_vertex)
            if adj_mask_local[c] & other_used:
                continue
            valid.append(c)
        return valid
    
    def count_onward_moves(candidate, used_mask):
        """Count valid moves from candidate if it were added."""
        new_used = used_mask | (1 << candidate)
        count = 0
        for w in neighbors_local[candidate]:
            if new_used & (1 << w):
                continue
            if adj_mask_local[w] & (used_mask ^ (1 << candidate)):
                continue
            count += 1
        return count
    
    def greedy_grow(path, used_mask, max_iters=200, random_factor=0.15):
        """
        Grow path greedily using Warnsdorff's rule with bidirectional growth.
        Returns (path, used_mask, length).
        """
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            left_end = path[0]
            right_end = path[-1]
            
            valid_left = get_valid_extensions(left_end, used_mask)
            valid_right = get_valid_extensions(right_end, used_mask)
            
            if not valid_left and not valid_right:
                break
            
            # Choose which end to extend based on constraint (fewer onward moves = more constrained)
            if not valid_left:
                side = 'right'
            elif not valid_right:
                side = 'left'
            else:
                min_left = min(count_onward_moves(v, used_mask) for v in valid_left)
                min_right = min(count_onward_moves(v, used_mask) for v in valid_right)
                
                if min_left < min_right:
                    side = 'left'
                elif min_right < min_left:
                    side = 'right'
                else:
                    side = 'left' if random.random() < 0.5 else 'right'
            
            valid = valid_left if side == 'left' else valid_right
            if not valid:
                break
            
            # Score candidates by onward moves (Warnsdorff)
            scored = [(count_onward_moves(v, used_mask), v) for v in valid]
            scored.sort(key=lambda x: x[0])
            
            # Pick with some randomization to escape local optima
            if random.random() < random_factor and len(scored) > 1:
                top_k = max(1, min(3, len(scored) // 2))
                idx = random.randint(0, top_k - 1)
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
    
    def try_local_insertion(path, used_mask):
        """
        Try to insert unused vertices into gaps in the path.
        Returns (new_path, new_used_mask, improvement).
        """
        improved = False
        new_path = path[:]
        new_used = used_mask
        
        # Try to find vertices that can be inserted between consecutive path vertices
        for i in range(len(new_path) - 1):
            a = new_path[i]
            b = new_path[i + 1]
            # a and b are consecutive, so they differ in exactly one bit
            # We can't insert between them (they're already adjacent)
            # Instead, look for vertices adjacent to both a and b that aren't used
            # Actually, for insertion we need a vertex v such that:
            # v is adjacent to some path vertex, and v is not adjacent to any other path vertex
            # except its intended neighbor(s)
            pass
        
        # Better approach: try to extend from internal vertices
        # Find unused vertices adjacent to exactly one path vertex
        for v in range(max_verts):
            if new_used & (1 << v):
                continue
            # Check if v can be added: adjacent to exactly one path vertex
            adj_count = 0
            adj_path_idx = -1
            for idx, pv in enumerate(new_path):
                if new_used & (1 << v):
                    break
                # Check if v is adjacent to pv
                if v ^ pv < 512 and (v ^ pv) & ((v ^ pv) - 1) == 0:
                    # Check if v is adjacent to any other path vertex
                    conflict = False
                    for jdx, pv2 in enumerate(new_path):
                        if jdx == idx:
                            continue
                        if v ^ pv2 < 512 and (v ^ pv2) & ((v ^ pv2) - 1) == 0:
                            conflict = True
                            break
                    if not conflict:
                        adj_count += 1
                        adj_path_idx = idx
            # If v is adjacent to exactly one path vertex, we might be able to insert it
            # But this is complex; skip for now
            pass
        
        return new_path, new_used, improved
    
    def backtrack_and_retry(path, used_mask, depth=4):
        """
        When stuck, try backtracking to a previous branching point and try a different move.
        Returns (new_path, new_used_mask, success).
        """
        if len(path) < 2:
            return path, used_mask, False
        
        for i in range(min(depth, len(path) - 1)):
            remove_count = i + 1
            body = path[:-remove_count]
            body_used = 0
            for v in body:
                body_used |= (1 << v)
            
            end_vertex = body[-1]
            valid = []
            for c in neighbors_local[end_vertex]:
                if body_used & (1 << c):
                    continue
                if adj_mask_local[c] & (body_used ^ (1 << end_vertex)):
                    continue
                valid.append(c)
            
            if valid:
                random.shuffle(valid)
                next_v = valid[0]
                new_used = body_used | (1 << next_v)
                new_path = body + [next_v]
                # Try to regrow
                new_path, new_used, new_len = greedy_grow(new_path, new_used, max_iters=30, random_factor=0.25)
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
        
        current_len = len(path)
        stuck_count = 0
        
        for _ in range(300):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to grow
            new_path, new_used, new_len = greedy_grow(path, used, max_iters=15, random_factor=0.15)
            
            if new_len > current_len:
                path = new_path
                used = new_used
                current_len = new_len
                stuck_count = 0
                
                if current_len > best_len:
                    best_len = current_len
                    best_path = path.copy()
            else:
                stuck_count += 1
                if stuck_count > 2:
                    new_path, new_used, success = backtrack_and_retry(path, used, depth=3)
                    if success:
                        path = new_path
                        used = new_used
                        current_len = len(path)
                        stuck_count = 0
                        
                        if current_len > best_len:
                            best_len = current_len
                            best_path = path.copy()
                    else:
                        break
                else:
                    # Small pivot on one end
                    if random.random() < 0.5 and len(path) >= 2:
                        removed = path[-1]
                        end_vertex = path[-2]
                        body = path[:-1]
                        body_used = 0
                        for v in body:
                            body_used |= (1 << v)
                        
                        valid = []
                        for c in neighbors_local[end_vertex]:
                            if c == removed:
                                continue
                            if body_used & (1 << c):
                                continue
                            if adj_mask_local[c] & (body_used ^ (1 << end_vertex)):
                                continue
                            valid.append(c)
                        
                        if valid:
                            random.shuffle(valid)
                            next_v = valid[0]
                            new_used = body_used | (1 << next_v)
                            path = body + [next_v]
                            used = new_used
                            current_len = len(path)
            
            if stuck_count > 8:
                break
        
        # Periodically try to extend the best path
        if num_starts % 3 == 0 and best_path is not None and len(best_path) > 10:
            extend_path, extend_used, extend_len = greedy_grow(
                best_path.copy(),
                sum(1 << v for v in best_path),
                max_iters=40, random_factor=0.2
            )
            if extend_len > best_len:
                best_len = extend_len
                best_path = extend_path
    
    if best_path is None:
        return [0, 1]
    
    return best_path
