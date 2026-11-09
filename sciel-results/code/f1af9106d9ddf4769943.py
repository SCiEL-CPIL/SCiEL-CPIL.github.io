import time
import random
import sys
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 115.0  # Safety margin for 120s limit

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: 
    1. Multi-start bidirectional greedy growth to find initial candidates.
    2. Intensive local search using limited-depth DFS with backtracking 
       to escape local optima and extend paths.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute a lookup for quick bit_count
    # Not needed since Python 3.10+ has int.bit_count()

    def update_best(path):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    # --- Greedy Growth (Bidirectional) ---
    def bidirectional_grow(v_start, v_end_init, time_limit):
        """
        Grow a snake bidirectionally from v_start and v_end_init.
        Uses Warnsdorff heuristic.
        """
        path = [v_start, v_end_init]
        used = bytearray(MAX_VERTS)
        used[v_start] = 1
        used[v_end_init] = 1
        adj_count = np.zeros(MAX_VERTS, dtype=np.uint8)

        for w in neighbors[v_start]:
            adj_count[w] += 1
        for w in neighbors[v_end_init]:
            adj_count[w] += 1

        if (v_start ^ v_end_init).bit_count() != 1:
            return None

        local_best = path[:]
        
        def add_vertex(v):
            used[v] = 1
            for w in neighbors[v]:
                adj_count[w] += 1

        def remove_vertex(v):
            used[v] = 0
            for w in neighbors[v]:
                adj_count[w] -= 1

        while time.time() - start_time < time_limit:
            v_left = path[0]
            v_right = path[-1]
            
            valid_left = []
            for u in neighbors[v_left]:
                if not used[u] and adj_count[u] == 1:
                    # Count onward moves for heuristic
                    cnt = 0
                    for w in neighbors[u]:
                        if not used[w] and adj_count[w] == 0:
                            cnt += 1
                    valid_left.append((cnt, u))
                    
            valid_right = []
            for u in neighbors[v_right]:
                if not used[u] and adj_count[u] == 1:
                    cnt = 0
                    for w in neighbors[u]:
                        if not used[w] and adj_count[w] == 0:
                            cnt += 1
                    valid_right.append((cnt, u))

            if not valid_left and not valid_right:
                break

            # Choose end to extend: prefer the one with fewer options (Warnsdorff)
            # If counts are equal, randomize
            extend_left = False
            if valid_left and not valid_right:
                extend_left = True
            elif valid_right and not valid_left:
                extend_left = False
            else:
                # Heuristic: extend the side with smaller minimum onward moves
                min_left = min(x[0] for x in valid_left)
                min_right = min(x[0] for x in valid_right)
                if min_left < min_right:
                    extend_left = True
                elif min_right < min_left:
                    extend_left = False
                else:
                    extend_left = random.random() < 0.5

            if extend_left:
                valid_left.sort(key=lambda x: x[0])
                best_score = valid_left[0][0]
                candidates = [x for x in valid_left if x[0] <= best_score + 1]
                _, next_v = random.choice(candidates)
                path.insert(0, next_v)
                add_vertex(next_v)
            else:
                valid_right.sort(key=lambda x: x[0])
                best_score = valid_right[0][0]
                candidates = [x for x in valid_right if x[0] <= best_score + 1]
                _, next_v = random.choice(candidates)
                path.append(next_v)
                add_vertex(next_v)
            
            if len(path) > len(local_best):
                local_best = path[:]
                update_best(path)

        return local_best

    # --- Local Search with DFS Backtracking ---
    def local_search_dfs(path, max_depth, time_limit):
        """
        Try to improve the path by truncating the end and using DFS with backtracking
        to find a longer extension.
        """
        if not path or len(path) < 2:
            return path
        
        current_best = path[:]
        
        # We will try to extend from the end of the path.
        # To allow backtracking, we rebuild the state from the truncated path.
        # We try truncating by k=1, 2, 3... up to a small limit.
        
        for k in range(1, 8):
            if time.time() - start_time > time_limit:
                break
            if len(current_best) <= k:
                break
                
            truncated = current_best[:-k]
            if len(truncated) < 2:
                continue
            
            # Rebuild state for the truncated path
            used = bytearray(MAX_VERTS)
            adj_count = np.zeros(MAX_VERTS, dtype=np.uint8)
            for v in truncated:
                used[v] = 1
                for w in neighbors[v]:
                    adj_count[w] += 1
            
            # DFS to extend from truncated end
            # We limit the depth of the DFS to max_depth to keep it fast
            # We use a stack for iterative DFS to avoid recursion limits and overhead
            # Stack items: (path, used_snapshot, adj_count_snapshot) - too expensive to snapshot.
            # Instead, we do recursive DFS but with strict time checks and depth limits.
            
            def dfs_extend(current_path, depth):
                nonlocal current_best, best_path, best_len
                
                if time.time() - start_time > time_limit:
                    return
                
                if len(current_path) > len(current_best):
                    current_best = current_path[:]
                    update_best(current_path)
                    
                if depth >= max_depth:
                    return
                
                v_end = current_path[-1]
                
                # Gather valid moves
                valid_moves = []
                for u in neighbors[v_end]:
                    if not used[u] and adj_count[u] == 1:
                        # Heuristic score: count onward moves (2-step lookahead)
                        cnt = 0
                        for w in neighbors[u]:
                            if not used[w] and adj_count[w] == 0:
                                # Also check if w has onward moves
                                cnt += 1
                                for x in neighbors[w]:
                                    if not used[x] and adj_count[x] == 0:
                                        cnt += 1
                                        break
                        valid_moves.append((cnt, u))
                        
                if not valid_moves:
                    return
                
                # Sort by heuristic
                valid_moves.sort(key=lambda x: x[0])
                
                # Try moves, limited to a few best candidates to control branching factor
                # This is a "Beam Search" style restriction on the DFS
                # Increase beam width for shallow depths
                beam_width = 5 if depth < 3 else 3
                candidates = valid_moves[:beam_width]
                
                for cnt, u in candidates:
                    if time.time() - start_time > time_limit:
                        return
                        
                    # Add u
                    used[u] = 1
                    for w in neighbors[u]:
                        adj_count[w] += 1
                    current_path.append(u)
                    
                    dfs_extend(current_path, depth + 1)
                    
                    # Remove u
                    current_path.pop()
                    used[u] = 0
                    for w in neighbors[u]:
                        adj_count[w] -= 1

            dfs_extend(truncated[:], 0)
            
            if len(current_best) > len(path):
                path = current_best[:]
                break # If we improved, restart the loop with the new best? 
                       # Or continue trying to improve further?
                       # Let's update the base path for the next k iteration if it improved.
            
        return current_best

    # --- Main Loop ---
    
    # Phase 1: Quick Greedy/Bidirectional starts to get a baseline
    # Run a few fast attempts
    for _ in range(20):
        if time.time() - start_time > DEADLINE - 30:
            break
        v0 = random.randint(0, MAX_VERTS - 1)
        v1 = random.choice(neighbors[v0])
        p = bidirectional_grow(v0, v1, time.time() - start_time + 1.0)
        if p:
            update_best(p)

    # Phase 2: Intensive Local Search on the best path found
    # We repeatedly apply local search to the current best path.
    # Each application might improve the path, allowing further improvements.
    
    no_improve_count = 0
    last_best_len = best_len
    
    while time.time() - start_time < DEADLINE - 5:
        if not best_path:
            break
            
        # Allocate time for this local search iteration
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 2.0:
            break
        
        # Limit the depth of DFS to keep it fast. 
        # As we get closer to the deadline, we can use more depth if we have time.
        # Let's use a fixed moderate depth for stability.
        max_depth = 15
        
        improved_path = local_search_dfs(best_path[:], max_depth, time.time() - start_time + remaining * 0.8)
        
        if improved_path and len(improved_path) > best_len:
            best_len = len(improved_path)
            best_path = improved_path[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If we've had several iterations without improvement, try a fresh start
        # to escape the local optimum
        if no_improve_count >= 3 and time.time() - start_time < DEADLINE - 15:
            no_improve_count = 0
            # Try a few fresh bidirectional growths
            for _ in range(5):
                if time.time() - start_time > DEADLINE - 10:
                    break
                v0 = random.randint(0, MAX_VERTS - 1)
                v1 = random.choice(neighbors[v0])
                p = bidirectional_grow(v0, v1, time.time() - start_time + 0.5)
                if p:
                    update_best(p)

    if best_path is None:
        return [0]
    
    return best_path

if __name__ == "__main__":
    # For local testing
    path = construct()
    print(f"Length: {len(path)}")
    # Verify
    for i in range(len(path) - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            print(f"Invalid edge at {i}")
            break
    for i in range(len(path)):
        for j in range(i+2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                print(f"Invalid non-consecutive adjacency at {i}, {j}")
                break
