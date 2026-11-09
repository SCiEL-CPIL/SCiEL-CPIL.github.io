import random
import time
from itertools import combinations

def construct():
    n = 20
    deadline = time.time() + 140  # 10s safety margin for return overhead
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(u, v, mask):
        """
        Check if adding edge (u, v) creates a K3,3.
        A K3,3 involving (u,v) must have u, v in opposite partitions.
        Partitions: P1 = {u, x1, x2}, P2 = {v, y1, y2}
        Requirements:
        1. x1, x2 in N(v)
        2. y1, y2 in N(u)
        3. Edges between {x1, x2} and {y1, y2} must exist.
        i.e., for some x1, x2 in N(v), | (N(x1) & N(x2)) & N(u) | >= 2
        """
        if u == v:
            return False
        if mask[u] & (1 << v):
            return False
            
        N_v_mask = mask[v]
        N_u_mask = mask[u]
        
        # Get neighbors of v
        neighbors_v = []
        temp = N_v_mask
        for j in range(n):
            if j != v and (temp & (1 << j)):
                neighbors_v.append(j)
                
        if len(neighbors_v) < 2:
            return True
            
        # Check all pairs in neighbors_v
        for i in range(len(neighbors_v)):
            x1 = neighbors_v[i]
            for j in range(i + 1, len(neighbors_v)):
                x2 = neighbors_v[j]
                # Common neighbors of x1 and x2
                common = mask[x1] & mask[x2]
                # How many of these common neighbors are also neighbors of u?
                count = popcount(common & N_u_mask)
                if count >= 2:
                    return False
        return True

    def add_edge(u, v, mask):
        mask[u] |= (1 << v)
        mask[v] |= (1 << u)

    def remove_edge(u, v, mask):
        mask[u] &= ~(1 << v)
        mask[v] &= ~(1 << u)

    def count_edges(mask):
        total = 0
        for i in range(n):
            total += popcount(mask[i])
        return total // 2

    def greedy_build():
        mask = [0] * n
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if can_add(u, v, mask):
                add_edge(u, v, mask)
        return mask

    def local_search(mask, time_limit):
        start_time = time.time()
        current_edges = count_edges(mask)
        improved = True
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try swaps: remove (a,b), add (c,d)
            # Or just try adding any edge first
            # 1. Try adding any edge
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_pairs)
            
            added_any = False
            for u, v in all_pairs:
                if not (mask[u] & (1 << v)):
                    if can_add(u, v, mask):
                        add_edge(u, v, mask)
                        current_edges += 1
                        improved = True
                        added_any = True
                        break # Restart search with new state? Or continue? 
                        # Usually restart to keep simple, but continuing is fine if we track state
                        # Let's restart the inner loop to ensure we don't miss more adds
                        break
            
            if added_any:
                continue
                
            # 2. If no edges can be added, try swaps
            # Remove an existing edge, see if we can add 2 new ones
            existing_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if mask[i] & (1 << j):
                        existing_edges.append((i, j))
                        
            if not existing_edges:
                break
                
            random.shuffle(existing_edges)
            
            # Try swapping one edge for two
            found_swap = False
            for u, v in existing_edges:
                if time.time() - start_time > time_limit:
                    break
                # Temporarily remove (u,v)
                remove_edge(u, v, mask)
                
                # Try to add 2 new edges
                added = []
                for x, y in all_pairs:
                    if x == u and y == v:
                        continue
                    if x == v and y == u:
                        continue
                    if not (mask[x] & (1 << y)):
                        if can_add(x, y, mask):
                            add_edge(x, y, mask)
                            added.append((x, y))
                            if len(added) == 2:
                                break
                                
                if len(added) == 2:
                    # Net gain +1
                    current_edges += 1
                    improved = True
                    found_swap = True
                    break
                else:
                    # Undo
                    for x, y in added:
                        remove_edge(x, y, mask)
                    add_edge(u, v, mask)
                    
            if found_swap:
                continue
            
            # 3. Try 2-for-1 swaps (remove 2, add 3) - rare but possible
            # This is expensive, so only do it occasionally or if stuck
            if random.random() < 0.1 and time.time() - start_time < time_limit:
                if len(existing_edges) >= 2:
                    e1, e2 = random.sample(existing_edges, 2)
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    remove_edge(u1, v1, mask)
                    remove_edge(u2, v2, mask)
                    
                    added = []
                    for x, y in all_pairs:
                        if (x == u1 and y == v1) or (x == v1 and y == u1):
                            continue
                        if (x == u2 and y == v2) or (x == v2 and y == u2):
                            continue
                        if not (mask[x] & (1 << y)):
                            if can_add(x, y, mask):
                                add_edge(x, y, mask)
                                added.append((x, y))
                                if len(added) == 3:
                                    break
                                    
                    if len(added) == 3:
                        current_edges += 1
                        improved = True
                        break
                    else:
                        for x, y in added:
                            remove_edge(x, y, mask)
                        add_edge(u1, v1, mask)
                        add_edge(u2, v2, mask)

        return mask, current_edges

    # Main Loop
    restarts = 0
    best_mask = [0] * n
    best_edge_count = 0
    
    while time.time() < deadline:
        mask = greedy_build()
        edges = count_edges(mask)
        
        if edges > best_edge_count:
            best_edge_count = edges
            best_mask = mask[:]
            
        # Local search
        remaining = deadline - time.time()
        if remaining > 5:
            # Allocate time for local search
            ls_time = min(2.0, remaining * 0.5)
            mask, edges = local_search(mask, ls_time)
            
            if edges > best_edge_count:
                best_edge_count = edges
                best_mask = mask[:]
                
        restarts += 1
        # If we hit a very high density, we can stop early
        if best_edge_count >= 80:
             # Continue a bit more to see if we can beat it, but be cautious
             if best_edge_count >= 85:
                 break

    # Convert best_mask to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_mask[i] & (1 << j):
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Edges: {len(edges)}, Time: {t1-t0:.2f}s")
