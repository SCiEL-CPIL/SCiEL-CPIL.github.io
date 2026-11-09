import random
import time
import math

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    # Bitmask representation for fast operations
    # adj[u] is an integer where bit v is 1 if (u,v) is an edge
    
    def get_edges_from_bitmasks(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                # Get the lowest set bit
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask ^= lsb
        return edges

    def count_edges(adj):
        # Sum of degrees / 2
        total_deg = sum(bin(a).count('1') for a in adj)
        return total_deg // 2

    def common_neighbors_count(u, v, adj):
        # Number of common neighbors of u and v
        return (adj[u] & adj[v]).bit_count()

    def can_add(u, v, adj):
        if u == v:
            return False
        # Check if edge already exists
        if (adj[u] >> v) & 1:
            return False
        
        # Condition 1: u and v must have at most 2 common neighbors
        if common_neighbors_count(u, v, adj) > 2:
            return False
            
        # Condition 2: For any neighbor y of v, u and y must have at most 1 common neighbor
        # (Because adding edge (u,v) makes v a common neighbor of u and y, so existing common neighbors must be <= 1)
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            y = lsb.bit_length() - 1
            if y != u:
                if common_neighbors_count(u, y, adj) > 1:
                    return False
            mask_v ^= lsb
            
        # Condition 3: For any neighbor y of u, v and y must have at most 1 common neighbor
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            y = lsb.bit_length() - 1
            if y != v:
                if common_neighbors_count(v, y, adj) > 1:
                    return False
            mask_u ^= lsb
            
        return True

    def add_edge(u, v, adj, edge_list):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if u > v:
            u, v = v, u
        edge_list.append((u, v))

    def remove_edge(u, v, adj, edge_list):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        if u > v:
            u, v = v, u
        for i, e in enumerate(edge_list):
            if e == (u, v):
                edge_list.pop(i)
                break

    def greedy_fill(adj, edge_list, rng):
        """Try to add edges until no more can be added."""
        improved = True
        while improved:
            improved = False
            # Generate missing edges
            missing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        missing.append((u, v))
            
            if not missing:
                break
                
            rng.shuffle(missing)
            for u, v in missing:
                if can_add(u, v, adj):
                    add_edge(u, v, adj, edge_list)
                    improved = True

    def simulated_annealing(adj, edge_list, rng, time_budget):
        """
        Simulated Annealing to maximize edge count.
        Moves:
        1. Add a random missing edge.
        2. Swap: Remove a random edge, try to add a random missing edge.
        3. Remove a random edge (to escape local minima).
        """
        current_count = len(edge_list)
        best_count = current_count
        best_adj = adj[:]
        best_edges = edge_list[:]
        
        sa_end_time = start_time + time_budget
        T = 1.0
        cooling = 0.9995
        
        # Precompute list of all possible edges for random selection
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        while (time.time() - start_time) < sa_end_time:
            if T < 0.005:
                break
                
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try to add a random edge
                u, v = rng.choice(all_edges)
                if can_add(u, v, adj):
                    add_edge(u, v, adj, edge_list)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                        best_edges = edge_list[:]
                        
            elif move_type < 0.7:
                # Swap move: Remove one edge, try to add another
                if not edge_list:
                    continue
                
                u1, v1 = rng.choice(edge_list)
                
                # Try to find a replacement edge
                found = False
                for _ in range(10):
                    u2, v2 = rng.choice(all_edges)
                    if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                        continue
                    if (adj[u2] >> v2) & 1:
                        continue
                        
                    remove_edge(u1, v1, adj, edge_list)
                    
                    if can_add(u2, v2, adj):
                        add_edge(u2, v2, adj, edge_list)
                        found = True
                        break
                    else:
                        add_edge(u1, v1, adj, edge_list)
                
                # Note: If found, edge_list has changed. If not, it's reverted.
                # current_count stays same for swap.

            else:
                # Remove a random edge (exploration)
                if not edge_list:
                    continue
                u, v = rng.choice(edge_list)
                remove_edge(u, v, adj, edge_list)
                current_count -= 1
                greedy_fill(adj, edge_list, rng)
                new_count = len(edge_list)
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                    best_edges = edge_list[:]
            
            T *= cooling
            
        return best_adj, best_edges, best_count

    def verify(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if common_neighbors_count(u, v, adj) > 2:
                    return False
        return True

    # Multiple restarts with different strategies
    best_edges = []
    best_count = 0
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333, 444, 555, 666, 777]
    
    # Strategy 1: Start empty, greedy fill
    # Strategy 2: Start with a cycle, greedy fill
    # Strategy 3: Start with random sparse graph, greedy fill
    
    for seed in seeds:
        if time.time() - start_time > time_limit:
            break

        rng = random.Random(seed)
        adj = [0] * n
        edge_list = []
        
        # Initialization strategy based on seed
        if seed % 3 == 0:
            # Empty start
            pass
        elif seed % 3 == 1:
            # Cycle start
            for i in range(n):
                add_edge(i, (i + 1) % n, adj, edge_list)
                add_edge(i, (i + 2) % n, adj, edge_list) # Chord
        else:
            # Random sparse start
            for _ in range(10):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u != v and can_add(u, v, adj):
                    add_edge(u, v, adj, edge_list)

        # Greedy fill
        greedy_fill(adj, edge_list, rng)
        
        # Simulated Annealing
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Allocate time for SA
        sa_time = remaining * 0.8
        best_adj, best_edges_sa, best_count_sa = simulated_annealing(adj, edge_list, rng, sa_time)
        
        # Update best if SA found something better
        if best_count_sa > best_count:
            best_count = best_count_sa
            best_edges = [[u, v] for u, v in best_edges_sa]
            # Verify
            if verify(best_adj):
                pass
            else:
                # If verification fails, discard
                best_count = 0
                best_edges = []

    # Fallback if no valid graph found (should not happen with empty start + greedy)
    if not best_edges:
        # Construct a valid graph: 5 disjoint K4s
        # Vertices 0-3, 4-7, 8-11, 12-15, 16-19
        edges = []
        for k in range(5):
            base = k * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return edges

    return best_edges

if __name__ == "__main__":
    # Quick self-test
    edges = construct()
    print(f"Found {len(edges)} edges")
    # Verify
    adj = [0] * 20
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    for u in range(20):
        for v in range(u + 1, 20):
            if (adj[u] & adj[v]).bit_count() > 2:
                print("Invalid graph!")
                break
