import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Use bitmasks for adjacency
    # adj_mask[i] is a bitmask where bit j is 1 if i-j is an edge
    
    def is_valid(adj_mask):
        """Check if any equal-degree pair has a length-3 path between them."""
        degs = [bin(adj_mask[i]).count('1') for i in range(n)]
        # Group vertices by degree
        groups = {}
        for i in range(n):
            d = degs[i]
            if d not in groups:
                groups[d] = []
            groups[d].append(i)
        
        for d, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                s = group[i]
                for j in range(i + 1, len(group)):
                    t = group[j]
                    # Check if there's a length-3 path s-u-v-t
                    # u in N(s)\{t}, v in N(t)\{s}, and (u,v) is an edge
                    mask_s = adj_mask[s] & ~(1 << t)
                    mask_t = adj_mask[t] & ~(1 << s)
                    # Check all edges
                    for u in range(n):
                        if not (mask_s >> u) & 1:
                            continue
                        for v in range(n):
                            if u == v:
                                continue
                            if not (mask_t >> v) & 1:
                                continue
                            # Check if (u,v) is an edge
                            if (adj_mask[u] >> v) & 1:
                                return False
                    # If we get here, no violation for this pair
        return True
    
    def count_edges(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    best_edge_count = 0
    best_adj_mask = None
    
    # Try multiple random starts
    for trial in range(500):
        if time.time() > deadline - 5:
            break
        
        random.seed(trial * 7919)
        
        # Initialize with random graph
        adj_mask = [0] * n
        for u, v in all_edges:
            if random.random() < 0.5:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        # Local search: try to add edges
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding each missing edge
            for u, v in all_edges:
                if time.time() > deadline - 5:
                    break
                if (adj_mask[u] >> v) & 1:
                    continue
                # Try adding
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                if is_valid(adj_mask):
                    improved = True
                    # Keep the edge
                    pass
                else:
                    # Remove
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
        
        # Also try removing edges to enable adding others? 
        # For now, just count
        ec = count_edges(adj_mask)
        if is_valid(adj_mask) and ec > best_edge_count:
            best_edge_count = ec
            best_adj_mask = list(adj_mask)
        
        # Try a more aggressive local search: swap edges
        if time.time() > deadline - 10:
            break
        
        # Simulated annealing style
        if best_adj_mask is None:
            adj_mask = [0] * n
        else:
            adj_mask = list(best_adj_mask)
        
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Randomly pick an edge to toggle
            u, v = random.choice(all_edges)
            if (adj_mask[u] >> v) & 1:
                # Remove
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                if is_valid(adj_mask):
                    ec = count_edges(adj_mask)
                    if ec > best_edge_count:
                        best_edge_count = ec
                        best_adj_mask = list(adj_mask)
                else:
                    # Re-add
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
            else:
                # Add
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                if is_valid(adj_mask):
                    ec = count_edges(adj_mask)
                    if ec > best_edge_count:
                        best_edge_count = ec
                        best_adj_mask = list(adj_mask)
                else:
                    # Remove
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
    
    if best_adj_mask is None:
        # Fallback: empty graph
        best_adj_mask = [0] * n
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj_mask[u] >> v) & 1:
                edges.append([int(u), int(v)])
    
    return edges
