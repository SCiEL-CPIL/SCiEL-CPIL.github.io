import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, safety margin
    
    # 1. Generate the best Grötzsch blow-up
    # Grötzsch Graph Edges (11 vertices)
    groetzsch_edges = []
    # Cycle 0-1-2-3-4-0
    cycle_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    groetzsch_edges.extend(cycle_edges)
    
    # Mycielski construction
    neighbors = {
        0: [1, 4],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 0]
    }
    
    for v in range(5):
        v_prime = 5 + v
        for u in neighbors[v]:
            groetzsch_edges.append((v_prime, u))
            
    # Root 10 connected to all copies 5..9
    for v in range(5):
        groetzsch_edges.append((10, 5 + v))
        
    # Find optimal sizes for 16 vertices
    def generate_compositions(n, k):
        if k == 1:
            yield [n]
        else:
            for i in range(1, n - k + 2):
                for tail in generate_compositions(n - i, k - 1):
                    yield [i] + tail
                    
    best_edges_count = -1
    best_sizes = None
    
    for comp in generate_compositions(16, 11):
        comp_sorted = sorted(comp, reverse=True)
        s = [0] * 11
        s[10] = comp_sorted[0]
        for i in range(5):
            s[5+i] = comp_sorted[1+i]
        for i in range(5):
            s[i] = comp_sorted[6+i]
            
        edge_count = 0
        for u, v in groetzsch_edges:
            edge_count += s[u] * s[v]
            
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_sizes = s[:]
            
    # Construct the 16-vertex graph using bitmasks for efficiency
    adj = [0] * 16  # adj[u] is a bitmask of neighbors of u
    
    offsets = [0] * 12
    for i in range(11):
        offsets[i+1] = offsets[i] + best_sizes[i]
        
    for u, v in groetzsch_edges:
        for i in range(best_sizes[u]):
            u_node = offsets[u] + i
            for j in range(best_sizes[v]):
                v_node = offsets[v] + j
                adj[u_node] |= (1 << v_node)
                adj[v_node] |= (1 << u_node)
                
    # 2. Optimized Greedy Edge Addition with bitmask operations
    # Precompute common neighbors for fast triangle check
    def can_add_edge(u, v, adj):
        # Adding (u,v) creates a triangle iff N(u) ∩ N(v) ≠ ∅
        return (adj[u] & adj[v]) == 0
    
    def edge_count_from_adj(adj):
        count = 0
        for u in range(16):
            count += bin(adj[u]).count('1')
        return count // 2
    
    best_adj = [x for x in adj]
    best_count = edge_count_from_adj(adj)
    
    # Try multiple greedy passes with different orderings
    import random
    rng = random.Random(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        work_adj = [x for x in best_adj]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle edge order for diversity
            pairs = [(u, v) for u in range(16) for v in range(u+1, 16) if not (work_adj[u] >> v) & 1]
            rng.shuffle(pairs)
            for u, v in pairs:
                if time.time() > deadline:
                    break
                if not (work_adj[u] >> v) & 1:
                    if can_add_edge(u, v, work_adj):
                        work_adj[u] |= (1 << v)
                        work_adj[v] |= (1 << u)
                        changed = True
        
        count = edge_count_from_adj(work_adj)
        if count > best_count:
            best_count = count
            best_adj = work_adj[:]
        
        # If no change, break early
        if count == best_count and not changed:
            break
    
    # 3. Final polish: try one more pass on the best found
    if time.time() < deadline:
        work_adj = [x for x in best_adj]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(16):
                for v in range(u+1, 16):
                    if not (work_adj[u] >> v) & 1:
                        if can_add_edge(u, v, work_adj):
                            work_adj[u] |= (1 << v)
                            work_adj[v] |= (1 << u)
                            changed = True
        count = edge_count_from_adj(work_adj)
        if count > best_count:
            best_count = count
            best_adj = work_adj[:]
    
    # 4. Return edges
    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
