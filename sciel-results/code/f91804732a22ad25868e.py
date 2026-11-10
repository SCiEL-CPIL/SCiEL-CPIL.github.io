import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Define Grötzsch Graph
    # Vertices 0..10
    # 0-4: Cycle
    # 5-9: Copies of 0-4
    # 10: Root
    
    groetzsch_edges = []
    # Cycle
    groetzsch_edges.extend([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    
    # Copies
    # v' (5+v) connected to neighbors of v
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
            
    # Root
    for v in range(5):
        groetzsch_edges.append((10, 5 + v))
        
    # Build adjacency set for base graph for quick lookup
    base_adj = set()
    for u, v in groetzsch_edges:
        base_adj.add((u, v))
        base_adj.add((v, u))
        
    # Precompute common neighbors for non-adjacent pairs
    # If (i, j) not in base_adj and N(i) cap N(j) is empty, we can add edges
    can_add_edges = set()
    for i in range(11):
        for j in range(i + 1, 11):
            if (i, j) not in base_adj:
                # Check common neighbors
                common = False
                for k in range(11):
                    if (i, k) in base_adj and (j, k) in base_adj:
                        common = True
                        break
                if not common:
                    can_add_edges.add((i, j))
                    can_add_edges.add((j, i))

    # 2. Search for optimal blow-up sizes
    # s[0..10] sum to 16, s[i] >= 1
    # Heuristic: Sort sizes descending, assign to vertices sorted by degree descending.
    # Degrees: 10:5, 5-9:3, 0-4:2
    
    def generate_compositions(n, k):
        if k == 1:
            yield [n]
        else:
            for i in range(1, n - k + 2):
                for tail in generate_compositions(n - i, k - 1):
                    yield [i] + tail

    best_edge_count = -1
    best_sizes = None
    
    # Iterate all compositions
    for comp in generate_compositions(16, 11):
        comp_sorted = sorted(comp, reverse=True)
        
        s = [0] * 11
        s[10] = comp_sorted[0]
        for i in range(5):
            s[5+i] = comp_sorted[1+i]
        for i in range(5):
            s[i] = comp_sorted[6+i]
            
        # Calculate edges
        edge_count = 0
        for u, v in groetzsch_edges:
            edge_count += s[u] * s[v]
            
        # Add edges for "far" pairs
        for i, j in can_add_edges:
            if i < j: # Avoid double counting in the set? The set has both (i,j) and (j,i). 
                     # But we only iterate i<j in the generation? 
                     # No, can_add_edges contains both. 
                     # In the loop above, I added both. 
                     # So I should only count once.
                     # Let's just check i < j here.
                     pass 
        
        # Correctly sum the "extra" edges
        for i in range(11):
            for j in range(i + 1, 11):
                if (i, j) in can_add_edges:
                    edge_count += s[i] * s[j]
                    
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_sizes = s[:]
            
    # 3. Construct the final graph
    adj_matrix = [[False] * 16 for _ in range(16)]
    offsets = [0] * 12
    for i in range(11):
        offsets[i+1] = offsets[i] + best_sizes[i]
        
    # Add base blow-up edges
    for u, v in groetzsch_edges:
        for i in range(best_sizes[u]):
            u_node = offsets[u] + i
            for j in range(best_sizes[v]):
                v_node = offsets[v] + j
                adj_matrix[u_node][v_node] = True
                adj_matrix[v_node][u_node] = True
                
    # Add extra edges for "far" pairs
    for i in range(11):
        for j in range(i + 1, 11):
            if (i, j) in can_add_edges:
                for a in range(best_sizes[i]):
                    u_node = offsets[i] + a
                    for b in range(best_sizes[j]):
                        v_node = offsets[j] + b
                        adj_matrix[u_node][v_node] = True
                        adj_matrix[v_node][u_node] = True
                        
    # 4. Greedy Edge Addition (Local Search)
    # Try to add any remaining edge if it doesn't create a triangle
    def count_edges(adj):
        count = 0
        for u in range(16):
            for v in range(u + 1, 16):
                if adj[u][v]:
                    count += 1
        return count
        
    def greedy_add_edges(adj, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            pairs = [(u, v) for u in range(16) for v in range(u + 1, 16) if not adj[u][v]]
            # Sort by potential gain? Or just shuffle?
            # To be deterministic and fast, we can just iterate.
            for u, v in pairs:
                if time.time() >= deadline:
                    break
                if adj[u][v]:
                    continue
                creates_triangle = False
                for w in range(16):
                    if adj[u][w] and adj[w][v]:
                        creates_triangle = True
                        break
                if not creates_triangle:
                    adj[u][v] = True
                    adj[v][u] = True
                    changed = True
        return adj

    # Run greedy
    adj_copy = [row[:] for row in adj_matrix]
    adj_copy = greedy_add_edges(adj_copy, deadline)
    
    # Return edges
    edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            if adj_copy[u][v]:
                edges.append([u, v])
                
    return edges

# Helper for compositions
def generate_compositions(n, k):
    if k == 1:
        yield [n]
    else:
        for i in range(1, n - k + 2):
            for tail in generate_compositions(n - i, k - 1):
                yield [i] + tail
