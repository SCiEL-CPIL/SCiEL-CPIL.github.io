import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = 18
    
    # Helper: Check if a graph (list of edges) is K4-free
    def is_k4_free(edges):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check all 4-cliques
        # Optimized: Iterate triangles, check if any common neighbor
        # Or just iterate all 4-subsets
        # C(18, 4) = 3060. Very fast.
        for quad in itertools.combinations(range(n), 4):
            u, v, w, x = quad
            if (v in adj[u] and w in adj[u] and x in adj[u] and
                w in adj[v] and x in adj[v] and
                x in adj[w]):
                return False
        return True

    # Helper: Find max triangle-free induced subgraph size
    def max_triangle_free_size(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Precompute triangles
        # A triangle is a set of 3 vertices {u,v,w} such that all pairs are edges
        triangles = []
        for u in range(n):
            # v > u
            vu = adj[u] & ~((1 << (u + 1)) - 1) # only bits > u
            # Actually, simpler to just iterate all triples
            # C(18,3) = 816
            pass
            
        # Let's just iterate all subsets of size k from 18 down to 1?
        # Or use a recursive search on vertices.
        
        # Recursive B&B
        best = 0
        
        # Sort vertices by degree? Maybe not necessary for n=18
        
        def search(current_mask, candidates_mask, count):
            nonlocal best
            if count + bin(candidates_mask).count('1') <= best:
                return
            
            # If candidates is empty
            if candidates_mask == 0:
                if count > best:
                    best = count
                return
            
            # Pick a vertex from candidates
            # Heuristic: pick vertex with highest degree in the induced subgraph?
            # Or just the first one.
            # Let's pick the lowest set bit
            v = (candidates_mask & -candidates_mask).bit_length() - 1
            candidates_mask &= ~(1 << v)
            
            # Option 1: Include v
            # Check if including v creates a triangle with existing vertices in current_mask?
            # No, current_mask is the set of ALREADY SELECTED vertices.
            # We need to ensure the NEW set (current_mask | {v}) is triangle-free.
            # This means v must not form a triangle with any two vertices in current_mask.
            # i.e., for any u, w in current_mask, if (u,v) and (w,v) and (u,w) are edges, then invalid.
            
            # Efficient check:
            # Neighbors of v in current_mask: N_v = adj[v] & current_mask
            # If N_v contains an edge, then invalid.
            # Check if N_v has an edge:
            # For each u in N_v, if adj[u] & N_v is not 0, then there is an edge.
            
            valid = True
            nv = adj[v] & current_mask
            # Check if subgraph induced by nv has any edge
            # Iterate bits in nv
            temp = nv
            while temp:
                u_bit = temp & -temp
                u = u_bit.bit_length() - 1
                if adj[u] & nv:
                    valid = False
                    break
                temp ^= u_bit
            
            if valid:
                search(current_mask | (1 << v), candidates_mask, count + 1)
            
            # Option 2: Exclude v
            search(current_mask, candidates_mask, count)

        # Initial call
        # candidates = all vertices
        all_mask = (1 << n) - 1
        search(0, all_mask, 0)
        
        return best

    # Generate Paley(17) + Isolated
    def get_paley_edges():
        p = 17
        edges = []
        # Quadratic residues mod 17
        residues = set()
        for x in range(p):
            residues.add((x * x) % p)
        
        for u in range(p):
            for v in range(u + 1, p):
                if (v - u) % p in residues or (u - v) % p in residues:
                    edges.append([u, v])
        # Vertex 17 is isolated, so no edges added
        return edges

    # Initial Graph
    edges = get_paley_edges()
    
    # Verify K4-free
    if not is_k4_free(edges):
        # Fallback to complete tripartite if Paley fails (it shouldn't)
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]

    current_score = max_triangle_free_size(edges)
    best_edges = edges[:]
    best_score = current_score
    
    # Local Search
    # Randomize edge order for flipping?
    # We can try flipping each edge and see if it improves.
    # Or random sampling.
    
    # To make it faster, we can pre-compute the effect of an edge flip on triangles?
    # That's complex. Given n=18, full re-evaluation is ~816 * (small cost).
    # Let's estimate cost of max_triangle_free_size.
    # It's a B&B. Worst case exponential, but for n=18 and sparse/dense graphs, it's usually fast.
    
    while time.time() < deadline:
        # Pick a random edge to flip
        # Or iterate through all edges?
        # Let's do a mix: try random edges.
        
        # Ensure we have a list of edges to flip
        # We can maintain a set of edges for O(1) check
        edge_set = set(map(tuple, edges))
        
        improved = False
        # Try a few random flips
        for _ in range(100):
            if time.time() > deadline:
                break
                
            # Pick random u, v
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Check if edge exists
            if (u, v) in edge_set or (v, u) in edge_set:
                # Remove edge
                if (u, v) in edge_set:
                    edge_set.remove((u, v))
                else:
                    edge_set.remove((v, u))
                new_edges = list(edge_set)
            else:
                # Add edge
                edge_set.add((u, v))
                new_edges = list(edge_set)
            
            # Check K4-free
            # Optimization: Only check K4s involving u or v
            # But is_k4_free is fast (3060 checks).
            if not is_k4_free(new_edges):
                # Revert
                if (u, v) in edge_set:
                    edge_set.remove((u, v))
                else:
                    edge_set.remove((v, u))
                continue
            
            # Evaluate
            new_score = max_triangle_free_size(new_edges)
            
            if new_score < best_score:
                best_score = new_score
                best_edges = new_edges
                current_score = new_score
                improved = True
            else:
                # Revert
                if (u, v) in edge_set:
                    edge_set.remove((u, v))
                else:
                    edge_set.remove((v, u))
        
        # If no improvement in 100 random tries, try systematic?
        # Or just continue random.
        
    return best_edges

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = 18
    
    # Helper: Check if a graph (list of edges) is K4-free
    def is_k4_free(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for quad in itertools.combinations(range(n), 4):
            u, v, w, x = quad
            if (v in adj[u] and w in adj[u] and x in adj[u] and
                w in adj[v] and x in adj[v] and
                x in adj[w]):
                return False
        return True

    # Helper: Find max triangle-free induced subgraph size
    def max_triangle_free_size(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        best = 0
        
        def search(current_mask, candidates_mask, count):
            nonlocal best
            if count + bin(candidates_mask).count('1') <= best:
                return
            
            if candidates_mask == 0:
                if count > best:
                    best = count
                return
            
            v = (candidates_mask & -candidates_mask).bit_length() - 1
            candidates_mask &= ~(1 << v)
            
            valid = True
            nv = adj[v] & current_mask
            temp = nv
            while temp:
                u_bit = temp & -temp
                u = u_bit.bit_length() - 1
                if adj[u] & nv:
                    valid = False
                    break
                temp ^= u_bit
            
            if valid:
                search(current_mask | (1 << v), candidates_mask, count + 1)
            
            search(current_mask, candidates_mask, count)

        all_mask = (1 << n) - 1
        search(0, all_mask, 0)
        
        return best

    def get_paley_edges():
        p = 17
        edges = []
        residues = set()
        for x in range(p):
            residues.add((x * x) % p)
        
        for u in range(p):
            for v in range(u + 1, p):
                if (v - u) % p in residues:
                    edges.append([u, v])
        return edges

    edges = get_paley_edges()
    
    if not is_k4_free(edges):
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]

    best_edges = edges[:]
    best_score = max_triangle_free_size(edges)
    
    edge_set = set(map(tuple, edges))
    
    while time.time() < deadline:
        improved = False
        for _ in range(200):
            if time.time() > deadline:
                break
                
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            if (u, v) in edge_set:
                edge_set.remove((u, v))
                new_edges = list(edge_set)
            elif (v, u) in edge_set:
                edge_set.remove((v, u))
                new_edges = list(edge_set)
            else:
                edge_set.add((u, v))
                new_edges = list(edge_set)
            
            if not is_k4_free(new_edges):
                if (u, v) in edge_set:
                    edge_set.remove((u, v))
                elif (v, u) in edge_set:
                    edge_set.remove((v, u))
                continue
            
            new_score = max_triangle_free_size(new_edges)
            
            if new_score < best_score:
                best_score = new_score
                best_edges = new_edges
                improved = True
            else:
                if (u, v) in edge_set:
                    edge_set.remove((u, v))
                elif (v, u) in edge_set:
                    edge_set.remove((v, u))
        
        if not improved:
            # Try a different strategy if stuck?
            # Maybe restart with a different seed?
            # For now, just continue.
            pass

    return best_edges
# EVOLVE-BLOCK-END
