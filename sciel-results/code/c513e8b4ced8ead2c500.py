import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    def largest_tf(adj, time_limit=None):
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy Lower Bound: build a triangle-free set greedily
        # A vertex v can be added to set S if v has no neighbors in S
        greedy_set = 0
        remaining = full_mask
        while remaining:
            # Pick vertex with minimum degree in remaining
            min_deg = n + 1
            chosen = -1
            c = remaining
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg = bin(adj[vid] & remaining).count('1')
                if deg < min_deg:
                    min_deg = deg
                    chosen = vid
                c ^= lsb
            
            if chosen >= 0:
                # Check if chosen has no neighbors in greedy_set
                if not (adj[chosen] & greedy_set):
                    greedy_set |= (1 << chosen)
                remaining &= ~(1 << chosen)
        
        best[0] = bin(greedy_set).count('1')
        
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Select vertex with max degree in C
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = bin(adj[vid] & C_mask).count('1')
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            # First try not including v
            bnb(S_mask, C_mask & ~v_bit)
            
            # Then try including v if it doesn't create a triangle
            if not (adj[v] & S_mask):
                bnb(S_mask | v_bit, C_mask & ~v_bit)
                
        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def build_turan_3_partite(part_sizes):
        """Build a 3-partite Turán graph where each part is a clique.
        Vertices in different parts are NOT connected (independent between parts).
        This ensures K4-free since any 4 vertices must have 2 in same part (clique)
        or 3 in same part, but with 3 parts by pigeonhole, 4 vertices have 2 in same part.
        Actually for K4-free: a K4 would need 4 vertices with all edges. In 3-partite
        with cliques within parts, edges only exist within parts. So a K4 must be entirely
        within one part. If each part has size <= 3, no K4 exists.
        """
        adj = [0] * n
        vertices = list(range(n))
        part_starts = [0]
        for s in part_sizes:
            part_starts.append(part_starts[-1] + s)
        
        for p in range(3):
            start = part_starts[p]
            end = part_starts[p + 1]
            for i in range(start, end):
                for j in range(i + 1, end):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def get_part_assignment(part_sizes):
        """Return list where part_assignment[v] = part index of vertex v."""
        assignment = [0] * n
        idx = 0
        for p, s in enumerate(part_sizes):
            for _ in range(s):
                assignment[idx] = p
                idx += 1
        return assignment

    def swap_vertices_between_parts(part_assignment, part_sizes, u, v):
        """Swap vertices u and v between their parts, updating part_sizes.
        Returns True if swap was made, False if u and v are in same part."""
        pu = part_assignment[u]
        pv = part_assignment[v]
        if pu == pv:
            return False
        part_assignment[u] = pv
        part_assignment[v] = pu
        part_sizes[pu] -= 1
        part_sizes[pv] -= 1
        part_sizes[pu] += 1
        part_sizes[pv] += 1
        return True

    def rebuild_adj_from_parts(part_assignment, part_sizes):
        """Rebuild adjacency from part assignment."""
        adj = [0] * n
        for p in range(3):
            part_vertices = [v for v in range(n) if part_assignment[v] == p]
            for i in range(len(part_vertices)):
                for j in range(i + 1, len(part_vertices)):
                    u = part_vertices[i]
                    v = part_vertices[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    random.seed(42)
    best_adj = None
    best_score = n

    def generate_random_k4_free():
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def generate_c9_union_c9():
        """Construct a K4-free graph using two disjoint C9 cycles with cross edges.
        Each C9 is triangle-free. Add edges between the two cycles carefully
        to maintain K4-free property while reducing max triangle-free subgraph."""
        adj = [0] * n
        # Create two 9-cycles: vertices 0-8 and 9-17
        for i in range(9):
            u = i
            v = (i + 1) % 9
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for i in range(9):
            u = 9 + i
            v = 9 + ((i + 1) % 9)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Add cross edges: connect i in first cycle to (i + offset) % 9 in second cycle
        # Be careful to avoid creating K4
        for i in range(9):
            for j in range(9):
                # Only add if it doesn't create a K4
                u = i
                v = 9 + j
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Try multiple strategies: random greedy and C9 union C9
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        
        # Alternate between random and structured initialization
        if restarts % 3 == 1:
            adj = generate_random_k4_free()
        elif restarts % 3 == 2:
            adj = generate_c9_union_c9()
        else:
            adj = generate_random_k4_free()
        
        # Initial Evaluation
        score = largest_tf(adj, time_limit=deadline - 5.0)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"Restart {restarts}: New best {best_score}")
            
        # Local Search: try adding and removing edges
        improved = True
        iterations = 0
        while improved and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            
            # Try adding edges
            random.shuffle(all_edges)
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                        new_score = largest_tf(adj, time_limit=deadline - 5.0)
                        
                        if new_score < score:
                            score = new_score
                            improved = True
                            if score < best_score:
                                best_score = score
                                best_adj = adj[:]
                                print(f"  Local Search Add Edge: {best_score}")
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                
                if time.time() > deadline - 5.0:
                    break
            
            if time.time() > deadline - 5.0:
                break
            
            # Try removing edges (sometimes removing an edge can help by allowing
            # other edges to be added later that reduce the triangle-free number)
            random.shuffle(all_edges)
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    new_score = largest_tf(adj, time_limit=deadline - 5.0)
                    
                    if new_score < score:
                        score = new_score
                        improved = True
                        if score < best_score:
                            best_score = score
                            best_adj = adj[:]
                            print(f"  Local Search Remove Edge: {best_score}")
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                
                if time.time() > deadline - 5.0:
                    break
        
        if time.time() > deadline - 10.0:
            break

    if best_adj is None:
        # Fallback
        best_adj = generate_random_k4_free()

    edges = get_edge_list(best_adj)
    return edges
