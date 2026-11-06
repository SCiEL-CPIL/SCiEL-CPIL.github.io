import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin under 120s limit
    
    n = 40
    
    # --- C6 Checking Logic (Fast Bitmask) ---
    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u,v) to the graph represented by adj creates a C6.
        Returns True if safe, False if C6 created.
        """
        # Neighbors of u excluding v
        nu_mask = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv_mask = adj[v] & ~(1 << u)
        
        # We need to check if there is a path of length 3 between u and v
        # u - a - b - c - d - v
        # a in N(u), d in N(v), a != d
        # b in N(a) \ {u,v,d}, c in N(d) \ {u,v,a}
        # b ~ c
        
        a = nu_mask
        while a:
            lb_a = a & (-a)
            ai = lb_a.bit_length() - 1
            a ^= lb_a
            
            d = nv_mask & ~lb_a  # d != a
            while d:
                ld_d = d & (-d)
                di = ld_d.bit_length() - 1
                d ^= ld_d
                
                # Candidates for b: neighbors of a, excluding u, v, d, a
                b_candidates = adj[ai] & ~(1 << u) & ~(1 << v) & ~(1 << di) & ~(1 << ai)
                
                # Candidates for c: neighbors of d, excluding u, v, a, d
                c_candidates = adj[di] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << di)
                
                # Check if any neighbor of any b in b_candidates is in c_candidates
                # Let U = union of neighbors of all b in b_candidates
                # If U & c_candidates is non-empty, then a C6 exists.
                
                union_neighbors = 0
                bm = b_candidates
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    union_neighbors |= adj[bi]
                    
                    # Early exit if we already found a connection
                    if union_neighbors & c_candidates:
                        return False
                        
        return True

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    # --- Bipartite Decomposition Strategy ---
    # Partition vertices into two sets A and B (size 20 each)
    # G1: edges within A and within B (intra-partition)
    # G2: edges between A and B (inter-partition)
    # C6-free constraint can be checked more efficiently if we decouple
    
    # Create random bipartite initialization
    def create_bipartite_init(seed):
        random.seed(seed)
        adj = [0] * n
        
        # Partition: first 20 in A, last 20 in B
        A = list(range(20))
        B = list(range(20, 40))
        
        # G1: Random edges within A and within B (sparse to avoid C6)
        # Start with a few random edges
        for _ in range(30):  # 30 random intra-partition edges
            if random.random() < 0.5:
                u = random.choice(A)
                v = random.choice(A)
                if u != v and not ((adj[u] >> v) & 1):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                u = random.choice(B)
                v = random.choice(B)
                if u != v and not ((adj[u] >> v) & 1):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # G2: Dense random bipartite edges between A and B
        # Start with ~50% density
        for u in A:
            for v in B:
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Remove edges that create C6
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        for u, v in all_edges:
            if (adj[u] >> v) & 1:
                # Check if removing this edge is necessary
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                if not can_add_edge(adj, u, v):
                    # Edge creates C6, leave it removed
                    pass
                else:
                    # Edge is safe, re-add it
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_adj = None
    best_count = 0
    
    # Multiple restarts with different bipartite initializations
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Create bipartite initialization
        current_adj = create_bipartite_init(10000 + restart * 137)
        current_edges = count_edges(current_adj)
        
        # --- Phase 1: Greedy Addition ---
        candidate_edges = []
        for u, v in all_edges:
            if not ((current_adj[u] >> v) & 1):
                candidate_edges.append((u, v))
                
        random.shuffle(candidate_edges)
        
        for u, v in candidate_edges:
            if time.time() > deadline - 10:
                break
            if not ((current_adj[u] >> v) & 1):
                if can_add_edge(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    
        # --- Phase 2: Local Search (Edge Swaps) ---
        existing_edges = []
        non_existing_edges = []
        for u, v in all_edges:
            if (current_adj[u] >> v) & 1:
                existing_edges.append((u, v))
            else:
                non_existing_edges.append((u, v))
                
        T = 5.0
        cooling_rate = 0.99
        min_T = 0.5
        max_sa_steps = 500
        
        steps = 0
        while steps < max_sa_steps and time.time() < deadline - 5:
            steps += 1
            if not existing_edges or not non_existing_edges:
                break
                
            idx_rem = random.randint(0, len(existing_edges) - 1)
            u1, v1 = existing_edges[idx_rem]
            
            idx_add = random.randint(0, len(non_existing_edges) - 1)
            u2, v2 = non_existing_edges[idx_add]
            
            if (u1, v1) == (u2, v2):
                continue
                
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            
            if can_add_edge(current_adj, u2, v2):
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                
                existing_edges[idx_rem] = (u2, v2)
                non_existing_edges[idx_add] = (u1, v1)
                
                if steps % 10 == 0:
                    if non_existing_edges:
                        idx_try = random.randint(0, len(non_existing_edges) - 1)
                        u_try, v_try = non_existing_edges[idx_try]
                        if can_add_edge(current_adj, u_try, v_try):
                            current_adj[u_try] |= (1 << v_try)
                            current_adj[v_try] |= (1 << u_try)
                            current_edges += 1
                            
                            existing_edges = []
                            non_existing_edges = []
                            for u, v in all_edges:
                                if (current_adj[u] >> v) & 1:
                                    existing_edges.append((u, v))
                                else:
                                    non_existing_edges.append((u, v))
                                    
                            if current_edges > best_count:
                                best_count = current_edges
                                best_adj = current_adj[:]
                                if best_count >= 105:
                                    return [[u, v] for u in range(n) for v in range(u + 1, n) if (best_adj[u] >> v) & 1]
            else:
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                
            T *= cooling_rate
            if T < min_T:
                T = min_T
                
        if current_edges > best_count:
            best_count = current_edges
            best_adj = current_adj[:]
            if best_count >= 105:
                return [[u, v] for u in range(n) for v in range(u + 1, n) if (best_adj[u] >> v) & 1]
        
    # Return the best graph found
    if best_adj is None:
        # Fallback to base
        best_adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v])
                
    return result

# Helper for quick testing (not used in construct)
if __name__ == "__main__":
    edges = construct()
    print(f"Edges found: {len(edges)}")
