import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    n = 18
    full_mask = (1 << n) - 1
    
    def is_independent(mask, adj):
        """Check if vertices in mask form an independent set."""
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)
    
    def largest_tf(adj, time_limit=None):
        """Exact branch and bound for largest triangle-free induced subgraph."""
        best = [0]
        
        # Greedy lower bound to start with
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        greedy_set = 0
        for v in order:
            if is_independent(adj[v] & greedy_set, adj):
                greedy_set |= (1 << v)
        best[0] = bin(greedy_set).count('1')
        
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            # Upper bound pruning
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Select vertex with max degree in C for branching
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
            
            # Branch 1: Exclude v
            bnb(S_mask, C_mask & ~v_bit)
            
            # Branch 2: Include v (if valid)
            if is_independent(adj[v] & S_mask, adj):
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
    
    def build_turan_3partite():
        """Build a 3-partite Turan graph T_3(18) with 6 vertices per part."""
        adj = [0] * n
        # Part 0: vertices 0-5, Part 1: vertices 6-11, Part 2: vertices 12-17
        parts = [[0,1,2,3,4,5], [6,7,8,9,10,11], [12,13,14,15,16,17]]
        # Within each part: clique (all edges)
        for part in parts:
            for i in range(len(part)):
                for j in range(i+1, len(part)):
                    u, v = part[i], part[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Between parts: complete bipartite (all edges)
        for i in range(3):
            for j in range(i+1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj
    
    def build_c9_cycles():
        """Build two disjoint C9 cycles (guaranteed score of 8)."""
        adj = [0] * n
        # First C9: vertices 0-8
        for i in range(9):
            u, v = i, (i+1) % 9
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Second C9: vertices 9-17
        for i in range(9):
            u, v = 9+i, 9+((i+1) % 9)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def add_random_edges(adj, count=20):
        """Add random edges while maintaining K4-free property."""
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        added = 0
        for u, v in all_edges:
            if added >= count:
                break
            if not ((adj[u] >> v) & 1):
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added += 1
        return adj
    
    def local_search_add_edges(adj, time_limit):
        """Try to add edges to reduce triangle-free number."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if can_add_edge(adj, u, v):
                        # Tentatively add
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        # Check if score improves (we want smaller)
                        if time.time() < time_limit - 1.0:
                            new_score = largest_tf(adj, time_limit=time_limit)
                            if new_score < best_score[0]:
                                best_score[0] = new_score
                                best_adj = adj[:]
                                improved = True
                            else:
                                # Revert
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                if time.time() > time_limit:
                    break
        return adj
    
    # Initialize best tracking
    best_score = [n]  # Start with worst case
    best_adj = [0] * n
    
    # Strategy 1: Start with 3-partite Turan graph
    random.seed(42)
    
    # Try multiple random restarts with different initial structures
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 20:
        attempts += 1
        
        # Choose initial structure
        if attempts % 3 == 1:
            adj = build_turan_3partite()
        elif attempts % 3 == 2:
            adj = build_c9_cycles()
        else:
            # Random K4-free graph
            adj = [0] * n
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Add some random edges to make it denser
        adj = add_random_edges(adj, count=10)
        
        # Evaluate current score
        if time.time() < deadline - 3.0:
            score = largest_tf(adj, time_limit=deadline - 2.0)
            
            if score < best_score[0]:
                best_score[0] = score
                best_adj = adj[:]
                print(f"New best: {score} at attempt {attempts}, time={time.time()-start_time:.1f}s")
            
            # Local search: try to add more edges
            if time.time() < deadline - 5.0 and score >= best_score[0]:
                adj = local_search_add_edges(adj, deadline - 3.0)
    
    # Final evaluation of best_adj
    if time.time() < deadline - 1.0:
        final_score = largest_tf(best_adj, time_limit=deadline - 0.5)
        if final_score < best_score[0]:
            best_score[0] = final_score
    
    edges = get_edge_list(best_adj)
    print(f"Final: score={best_score[0]}, edges={len(edges)}")
    return edges
