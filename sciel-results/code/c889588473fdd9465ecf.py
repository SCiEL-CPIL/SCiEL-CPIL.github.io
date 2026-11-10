import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 20
    
    def popcount(x):
        return bin(x).count('1')
    
    def common_neighbors(adj, u, v):
        return adj[u] & adj[v]
    
    def is_k23_free(adj):
        # Check all pairs have at most 2 common neighbors
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) > 2:
                    return False
        return True
    
    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                lsb = v_mask & (-v_mask)
                v_mask ^= lsb
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
        return edges
    
    def spectral_construction():
        """
        Use quadratic residues in GF(19) to build a Paley-like graph.
        GF(19): QR = {1, 4, 5, 6, 7, 9, 11, 16, 17}
        Map vertex i (0..18) to field element i, vertex 19 is "infinity".
        Edge (i, j) for i,j in 0..18 if (j-i) mod 19 is a QR.
        Vertex 19 connects to a subset to maintain K2,3-free.
        """
        # Quadratic residues mod 19
        qr = set()
        for x in range(1, 19):
            qr.add((x * x) % 19)
        # qr = {1, 4, 5, 6, 7, 9, 11, 16, 17}
        
        adj = [0] * n
        
        # Build Paley graph on vertices 0..18
        for i in range(19):
            for j in range(i + 1, 19):
                diff = (j - i) % 19
                if diff in qr:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Vertex 19: connect to vertices to balance degree
        # Try connecting to a set S such that K2,3-free is maintained
        # Start with empty, try adding
        for v in range(19):
            # Check if adding (19, v) keeps K2,3-free
            # Pair (19, v): common neighbors = N(v) ∩ N(19) = N(v) ∩ current_N(19)
            # Since N(19) is empty initially, common = 0, always OK
            # But also check pairs (19, w) for w in N(19) after adding
            # and pairs (v, w) for w in N(19)
            # Since N(19) is empty before adding, only need to check pair (19,v)
            # which has 0 common neighbors. So always safe to add.
            adj[19] |= (1 << v)
            adj[v] |= (1 << 19)
        
        # Now we have a complete graph on 19 vertices plus vertex 19 connected to all.
        # This is K20, which is NOT K2,3-free. We need to remove edges.
        
        # Better approach: start with Paley(19) and add vertex 19 carefully
        adj = [0] * n
        for i in range(19):
            for j in range(i + 1, 19):
                diff = (j - i) % 19
                if diff in qr:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Now try to add edges from vertex 19
        # For each v in 0..18, check if (19, v) can be added
        for v in range(19):
            # Pair (19, v): common = N(v) ∩ N(19). N(19) is currently empty, so OK.
            # But after adding, for any w in N(19), pair (19, w) and pair (v, w) need checking.
            # Since we're adding one at a time, check incrementally.
            # Check pair (19, v): common = N(v) & N(19)
            if popcount(adj[v] & adj[19]) > 2:
                continue
            # Check for all w in N(v): pair (19, w) gains v as common neighbor
            # So common(19, w) must be <= 1 before adding
            w = adj[v]
            ok = True
            while w:
                lsb = w & (-w)
                w ^= lsb
                wi = lsb.bit_length() - 1
                if wi == 19:
                    continue
                if popcount(adj[19] & adj[wi]) > 1:
                    ok = False
                    break
            if ok:
                # Check for all w in N(19): pair (v, w) gains 19 as common neighbor
                w = adj[19]
                while w:
                    lsb = w & (-w)
                    w ^= lsb
                    wi = lsb.bit_length() - 1
                    if wi == v:
                        continue
                    if popcount(adj[v] & adj[wi]) > 1:
                        ok = False
                        break
            if ok:
                adj[19] |= (1 << v)
                adj[v] |= (1 << 19)
        
        return adj
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if (adj[u] >> v) & 1:
            return False
        
        # 1. Pair (u,v): common neighbors must be <= 2
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        # 2. For all w in N(v), pair (u,w) gains v as common neighbor
        w = adj[v]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if wi == u:
                continue
            if popcount(adj[u] & adj[wi]) > 1:
                return False
        
        # 3. For all w in N(u), pair (v,w) gains u as common neighbor
        w = adj[u]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if wi == v:
                continue
            if popcount(adj[v] & adj[wi]) > 1:
                return False
        
        return True
    
    def coordinate_ascent(adj, max_time=10.0):
        """
        Coordinate ascent: try to flip individual edges (add or remove) to improve.
        Also try 2-opt swaps: remove edge (a,b), add edge (c,d).
        """
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        end_time = time.time() + max_time
        no_improve_count = 0
        
        while time.time() < end_time:
            improved = False
            
            # Phase 1: Try adding missing edges
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > end_time:
                        break
                    if (current_adj[u] >> v) & 1:
                        continue
                    if can_add_edge(current_adj, u, v):
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        current_edges += 1
                        improved = True
                if time.time() > end_time:
                    break
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = current_adj[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count > 5:
                break
            
            # Phase 2: Try 2-opt swaps (remove one edge, add another)
            if time.time() > end_time:
                break
            
            # Collect existing edges
            existing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (current_adj[u] >> v) & 1:
                        existing.append((u, v))
            
            # Try random swaps
            attempts = 0
            while attempts < 200 and time.time() < end_time:
                attempts += 1
                if not existing:
                    break
                u1, v1 = random.choice(existing)
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                if u2 > v2:
                    u2, v2 = v2, u2
                if (u1, v1) == (u2, v2):
                    continue
                if (current_adj[u2] >> v2) & 1:
                    continue
                
                # Remove (u1, v1)
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)
                
                # Try to add (u2, v2)
                if can_add_edge(current_adj, u2, v2):
                    current_adj[u2] |= (1 << v2)
                    current_adj[v2] |= (1 << u2)
                    # Edge count unchanged
                    # Accept swap to explore different structures
                    if random.random() < 0.3:
                        # Keep the swap
                        pass
                    else:
                        # Revert
                        current_adj[u2] &= ~(1 << v2)
                        current_adj[v2] &= ~(1 << u2)
                        current_adj[u1] |= (1 << v1)
                        current_adj[v1] |= (1 << u1)
                else:
                    # Revert
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
            
            # Phase 3: Random restart perturbation
            if time.time() > end_time:
                break
            
            if random.random() < 0.1:
                # Perturb: remove a few random edges
                for _ in range(3):
                    if not existing:
                        break
                    u1, v1 = random.choice(existing)
                    current_adj[u1] &= ~(1 << v1)
                    current_adj[v1] &= ~(1 << u1)
                    current_edges -= 1
                    # Update existing
                    existing = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (current_adj[u] >> v) & 1:
                                existing.append((u, v))
        
        return best_adj, best_edges
    
    def greedy_fill(adj):
        """Greedily add all possible edges in random order."""
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    # Multi-start: combine spectral construction with random starts
    best_adj = [0] * n
    best_edges = 0
    
    # Start 1: Spectral construction
    adj1 = spectral_construction()
    adj1 = greedy_fill(adj1)
    adj1, e1 = coordinate_ascent(adj1, max_time=8.0)
    if e1 > best_edges:
        best_edges = e1
        best_adj = adj1[:]
    
    if time.time() > deadline:
        return adj_to_edges(best_adj)
    
    # Start 2: Random greedy
    adj2 = [0] * n
    adj2 = greedy_fill(adj2)
    adj2, e2 = coordinate_ascent(adj2, max_time=8.0)
    if e2 > best_edges:
        best_edges = e2
        best_adj = adj2[:]
    
    if time.time() > deadline:
        return adj_to_edges(best_adj)
    
    # Start 3: Another random greedy with different seed
    adj3 = [0] * n
    adj3 = greedy_fill(adj3)
    adj3, e3 = coordinate_ascent(adj3, max_time=8.0)
    if e3 > best_edges:
        best_edges = e3
        best_adj = adj3[:]
    
    if time.time() > deadline:
        return adj_to_edges(best_adj)
    
    # Final polish on best
    best_adj, _ = coordinate_ascent(best_adj, max_time=5.0)
    
    return adj_to_edges(best_adj)
