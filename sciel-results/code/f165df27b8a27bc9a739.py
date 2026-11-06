import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    
    # Fast C6 check using bitsets
    # adj_bits[i] is a 64-bit integer representing neighbors of vertex i
    def would_create_c6(u, v, adj_bits):
        # Check if there's a path of length 5 from u to v: u-a-b-c-d-v
        # N1 = neighbors of u (excluding v)
        n1 = adj_bits[u] & ~(1 << v)
        if n1 == 0:
            return False
        
        # N2 = neighbors of N1 (excluding u and v), but we need to be careful about simplicity
        # Actually, let's use the standard approach:
        # A C6 exists iff there's a simple path u-a-b-c-d-v
        # This is equivalent to: there exists a in N(u), b in N(a)\{u,v}, c in N(b)\{u,v,a}, 
        # d in N(c)\{u,v,a,b} such that v in N(d)
        
        # More efficient: 
        # N2 = {b : b is adjacent to some a in N(u), b not in {u,v}}
        # N3 = {c : c is adjacent to some b in N2, c not in {u,v} and c not adjacent to u via a path that would create a shorter cycle}
        
        # Let's use the intersection method:
        # N1 = N(u) \ {v}
        # N2 = (union of N(a) for a in N1) \ {u, v}
        # N3 = (union of N(b) for b in N2) \ {u, v}
        # N4 = (union of N(c) for c in N3) \ {u, v}
        # Check if N4 intersects N(v)
        
        # But we need to ensure the path is simple. The above overcounts.
        
        # Better approach: use BFS with depth limit and visited set
        # Since n=40, we can use a 64-bit mask for visited
        
        # Start from u, find all nodes reachable in exactly 4 steps with a simple path
        # Then check if any of those nodes is adjacent to v
        
        # Let's do iterative deepening with bitmask
        # Depth 0: just u
        # We want to reach depth 4, then check adjacency to v
        
        # Use a list of (node, visited_mask) pairs, but that's expensive
        # Instead, let's use the fact that the path length is only 5
        
        # Alternative: enumerate all simple paths of length 4 from u
        # For each such path ending at d, check if d is adjacent to v
        
        # N1 = N(u)
        current_nodes = []
        for a in range(n):
            if (adj_bits[u] >> a) & 1 and a != v:
                current_nodes.append((a, 1 << u))
        
        # Depth 1: nodes at distance 1 from u (excluding v)
        # Expand to depth 2
        next_nodes = []
        for a, mask in current_nodes:
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if (adj_bits[a] >> b) & 1:
                    next_nodes.append((b, mask | (1 << a)))
        
        # Depth 2
        current_nodes = next_nodes
        next_nodes = []
        for b, mask in current_nodes:
            for c in range(n):
                if (mask >> c) & 1:
                    continue
                if c == v:
                    continue
                if (adj_bits[b] >> c) & 1:
                    next_nodes.append((c, mask | (1 << b)))
        
        # Depth 3
        current_nodes = next_nodes
        next_nodes = []
        for c, mask in current_nodes:
            for d in range(n):
                if (mask >> d) & 1:
                    continue
                if d == v:
                    continue
                if (adj_bits[c] >> d) & 1:
                    next_nodes.append((d, mask | (1 << c)))
        
        # Depth 4: check if any d is adjacent to v
        for d, mask in next_nodes:
            if (adj_bits[d] >> v) & 1:
                return True
        
        return False
    
    def greedy_construct(adj_bits, all_edges, order):
        """Greedy add edges in given order, skipping those that create C6"""
        current_edges = []
        for u, v in order:
            if would_create_c6(u, v, adj_bits):
                continue
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            current_edges.append((u, v))
        return current_edges
    
    def local_search(edges, adj_bits, max_iterations=1000):
        """Try to improve by swapping edges"""
        current_edges = list(edges)
        current_count = len(current_edges)
        edge_set = set(current_edges)
        
        # Precompute which edges are present
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding a random non-edge
            for _ in range(50):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or (u, v) in edge_set or (v, u) in edge_set:
                    continue
                
                # Check if adding (u,v) creates C6
                if would_create_c6(u, v, adj_bits):
                    continue
                
                # Try to remove an edge to allow adding (u,v)
                # Actually, if adding doesn't create C6, just add it
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                current_edges.append((u, v))
                edge_set.add((u, v))
                current_count += 1
                improved = True
                break
            
            if not improved:
                # Try removing an edge and adding two others
                if current_edges:
                    rem_idx = random.randint(0, len(current_edges)-1)
                    ru, rv = current_edges[rem_idx]
                    
                    # Temporarily remove
                    adj_bits[ru] &= ~(1 << rv)
                    adj_bits[rv] &= ~(1 << ru)
                    current_edges.pop(rem_idx)
                    edge_set.discard((ru, rv))
                    edge_set.discard((rv, ru))
                    
                    # Try to add two new edges
                    added = 0
                    for _ in range(100):
                        u = random.randint(0, n-1)
                        v = random.randint(0, n-1)
                        if u == v or (u, v) in edge_set or (v, u) in edge_set:
                            continue
                        if would_create_c6(u, v, adj_bits):
                            continue
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        current_edges.append((u, v))
                        edge_set.add((u, v))
                        added += 1
                        if added >= 2:
                            break
                    
                    if added > 1:
                        improved = True
                    else:
                        # Revert: remove the added edges and re-add the removed one
                        for _ in range(added):
                            au, av = current_edges.pop()
                            adj_bits[au] &= ~(1 << av)
                            adj_bits[av] &= ~(1 << au)
                            edge_set.discard((au, av))
                            edge_set.discard((av, au))
                        
                        adj_bits[ru] |= (1 << rv)
                        adj_bits[rv] |= (1 << ru)
                        current_edges.append((ru, rv))
                        edge_set.add((ru, rv))
                        edge_set.add((rv, ru))
        
        return current_edges, current_count
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = []
    best_count = 0
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        random.shuffle(all_edges)
        
        adj_bits = [0] * n
        current_edges = greedy_construct(adj_bits, all_edges, all_edges)
        current_count = len(current_edges)
        
        # Local search to improve
        if time.time() < deadline - 5:
            improved_edges, improved_count = local_search(current_edges, adj_bits, max_iterations=200)
            if improved_count > current_count:
                current_edges = improved_edges
                current_count = improved_count
        
        if current_count > best_count:
            best_count = current_count
            best_edges = current_edges[:]
        
        # Early stop if we reach the known bound
        if best_count >= 105:
            break
    
    # Convert to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
