import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Q3 structure: vertices labeled 0..7 (binary 000..111)
    # Edges between vertices differing in exactly one bit
    # Bipartition: even parity {0,3,5,6} vs odd parity {1,2,4,7}
    
    # For a given 8-vertex set, try all ways to partition into two 4-sets (A, B),
    # then check if the bipartite graph G[A,B] contains Q3 as a bipartite subgraph.
    
    # Q3 bipartite adjacency (A=even, B=odd):
    # A[0]=000 connects to B: 001,010,100 -> indices 0,1,3 in B={001,010,111,100}
    # A[1]=011 connects to B: 001,010,111 -> indices 0,1,2
    # A[2]=101 connects to B: 001,111,100 -> indices 0,2,3
    # A[3]=110 connects to B: 010,111,100 -> indices 1,2,3
    
    perms24 = list(itertools.permutations(range(4)))
    idx4_of_8 = list(itertools.combinations(range(8), 4))
    
    # Precompute required neighbor index tuples for each permutation
    req_patterns = []
    for p in perms24:
        req = [
            (p[0], p[1], p[3]),
            (p[0], p[1], p[2]),
            (p[0], p[2], p[3]),
            (p[1], p[2], p[3])
        ]
        req_patterns.append(req)
    
    # Precompute for each pair (u,v) the list of 6-tuples of other vertices
    pair_to_6sets = {}
    for u in range(16):
        for v in range(u+1, 16):
            others = [x for x in range(16) if x != u and x != v]
            pair_to_6sets[(u,v)] = list(itertools.combinations(others, 6))
    
    # Helper function to check if a specific 8-vertex set contains Q3
    def contains_q3_in_set(v8, adj_mask):
        """Check if the induced subgraph on v8 contains Q3 as a subgraph."""
        v_list = list(v8)
        
        for (i, j, k, l) in idx4_of_8:
            A = (v_list[i], v_list[j], v_list[k], v_list[l])
            B = tuple(v_list[idx] for idx in range(8) if idx not in (i, j, k, l))
            
            # Check orientation 1: A as even part, B as odd part
            for req in req_patterns:
                ok = True
                for a_idx in range(4):
                    m = adj_mask[A[a_idx]]
                    for b_idx in req[a_idx]:
                        if not (m & (1 << B[b_idx])):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return True
            
            # Check orientation 2: B as even part, A as odd part
            for req in req_patterns:
                ok = True
                for b_idx in range(4):
                    m = adj_mask[B[b_idx]]
                    for a_idx in req[b_idx]:
                        if not (m & (1 << A[a_idx])):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return True
        
        return False
    
    # Initial graph: K_{3,13}
    # U = {0, 1, 2}, V = {3, ..., 15}
    edges = set()
    for u in range(3):
        for v in range(3, 16):
            edges.add((u, v))
    
    # Build adjacency masks
    adj_mask = [0] * 16
    for (u, v) in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    all_pairs = set(itertools.combinations(range(16), 2))
    
    def degree(v):
        return bin(adj_mask[v]).count('1')
    
    def try_add_edge(u, v):
        """Try to add edge (u,v). Returns True if safe."""
        if (u, v) in edges:
            return True
        if (v, u) in edges:
            return True
        
        # Temporarily add
        edges.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        # Check all 8-sets containing u and v
        for combo6 in pair_to_6sets[(u, v)]:
            v8 = (u, v) + combo6
            if contains_q3_in_set(v8, adj_mask):
                # Remove and return False
                edges.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                return False
        
        return True
    
    # Greedy addition with multiple passes
    best_edges = set(edges)
    best_count = len(edges)
    
    # Multiple passes
    for pass_num in range(20):
        if time.time() > deadline - 5:
            break
        
        # Get current non-edges
        non_edges = [e for e in all_pairs if e not in edges]
        
        # Sort by combined degree (higher first) for deterministic pass
        if pass_num % 2 == 0:
            non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]), reverse=True)
        else:
            # Random shuffle for diversity
            random.seed(pass_num * 12345)
            random.shuffle(non_edges)
        
        changed = False
        for e in non_edges:
            if time.time() > deadline - 3:
                break
            u, v = e
            if try_add_edge(u, v):
                changed = True
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = set(edges)
        
        if not changed:
            break
    
    # Restore best if current is worse
    if len(best_edges) > len(edges):
        edges = best_edges
        adj_mask = [0] * 16
        for (u, v) in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
    
    return [list(e) for e in edges]
