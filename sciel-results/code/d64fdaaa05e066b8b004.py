import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial graph: K_{3,13}
    # U = {0, 1, 2}, V = {3, ..., 15}
    edges = set()
    for u in range(3):
        for v in range(3, 16):
            edges.add((u, v))
    
    # Precompute bitmask for each vertex: bit j is set if (i,j) in edges
    # Use a 16-bit integer per vertex
    adj_mask = [0] * 16
    for (u, v) in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Helper: check if 8 vertices contain Q3 as a subgraph
    # Q3 is the 3-cube: 8 vertices, 12 edges, each vertex degree 3
    # Q3 is bipartite with parts of size 4
    # A set of 8 vertices contains Q3 iff there exists a bipartition (A,B) with |A|=|B|=4
    # such that the induced bipartite graph between A and B contains Q3 as a subgraph.
    # Q3 as a bipartite graph: each vertex in A connects to exactly 3 vertices in B,
    # and each vertex in B connects to exactly 3 vertices in A.
    # So we need: for some partition (A,B), the bipartite graph G[A,B] is a supergraph of Q3.
    # Equivalently, the complement (missing edges) between A and B must NOT contain
    # a perfect matching... wait, let me think more carefully.
    
    # Actually, Q3 has 12 edges total, 6 on each side of the bipartition.
    # For the induced subgraph on A∪B to contain Q3, we need at least 12 edges present
    # in a pattern that forms Q3.
    
    # Better approach: enumerate all 70 ways to split 8 vertices into two sets of 4.
    # For each split (A, B), check if there's a bijection f: A -> B such that
    # each a in A is connected to exactly 3 vertices in B (including f(a)), and
    # each b in B is connected to exactly 3 vertices in A (including f^{-1}(b)).
    # Wait, that's not quite right either.
    
    # Let me use a cleaner formulation:
    # The 3-cube Q3 can be labeled by binary strings of length 3.
    # Vertices: 000, 001, 010, 011, 100, 101, 110, 111
    # Edges: between vertices differing in exactly one bit.
    # Bipartition: even parity (000,011,101,110) vs odd parity (001,010,111,100)
    # Each vertex has degree 3.
    
    # For a given set of 8 vertices and a partition into A (size 4) and B (size 4):
    # We need to check if there exists a labeling of A by even-parity bits and B by
    # odd-parity bits (or vice versa) such that all 12 cube edges are present.
    
    # There are 4! = 24 ways to assign the 4 even-parity labels to A, and 4! = 24
    # ways to assign the 4 odd-parity labels to B. Total 576 labelings per partition.
    # But we can be smarter.
    
    # For a fixed partition (A, B), let's define the adjacency matrix M where
    # M[a][b] = 1 if a and b are adjacent in the graph.
    # We need to find a permutation π of B such that the resulting bipartite graph
    # is a supergraph of Q3.
    
    # Q3 bipartite structure: fix A = {a0, a1, a2, a3} with labels:
    # a0: 000, a1: 011, a2: 101, a3: 110
    # B = {b0, b1, b2, b3} with labels:
    # b0: 001, b1: 010, b2: 111, b3: 100
    # Edges in Q3:
    # a0(000) connects to b0(001), b1(010), b3(100)
    # a1(011) connects to b0(001), b2(111), b1(010)
    # a2(101) connects to b0(001), b2(111), b3(100)
    # a3(110) connects to b1(010), b2(111), b3(100)
    
    # So the required adjacency pattern (rows=A, cols=B) is:
    # a0: b0, b1, b3
    # a1: b0, b1, b2
    # a2: b0, b2, b3
    # a3: b1, b2, b3
    
    # For a given partition, we need to check if there's a permutation of B
    # such that all 12 required edges are present.
    
    # Precompute the required edge pattern as a list of (row_idx, col_idx) pairs
    # where row_idx is 0..3 for A and col_idx is 0..3 for B.
    # The pattern is fixed once we fix the labeling of A and B.
    # But we can try all 4! = 24 permutations of B's assignment to the 4 odd-parity slots.
    # And also try swapping A and B (since the bipartition is not ordered).
    
    # Actually, let's just try all 24 permutations of B for a fixed ordering of A.
    # The required edges for a given permutation p (where p[i] is the index in B assigned
    # to odd-parity label i) are:
    # a0 connects to B[p[0]], B[p[1]], B[p[3]]  (labels 001, 010, 100)
    # a1 connects to B[p[0]], B[p[1]], B[p[2]]  (labels 001, 010, 111)
    # a2 connects to B[p[0]], B[p[2]], B[p[3]]  (labels 001, 111, 100)
    # a3 connects to B[p[1]], B[p[2]], B[p[3]]  (labels 010, 111, 100)
    
    # So for permutation p, the required edges are:
    # (a0, B[p[0]]), (a0, B[p[1]]), (a0, B[p[3]])
    # (a1, B[p[0]]), (a1, B[p[1]]), (a1, B[p[2]])
    # (a2, B[p[0]]), (a2, B[p[2]]), (a2, B[p[3]])
    # (a3, B[p[1]]), (a3, B[p[2]]), (a3, B[p[3]])
    
    # That's 12 edges. We need all of them to be present.
    
    # Precompute all 24 permutations of [0,1,2,3]
    perms24 = list(itertools.permutations(range(4)))
    
    # Precompute the 70 index subsets for choosing 4 out of 8
    idx4 = list(itertools.combinations(range(8), 4))
    
    def contains_q3(vertices):
        if len(vertices) != 8:
            return False
        
        v_list = list(vertices)
        
        for (i, j, k, l) in idx4:
            A = [v_list[i], v_list[j], v_list[k], v_list[l]]
            B = [v for idx, v in enumerate(v_list) if idx not in (i, j, k, l)]
            
            # Try all 24 permutations of B for orientation A=even, B=odd
            for p in perms24:
                a0m = adj_mask[A[0]]
                if not ((a0m >> B[p[0]]) & 1): continue
                if not ((a0m >> B[p[1]]) & 1): continue
                if not ((a0m >> B[p[3]]) & 1): continue
                a1m = adj_mask[A[1]]
                if not ((a1m >> B[p[0]]) & 1): continue
                if not ((a1m >> B[p[1]]) & 1): continue
                if not ((a1m >> B[p[2]]) & 1): continue
                a2m = adj_mask[A[2]]
                if not ((a2m >> B[p[0]]) & 1): continue
                if not ((a2m >> B[p[2]]) & 1): continue
                if not ((a2m >> B[p[3]]) & 1): continue
                a3m = adj_mask[A[3]]
                if not ((a3m >> B[p[1]]) & 1): continue
                if not ((a3m >> B[p[2]]) & 1): continue
                if not ((a3m >> B[p[3]]) & 1): continue
                return True
            
            # Try all 24 permutations of A for orientation A=odd, B=even
            # In this case, the required edges swap roles:
            # B[0] connects to A[p[0]], A[p[1]], A[p[3]]
            # B[1] connects to A[p[0]], A[p[1]], A[p[2]]
            # B[2] connects to A[p[0]], A[p[2]], A[p[3]]
            # B[3] connects to A[p[1]], A[p[2]], A[p[3]]
            for p in perms24:
                b0m = adj_mask[B[0]]
                if not ((b0m >> A[p[0]]) & 1): continue
                if not ((b0m >> A[p[1]]) & 1): continue
                if not ((b0m >> A[p[3]]) & 1): continue
                b1m = adj_mask[B[1]]
                if not ((b1m >> A[p[0]]) & 1): continue
                if not ((b1m >> A[p[1]]) & 1): continue
                if not ((b1m >> A[p[2]]) & 1): continue
                b2m = adj_mask[B[2]]
                if not ((b2m >> A[p[0]]) & 1): continue
                if not ((b2m >> A[p[2]]) & 1): continue
                if not ((b2m >> A[p[3]]) & 1): continue
                b3m = adj_mask[B[3]]
                if not ((b3m >> A[p[1]]) & 1): continue
                if not ((b3m >> A[p[2]]) & 1): continue
                if not ((b3m >> A[p[3]]) & 1): continue
                return True
        return False

    # Greedy addition with smart ordering
    all_pairs = set(itertools.combinations(range(16), 2))
    non_edges = list(all_pairs - edges)
    
    # Sort non-edges: prioritize edges between high-degree vertices
    def degree(v):
        return bin(adj_mask[v]).count('1')
    
    non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]), reverse=True)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        current_non_edges = [e for e in non_edges if e not in edges]
        # Re-sort by current degree
        current_non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]), reverse=True)
        for e in current_non_edges:
            if time.time() > deadline:
                break
            u, v = e
            remaining = [x for x in range(16) if x != u and x != v]
            safe = True
            
            # Temporarily add edge
            edges.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            for combo in itertools.combinations(remaining, 6):
                verts = [u, v] + list(combo)
                if contains_q3(verts):
                    safe = False
                    break
            
            if safe:
                changed = True
            else:
                edges.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                
                # If we're close to deadline, stop trying more edges
                if time.time() > deadline - 2:
                    break

    return [list(e) for e in edges]
