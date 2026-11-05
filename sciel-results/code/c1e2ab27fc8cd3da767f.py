import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n = 16
    all_vertices = set(range(n))
    
    # Q3 edge list (8 vertices, 12 edges)
    q3_edges = [
        (0,1),(0,2),(0,4),
        (1,3),(1,5),
        (2,3),(2,6),
        (3,7),
        (4,5),(4,6),
        (5,7),
        (6,7)
    ]
    
    def has_q3_with_edge(adj_mask, u, v):
        """Check if adding edge u-v creates a Q3. adj_mask is list of int bitmasks."""
        neu = [x for x in range(n) if x != u and x != v and (adj_mask[u] >> x) & 1]
        nev = [x for x in range(n) if x != v and x != u and (adj_mask[v] >> x) & 1]
        
        if len(neu) < 2 or len(nev) < 2:
            return False
        
        for pos2 in neu:
            for pos4 in neu:
                if pos2 == pos4:
                    continue
                for pos3 in nev:
                    if pos3 == pos2 or pos3 == pos4:
                        continue
                    for pos5 in nev:
                        if pos5 == pos2 or pos5 == pos3 or pos5 == pos4:
                            continue
                        
                        if not ((adj_mask[pos2] >> pos3) & 1): continue
                        if not ((adj_mask[pos4] >> pos5) & 1): continue
                        
                        excluded = {u, v, pos2, pos3, pos4, pos5}
                        cand_pos6 = []
                        for x in range(n):
                            if x in excluded: continue
                            if ((adj_mask[pos2] >> x) & 1) and ((adj_mask[pos4] >> x) & 1):
                                cand_pos6.append(x)
                        
                        cand_pos7 = []
                        for x in range(n):
                            if x in excluded: continue
                            if ((adj_mask[pos3] >> x) & 1) and ((adj_mask[pos5] >> x) & 1):
                                cand_pos7.append(x)
                        
                        for pos6 in cand_pos6:
                            for pos7 in cand_pos7:
                                if pos6 == pos7: continue
                                if ((adj_mask[pos6] >> pos7) & 1) and \
                                   ((adj_mask[pos3] >> pos7) & 1) and \
                                   ((adj_mask[pos5] >> pos7) & 1):
                                    return True
        return False
    
    def count_edges(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    def verify_no_q3(adj_mask):
        """Verify the graph has no Q3 subgraph."""
        for u in range(n):
            for v in range(u+1, n):
                if (adj_mask[u] >> v) & 1:
                    if has_q3_with_edge(adj_mask, u, v):
                        return False
        return True
    
    def adj_mask_to_numpy(adj_mask):
        """Convert adjacency mask list to numpy adjacency matrix."""
        A = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                if (adj_mask[i] >> j) & 1:
                    A[i, j] = 1
        return A
    
    def trace_A4(A):
        """Compute trace of A^4 using matrix multiplication."""
        A2 = A @ A
        A4 = A2 @ A2
        return np.trace(A4)
    
    def remove_edge(adj_mask, A, u, v):
        """Remove edge u-v from adj_mask and A."""
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        A[u, v] = 0
        A[v, u] = 0
    
    def add_edge(adj_mask, A, u, v):
        """Add edge u-v to adj_mask and A."""
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        A[u, v] = 1
        A[v, u] = 1
    
    best_edge_count = 0
    best_adj_mask = None
    
    # === Phase 1: Spectral heuristic - start from K16, remove edges to minimize trace(A^4) ===
    
    # Start with complete graph K16
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj_mask[i] |= (1 << j)
            adj_mask[j] |= (1 << i)
    
    A = adj_mask_to_numpy(adj_mask)
    current_trace = trace_A4(A)
    
    # Iteratively remove the edge whose removal maximizes reduction in trace(A^4)
    # In Q3, there are 6 4-cycles. trace(A^4) counts closed walks of length 4,
    # which includes 4-cycles. Removing edges that break 4-cycles helps avoid Q3.
    
    max_iterations = 50  # We'll remove at most this many edges
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # Check if current graph is already Q3-free
        if verify_no_q3(adj_mask):
            ec = count_edges(adj_mask)
            if ec > best_edge_count:
                best_edge_count = ec
                best_adj_mask = adj_mask[:]
            break
        
        # For each existing edge, compute how much trace(A^4) decreases if we remove it
        best_edge_to_remove = None
        best_reduction = -1
        
        all_existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (adj_mask[i] >> j) & 1]
        
        for u, v in all_existing_edges:
            # Temporarily remove edge
            A[u, v] = 0
            A[v, u] = 0
            
            new_trace = trace_A4(A)
            reduction = current_trace - new_trace
            
            # Restore
            A[u, v] = 1
            A[v, u] = 1
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_edge_to_remove = (u, v)
        
        if best_edge_to_remove is None:
            break
        
        u, v = best_edge_to_remove
        remove_edge(adj_mask, A, u, v)
        current_trace = current_trace - best_reduction
        
        # Periodic check
        if iteration % 5 == 4:
            if verify_no_q3(adj_mask):
                ec = count_edges(adj_mask)
                if ec > best_edge_count:
                    best_edge_count = ec
                    best_adj_mask = adj_mask[:]
                break
    
    # === Phase 2: Greedy edge addition on the spectrally-pruned graph ===
    if time.time() < deadline - 5:
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_mask[i] >> j) & 1)]
        
        # Sort by some heuristic - try adding edges that are "safe" first
        random.seed(42)
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            
            add_edge(adj_mask, A, u, v)
            
            if has_q3_with_edge(adj_mask, u, v):
                remove_edge(adj_mask, A, u, v)
        
        ec = count_edges(adj_mask)
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj_mask = adj_mask[:]
    
    # === Phase 3: Local search with kicks ===
    if time.time() < deadline - 5 and best_adj_mask is not None:
        current_mask = best_adj_mask[:]
        current_ec = count_edges(current_mask)
        
        for kick_iter in range(100):
            if time.time() > deadline - 3:
                break
            
            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((current_mask[i] >> j) & 1)]
            random.shuffle(missing)
            
            improved = False
            for u, v in missing:
                if time.time() > deadline - 3:
                    break
                
                current_mask[u] |= (1 << v)
                current_mask[v] |= (1 << u)
                
                if has_q3_with_edge(current_mask, u, v):
                    current_mask[u] &= ~(1 << v)
                    current_mask[v] &= ~(1 << u)
                else:
                    improved = True
            
            if not improved:
                # Kick: remove 1-3 random edges and try again
                if current_ec > 0:
                    existing = [(i, j) for i in range(n) for j in range(i+1, n) if (current_mask[i] >> j) & 1]
                    if existing:
                        num_kick = min(3, len(existing))
                        kick_edges = random.sample(existing, num_kick)
                        for u, v in kick_edges:
                            current_mask[u] &= ~(1 << v)
                            current_mask[v] &= ~(1 << u)
                        continue
            
            ec = count_edges(current_mask)
            if ec > current_ec:
                current_ec = ec
                if ec > best_edge_count:
                    best_edge_count = ec
                    best_adj_mask = current_mask[:]
    
    # Final verification
    if best_adj_mask is None:
        best_adj_mask = [0] * n
        for u in range(3):
            for v in range(3, 16):
                best_adj_mask[u] |= (1 << v)
                best_adj_mask[v] |= (1 << u)
    
    if not verify_no_q3(best_adj_mask):
        best_adj_mask = [0] * n
        for u in range(3):
            for v in range(3, 16):
                best_adj_mask[u] |= (1 << v)
                best_adj_mask[v] |= (1 << u)
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_mask[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
