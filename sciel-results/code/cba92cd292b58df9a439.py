# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 40
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_c6_free_vec(A, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        Equivalent to checking if there is a path of length 5 between u and v.
        Uses matrix-vector multiplication for efficiency.
        """
        vec = np.zeros(n, dtype=np.int32)
        vec[u] = 1
        # Compute A^5 * e_u
        for _ in range(5):
            vec = A @ vec
        return vec[v] == 0

    def greedy_add(A, edges, edge_set, candidates):
        """
        Greedily add edges from candidates to A if C6-free.
        """
        for u, v in candidates:
            if A[u, v] == 0:
                if is_c6_free_vec(A, u, v):
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.append((u, v))
                    edge_set.add((u, v))
        return edges, edge_set

    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 115.0

    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(restart * 7919 + 13)
        
        A = np.zeros((n, n), dtype=np.int32)
        edges = []
        edge_set = set()
        
        # Initial Greedy
        candidates = list(all_edges)
        random.shuffle(candidates)
        edges, edge_set = greedy_add(A, edges, edge_set, candidates)
        current_count = len(edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = edges.copy()
            
        # ILS Loop
        while time.time() - start_time < time_limit:
            # Try to add any missing edge
            added = False
            # Get list of non-edges
            non_edges = [(u, v) for u, v in all_edges if A[u, v] == 0]
            
            if not non_edges:
                break
                
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if is_c6_free_vec(A, u, v):
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.append((u, v))
                    edge_set.add((u, v))
                    current_count += 1
                    added = True
                    break
                    
            if added:
                if current_count > best_count:
                    best_count = current_count
                    best_edges = edges.copy()
                continue
                
            # Stuck: Perturb
            # Remove k random edges
            k = random.randint(2, 8)
            if len(edges) < k:
                break
                
            # Sample indices to remove
            idxs = random.sample(range(len(edges)), k)
            # Remove them
            for idx in idxs:
                u, v = edges[idx]
                A[u, v] = 0
                A[v, u] = 0
                
            # Rebuild edge list
            edges = []
            edge_set = set()
            for u, v in all_edges:
                if A[u, v] == 1:
                    edges.append((u, v))
                    edge_set.add((u, v))
                    
            current_count = len(edges)
            
            # Re-greedy
            candidates = list(all_edges)
            random.shuffle(candidates)
            edges, edge_set = greedy_add(A, edges, edge_set, candidates)
            current_count = len(edges)
            
            if current_count > best_count:
                best_count = current_count
                best_edges = edges.copy()

    return best_edges
# EVOLVE-BLOCK-END
