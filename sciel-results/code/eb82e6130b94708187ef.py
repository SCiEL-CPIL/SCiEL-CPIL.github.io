"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 40
    target_edges = 401
    total_possible_edges = n * (n - 1) // 2
    num_non_edges = total_possible_edges - target_edges

    # Use NumPy for fast adjacency operations
    A = np.zeros((n, n), dtype=bool)
    
    def get_edges(A):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    edges.append((i, j))
        return edges

    def count_edges(A):
        return np.sum(A) // 2

    def get_max_book(A):
        """Compute max book size efficiently."""
        max_b = 0
        # Iterate only over edges
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    b = np.sum(A[i] & A[j])
                    if b > max_b:
                        max_b = b
        return max_b

    # 1. Initialize with a random graph of target_edges
    A = np.zeros((n, n), dtype=bool)
    while count_edges(A) < target_edges:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v and not A[u, v]:
            A[u, v] = True
            A[v, u] = True
    
    current_max = get_max_book(A)
    best_A = A.copy()
    best_max = current_max
    
    # Precompute lists of edges and non-edges for fast sampling?
    # Or just generate on the fly.
    # For N=40, generating on the fly is fast enough if we cache the current state.
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-5
    
    # We will run SA for a fixed number of steps or until time limit
    # Each step: pick a random edge to remove, pick a random non-edge to add.
    
    # To speed up sampling, we can maintain lists of edges and non-edges.
    # But updating them is O(N^2) per swap.
    # Instead, we can just sample random pairs and check.
    # Probability of picking a valid edge/non-edge is ~ 400/800 = 0.5.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Pick a random edge to remove
        # And a random non-edge to add
        
        # To make it faster, we can pre-compute the list of edges and non-edges
        # every K steps or if the list gets too stale?
        # Actually, let's just maintain the lists.
        # Since we start with 401 edges, we can build the list once.
        # When we swap, we update the lists.
        
        # Let's build the lists initially
        edge_list = []
        non_edge_list = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    edge_list.append((i, j))
                else:
                    non_edge_list.append((i, j))
        
        # SA Loop
        while time.time() < deadline and T > min_T:
            # Pick random edge and non-edge
            u, v = random.choice(edge_list)
            a, b = random.choice(non_edge_list)
            
            # Perform swap
            A[u, v] = False
            A[v, u] = False
            A[a, b] = True
            A[b, a] = True
            
            new_max = get_max_book(A)
            delta = new_max - current_max
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept move
                current_max = new_max
                
                # Update lists
                # Remove (u, v) from edge_list, add to non_edge_list
                # Remove (a, b) from non_edge_list, add to edge_list
                # O(N) operation to find and remove/swap
                idx = edge_list.index((u, v))
                edge_list[idx] = (a, b)
                
                idx = non_edge_list.index((a, b))
                non_edge_list[idx] = (u, v)
                
                # Update best
                if current_max < best_max:
                    best_max = current_max
                    best_A = A.copy()
            else:
                # Reject move
                A[u, v] = True
                A[v, u] = True
                A[a, b] = False
                A[b, a] = False
            
            T *= cooling_rate
            
            # If we are stuck for a while, we might want to reheat or kick?
            # For now, just continue.
            
            # Check time
            if time.time() > deadline:
                break

    # Convert best_A to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                result.append([i, j])
    
    # Verify count
    if len(result) != 401:
        # Fallback
        pass
        
    return result
# EVOLVE-BLOCK-END
