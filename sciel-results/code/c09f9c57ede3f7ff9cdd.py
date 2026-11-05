import time
import random
import sys

def _is_3_colorable(adj, n):
    """
    Check if the graph defined by adjacency bitmasks is 3-colorable.
    Uses backtracking with vertex ordering by degree.
    """
    colors = [-1] * n
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda v: bin(adj[v]).count('1'), reverse=True)
    
    def backtrack(pos):
        if pos == n:
            return True
        
        i = order[pos]
        
        # Calculate forbidden colors based on colored neighbors
        forbidden = 0
        neighbors = adj[i]
        while neighbors:
            lsb = neighbors & -neighbors
            j = lsb.bit_length() - 1
            if colors[j] != -1:
                forbidden |= (1 << colors[j])
            neighbors ^= lsb
        
        available = 7 ^ forbidden # 3 colors: 0, 1, 2 -> mask 111
        
        if available == 0:
            return False
        
        # Try each available color
        while available:
            lsb = available & -available
            c = lsb.bit_length() - 1
            colors[i] = c
            if backtrack(pos + 1):
                return True
            colors[i] = -1
            available ^= lsb
            
        return False
    
    return backtrack(0)

def _is_triangle_free(adj, n):
    """
    Check if the graph is triangle-free.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                if adj[u] & adj[v]:
                    return False
    return True

def _count_edges(adj, n):
    count = 0
    for u in range(n):
        count += bin(adj[u]).count('1')
    return count // 2

def _get_non_edges(adj, n):
    """
    Return list of non-edges (u, v) with u < v.
    """
    non_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                non_edges.append((u, v))
    return non_edges

def _get_edges(adj, n):
    """
    Return list of edges (u, v) with u < v.
    """
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append((u, v))
    return edges

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Initialize with Clebsch Graph
    # Vertices 0..15. Adjacent if Hamming distance is 3 or 4.
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            diff = bin(u ^ v).count('1')
            if diff >= 3:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    best_adj = list(adj)
    best_edges_count = _count_edges(adj, n)
    
    # Verify initial state
    if _is_triangle_free(adj, n) and not _is_3_colorable(adj, n):
        pass # Valid
    else:
        # Fallback if construction failed (should not happen)
        adj = [0] * n
        best_adj = list(adj)
        best_edges_count = 0

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Local Search
    # We will use a simple hill climbing with random restarts and swaps.
    # Since n=16 is small, we can be aggressive.
    
    current_adj = list(adj)
    current_edges_count = best_edges_count
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart or significant perturbation
            # Try a random restart from a slightly perturbed Clebsch graph
            current_adj = list(best_adj)
            # Randomly remove 2-3 edges and re-add random ones?
            # Or just start fresh from Clebsch
            current_adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    diff = bin(u ^ v).count('1')
                    if diff >= 3:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
            current_edges_count = _count_edges(current_adj, n)
            no_improve = 0
            continue

        # Generate candidate moves
        moves = []
        
        # 1. Try adding a random non-edge
        non_edges = _get_non_edges(current_adj, n)
        if non_edges:
            # Sample a few
            for _ in range(50):
                if not non_edges: break
                idx = random.randrange(len(non_edges))
                u, v = non_edges[idx]
                # Check triangle constraint
                if current_adj[u] & current_adj[v]:
                    continue
                
                # Tentatively add
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
                # Check 3-colorability
                if not _is_3_colorable(current_adj, n):
                    # Valid move, improves edge count
                    current_edges_count += 1
                    if current_edges_count > best_edges_count:
                        best_edges_count = current_edges_count
                        best_adj = list(current_adj)
                    no_improve = 0
                    break # Try next step
                else:
                    # Revert
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    no_improve += 1
        else:
            no_improve += 1

        # 2. If no add move worked, try a swap move
        # Remove edge (a,b) and add edge (u,v)
        edges = _get_edges(current_adj, n)
        non_edges = _get_non_edges(current_adj, n)
        
        if edges and non_edges:
            # Sample a swap
            # To be efficient, we don't iterate all, just sample
            swap_found = False
            for _ in range(100):
                a, b = random.choice(edges)
                u, v = random.choice(non_edges)
                
                # Ensure (u,v) is not (a,b) or (b,a)
                if (u == a and v == b) or (u == b and v == a):
                    continue
                
                # Check if adding (u,v) creates triangle with existing edges (excluding a,b if relevant? No, we remove first)
                # Let's simulate: Remove (a,b), Add (u,v)
                
                # Remove (a,b)
                current_adj[a] &= ~(1 << b)
                current_adj[b] &= ~(1 << a)
                
                # Check if (u,v) can be added without creating triangle
                # Note: The graph changed by removing (a,b).
                if current_adj[u] & current_adj[v]:
                    # Revert removal
                    current_adj[a] |= (1 << b)
                    current_adj[b] |= (1 << a)
                    continue
                
                # Add (u,v)
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
                # Check 3-colorability
                if not _is_3_colorable(current_adj, n):
                    # Valid swap. Edge count remains same.
                    # This helps escape local optima.
                    # We don't update best_edges_count as it's the same.
                    # But we might have moved to a better position.
                    no_improve = 0
                    swap_found = True
                    break
                else:
                    # Revert addition and removal
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_adj[a] |= (1 << b)
                    current_adj[b] |= (1 << a)
                    no_improve += 1
            
            if not swap_found:
                no_improve += 1

    # Return the best graph found
    # Convert best_adj to list of edges
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result_edges.append([u, v])
    
    return result_edges
