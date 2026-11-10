import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Uses a model-based approach: start from a dense structure and optimize.
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    # Helper to check if adding edge (u, v) creates a C6.
    # A C6 is a cycle of length 6. Adding edge (u,v) creates a C6 if there is a path of length 5 from u to v.
    # Path: u - n1 - n2 - n3 - n4 - v.
    # The nodes n1..n4 must be distinct and not u, v.

    def can_add(adj_bits, u, v):
        # BFS from u to depth 4, tracking visited nodes per layer.
        # A path u-n1-n2-n3-n4-v exists if n4 is adjacent to v.
        # We need n1,n2,n3,n4 distinct and not u,v.
        
        # Layer 1: neighbors of u (excluding v)
        # Layer 2: neighbors of layer1 (excluding u, v, and layer1 nodes)
        # etc.
        
        # Use bitmasks for each layer
        layer1 = adj_bits[u] & ~(1 << v) & ~(1 << u)
        if not layer1:
            return True
            
        # For each node in layer1, find neighbors not in {u, v} and not in layer1
        # But we need to be careful: n2 must not equal n1
        # Actually, we need a simple path, so all nodes distinct.
        
        # Let's do proper BFS with visited set
        visited = 1 << u | 1 << v
        current_layer = adj_bits[u] & ~(1 << v) & ~(1 << u)
        
        if not current_layer:
            return True
            
        for depth in range(1, 4):
            if not current_layer:
                return True
            next_layer = 0
            mask = current_layer
            while mask:
                node = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                # Neighbors of node, excluding visited
                neighbors = adj_bits[node] & ~visited
                next_layer |= neighbors
            visited |= next_layer
            current_layer = next_layer
            
        # Now current_layer contains nodes at distance 4 from u (via simple paths)
        # Check if any is adjacent to v
        if current_layer & adj_bits[v]:
            return False
        return True

    def edge_count(bits):
        return sum(bin(b).count('1') for b in bits) // 2

    def get_edges_from_bits(bits):
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if bits[u] & (1 << v):
                    edges.append((u, v))
        return edges

    def set_edges(bits, edges):
        for u in range(N):
            bits[u] = 0
        for u, v in edges:
            bits[u] |= (1 << v)
            bits[v] |= (1 << u)

    # Initialize with 8 disjoint K5s
    adj_bits = [0] * N
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
    current_edges = edge_count(adj_bits)
    best_edges = current_edges
    best_adj_bits = adj_bits[:]
    
    # List of all possible edges
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Maintain missing edges list
    missing_edges = [p for p in all_pairs if not (adj_bits[p[0]] & (1 << p[1]))]
    
    # Local Search with kicks
    no_improve_count = 0
    
    while time.time() < deadline:
        random.shuffle(missing_edges)
        
        added_in_pass = 0
        new_missing = []
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
                
            if can_add(adj_bits, u, v):
                # Add edge
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                added_in_pass += 1
                current_edges += 1
            else:
                new_missing.append((u, v))
                
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_bits = adj_bits[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        missing_edges = new_missing
        
        if added_in_pass == 0 or no_improve_count >= 3:
            # Stuck. Perform a kick.
            if current_edges > 0:
                kick_size = random.randint(2, 5)
                
                # Pick random existing edges to remove
                existing_edges = get_edges_from_bits(adj_bits)
                
                random.shuffle(existing_edges)
                
                for _ in range(kick_size):
                    if not existing_edges:
                        break
                    u, v = existing_edges.pop()
                    # Remove edge
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    current_edges -= 1
                    missing_edges.append((u, v))
                    
                no_improve_count = 0

    # Final polish: try to add any remaining missing edges
    for u, v in missing_edges:
        if can_add(adj_bits, u, v):
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            current_edges += 1
            
    if current_edges > best_edges:
        best_edges = current_edges
        best_adj_bits = adj_bits[:]

    # Return best graph
    final_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj_bits[u] & (1 << v):
                final_edges.append([u, v])
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
