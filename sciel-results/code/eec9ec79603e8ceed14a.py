import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    E = len(all_edges)
    
    # Map edge to index
    edge_idx = {e: i for i, e in enumerate(all_edges)}
    edge_list = all_edges
    
    # Bitmask adjacency for the current graph
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    # Helper to check if adding edge (u, v) creates a Q3
    # Assumes (u, v) is NOT currently in adj
    def creates_q3(u, v):
        # Neighbors of u and v
        nu = adj[u]
        nv = adj[v]
        
        # Find all squares u-a-b-v-u
        # a in N(u), b in N(v), (a,b) in E
        # Also need a != b
        squares = []
        
        # Iterate a in N(u)
        a_mask = nu
        while a_mask:
            a = (a_mask & -a_mask).bit_length() - 1
            a_mask &= a_mask - 1
            
            # b must be in N(v) AND N(a)
            b_mask = nv & adj[a]
            # Exclude a itself? No, a is not in N(v) usually unless triangle, but b != a required for square in simple graph context of Q3 embedding?
            # In Q3, a and b are distinct vertices.
            # Also a cannot be v (since u-v not edge, v not in N(u)).
            # b cannot be u (since u-v not edge, u not in N(v)).
            while b_mask:
                b = (b_mask & -b_mask).bit_length() - 1
                b_mask &= b_mask - 1
                if a == b:
                    continue
                squares.append((a, b))
        
        if len(squares) < 2:
            return False
            
        # We need two distinct squares (a,b) and (c,d)
        # Vertices {u,v,a,b} and {u,v,c,d} must be disjoint except for u,v
        # So a,b,c,d must be distinct.
        # Then we need e, f such that:
        # e ~ a, c
        # f ~ b, d
        # e ~ f
        # e, f distinct from {u,v,a,b,c,d}
        
        num_squares = len(squares)
        for i in range(num_squares):
            a, b = squares[i]
            for j in range(i + 1, num_squares):
                c, d = squares[j]
                
                # Check distinctness of a,b,c,d
                if a == c or a == d or b == c or b == d:
                    continue
                
                # Used vertices so far: u, v, a, b, c, d
                used_mask = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                
                # Candidates for e: N(a) & N(c) & ~used_mask
                e_cands = adj[a] & adj[c] & ~used_mask
                if not e_cands:
                    continue
                
                # Candidates for f: N(b) & N(d) & ~used_mask
                f_cands = adj[b] & adj[d] & ~used_mask
                if not f_cands:
                    continue
                
                # Check if any e in e_cands is connected to any f in f_cands
                # e != f is guaranteed if we check adj[e] & f_cands, because simple graph has no self-loops.
                # If e is in f_cands, adj[e] does not have bit e.
                found = False
                e_mask = e_cands
                while e_mask and not found:
                    e = (e_mask & -e_mask).bit_length() - 1
                    e_mask &= e_mask - 1
                    if adj[e] & f_cands:
                        found = True
                
                if found:
                    return True
                    
        return False

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def graph_size():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    best_edges = 0
    best_adj = [0] * n
    best_graph = set()
    
    # Local Search with 1-1 moves (swap)
    # Start with a random greedy construction
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Reset adj
        adj = [0] * n
        current_graph = set()
        
        # Greedy Init
        random.shuffle(all_edges)
        for edge in all_edges:
            u, v = edge
            if not creates_q3(u, v):
                current_graph.add(edge)
                add_edge(u, v)
        
        current_size = len(current_graph)
        
        # Local Improvement Loop
        improved = True
        no_improve_count = 0
        max_no_improve = 20 # Escape local minima
        
        while improved and time.time() < deadline:
            improved = False
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                break
                
            # Try to add an edge
            added = False
            # To speed up, we can keep a list of missing edges? 
            # Or just iterate all_edges. With n=16, E=120, this is fast.
            for edge in all_edges:
                if edge in current_graph:
                    continue
                u, v = edge
                if not creates_q3(u, v):
                    current_graph.add(edge)
                    add_edge(u, v)
                    current_size += 1
                    improved = True
                    added = True
                    no_improve_count = 0
                    break
                    
            if added:
                continue
                
            # Try a 1-1 swap (remove one, add one)
            # Pick a random edge to remove
            if current_graph:
                remove_edge = random.choice(list(current_graph))
                u1, v1 = remove_edge
                current_graph.remove(remove_edge)
                remove_edge(u1, v1)
                current_size -= 1
                
                # Try to add a different edge
                # We want to add an edge that was previously blocked by remove_edge
                # Or just any edge.
                # Let's try to add an edge from the set of missing edges.
                # To be efficient, we can just iterate all missing edges.
                # But we want to prioritize edges that might become valid.
                # A simple approach: iterate all missing edges, try to add.
                # If we add one, we have a net 0 change in size (removed 1, added 1).
                # But we might escape a local minimum where no single add was possible.
                
                added_swap = False
                # Shuffle to randomize order? Or just iterate.
                # Iterating all 120 edges is cheap.
                for edge in all_edges:
                    if edge in current_graph:
                        continue
                    u2, v2 = edge
                    if not creates_q3(u2, v2):
                        current_graph.add(edge)
                        add_edge(u2, v2)
                        current_size += 1
                        added_swap = True
                        no_improve_count = 0
                        # This is a neutral move in terms of size, but changes structure.
                        # We mark improved to continue searching from this new state.
                        improved = True 
                        break
                        
                if not added_swap:
                    # Revert the removal to avoid losing an edge
                    current_graph.add(remove_edge)
                    add_edge(u1, v1)
                    current_size += 1
                    # If we couldn't add anything after removal, we are stuck.
                    # Break out of this local search phase.
                    break
        # End while improved
        
        if current_size > best_edges:
            best_edges = current_size
            best_adj = adj[:]
            best_graph = set(current_graph)
            
    # Final check: maybe we can add more edges to the best graph found?
    # The local search above already tries to add edges until it can't.
    # But the last state might not be the best.
    # Let's do a final pass on the best graph to ensure maximality.
    
    adj = best_adj[:]
    current_graph = set(best_graph)
    current_size = len(current_graph)
    
    # Try to add any remaining edges
    for edge in all_edges:
        if edge in current_graph:
            continue
        u, v = edge
        if not creates_q3(u, v):
            current_graph.add(edge)
            add_edge(u, v)
            current_size += 1
            
    if current_size > best_edges:
        best_edges = current_size
        best_graph = current_graph

    return [[u, v] for u, v in best_graph]

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges")
