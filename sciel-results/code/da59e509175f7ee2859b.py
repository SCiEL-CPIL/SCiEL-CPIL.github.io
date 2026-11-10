import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110
    n = 40
    # Initial state: 8 disjoint K5s (104 edges)
    # This is a known C6-free structure with high edge count.
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append((5 * k + a, 5 * k + b))
    
    # Bitset representation of the graph for fast C6 checking
    # adj_mask[i] is an integer where bit j is 1 if (i, j) is an edge
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute all possible edges not in the initial set
    all_edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    initial_edge_set = set(edges)
    candidate_edges = [e for e in all_edges if e not in initial_edge_set]
    
    def has_c6_fast(adj_mask, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        A C6 containing (u, v) is a path u-w-x-y-z-v of length 5 (edges) 
        where u, v, w, x, y, z are distinct.
        This is equivalent to checking if there is a path of length 4 
        from u to v in G - (u, v).
        Since (u, v) is not yet in the graph, we just check for a path of length 4.
        
        Path: u - w - x - y - z - v
        Edges: (u,w), (w,x), (x,y), (y,z), (z,v)
        
        We can iterate over w in N(u), then x in N(w), etc.
        Using bitsets makes the inner loops very fast.
        """
        # We need distinct vertices u, v, w, x, y, z
        # w in N(u), w != v
        # x in N(w), x not in {u, v, w}
        # y in N(x), y not in {u, v, w, x}
        # z in N(y), z not in {u, v, w, x, y}, and z in N(v)
        
        # Optimization: 
        # 1. Get neighbors of u and v
        # 2. Iterate w in N(u) \ {v}
        # 3. Iterate x in N(w) \ {u, v, w}
        # 4. Check if there exists y in N(x) \ {u, v, w, x} such that N(y) intersects (N(v) \ {u, w, x, y})
        
        # Let's implement with bit operations where possible.
        # N(u) mask
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # Exclude u and v from consideration for intermediate nodes
        # But we are checking for a path u-w-x-y-z-v.
        # w is neighbor of u.
        # z is neighbor of v.
        
        # Iterate w
        # w must be in N(u) and w != v
        w_mask = mask_u & ~(1 << v)
        
        # For each w in w_mask:
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            
            # x must be in N(w), x not in {u, v, w}
            mask_w = adj_mask[w]
            x_mask = mask_w & ~(1 << u) & ~(1 << v) & ~(1 << w)
            
            while x_mask:
                x_bit = x_mask & -x_mask
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit
                
                # y must be in N(x), y not in {u, v, w, x}
                mask_x = adj_mask[x]
                y_mask = mask_x & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                
                while y_mask:
                    y_bit = y_mask & -y_mask
                    y = y_bit.bit_length() - 1
                    y_mask ^= y_bit
                    
                    # z must be in N(y), z not in {u, v, w, x, y}
                    # AND z must be in N(v)
                    mask_y = adj_mask[y]
                    z_mask = mask_y & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x) & ~(1 << y)
                    
                    # Check intersection with N(v)
                    if z_mask & mask_v:
                        return True
        
        return False

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    current_edges = set(edges)
    best_edges = set(edges)
    best_score = len(best_edges)
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    # We will perform a fixed number of steps or until time runs out
    
    steps = 0
    # Randomize initial order of candidates for diversity
    random.shuffle(candidate_edges)
    
    while time.time() < deadline:
        # Pick a random move: add or remove an edge
        # Bias towards adding edges since we want to maximize
        if random.random() < 0.7:
            # Try to add a random edge from candidate list (edges not currently in graph)
            # To be efficient, we can just pick a random possible edge and check if it's already there
            # Or maintain a list of missing edges. For simplicity, pick random pair.
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            if (u, v) in current_edges:
                continue
                
            # Check C6
            if has_c6_fast(adj_mask, u, v):
                continue # Move invalid, reject immediately (or count as bad)
            
            # Move is valid (graph remains C6-free)
            # Add edge
            add_edge(u, v)
            current_edges.add((u, v))
            new_score = len(current_edges)
            
            # Acceptance: always accept improvements
            # This is a greedy improvement step within SA framework
            if new_score > best_score:
                best_score = new_score
                best_edges = set(current_edges)
        
        else:
            # Try to remove a random edge (to escape local optima)
            if not current_edges:
                continue
            e = random.choice(list(current_edges))
            u, v = e
            remove_edge(u, v)
            current_edges.remove(e)
            new_score = len(current_edges)
            
            # Acceptance criterion for worsening move
            # Delta = new_score - old_score = -1
            # Accept with probability exp(-1/T)
            if random.random() < np.exp(-1.0 / T):
                pass # Accepted
            else:
                # Reject, revert
                add_edge(u, v)
                current_edges.add(e)
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart with higher temperature to escape deep local minima?
            # Or just keep going. Let's restart T occasionally.
            if steps % 1000 == 0:
                T = 0.5
        
        steps += 1

    return [[a, b] for (a, b) in sorted(best_edges)]
