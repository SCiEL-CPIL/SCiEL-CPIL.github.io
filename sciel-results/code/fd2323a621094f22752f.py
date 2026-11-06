import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n = 31
    m = 91
    
    # Initialize a random graph
    # We can start with a random set of 91 edges
    edges = set()
    while len(edges) < m:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            edges.add((min(u, v), max(u, v)))
            
    # Build adjacency list
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    def count_c4():
        # Count all C4s
        # A C4 is u-v-w-x-u with u<v<w<x? No, just count subgraphs.
        # Number of C4s = 1/4 * sum over all pairs (u,v) of (common neighbors of u and v choose 2)?
        # No, that's for C4s where u,v are opposite?
        # Let's use the path count method or direct enumeration.
        # Direct enumeration:
        # For each pair of vertices (u, v), let c = |N(u) \cap N(v)|.
        # The number of C4s containing edge (u,v) is ... no.
        # The number of C4s in total is:
        # Sum over all 4-tuples?
        
        # Efficient way:
        # For each pair of vertices (u, v), let c_uv be the number of common neighbors.
        # The number of 4-cycles is 1/4 * sum_{u < v} c_uv * (c_uv - 1)?
        # No, that counts C4s where u and v are opposite vertices.
        # Each C4 has 2 pairs of opposite vertices.
        # So total C4s = 1/2 * sum_{u < v} \binom{c_uv}{2}.
        
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                # Common neighbors
                common = neighbors[u].intersection(neighbors[v])
                c = len(common)
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2

    current_c4 = count_c4()
    best_c4 = current_c4
    best_edges = set(edges)
    
    # Local Search
    # We will try to swap edges to reduce C4 count.
    
    while time.time() < deadline:
        # Pick a random edge to remove
        if not edges:
            break
        e_out = random.choice(list(edges))
        u_out, v_out = e_out
        
        # Pick a random non-edge to add
        # Generate random pairs until we find a non-edge
        # Or maintain a list of non-edges?
        # n=31, total pairs = 465. m=91. non-edges = 374.
        # Just sampling is fine.
        e_in = None
        attempts = 0
        while e_in is None and attempts < 100:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x != y:
                e_in_candidate = (min(x, y), max(x, y))
                if e_in_candidate not in edges:
                    e_in = e_in_candidate
            attempts += 1
            
        if e_in is None:
            continue
            
        x_in, y_in = e_in
        
        # Calculate change in C4 count
        # Remove e_out: subtract C4s containing e_out
        # Add e_in: add C4s containing e_in
        
        # Function to count C4s containing a specific edge (u, v)
        # A C4 containing (u, v) is a path of length 3 from u to v?
        # No, (u, v) is an edge in the C4.
        # The other two vertices w, z must be adjacent to both u and v?
        # No.
        # C4: u-v-w-z-u.
        # Edge (u, v) is present.
        # Then v-w, w-z, z-u must be present.
        # So w must be a neighbor of v and z must be a neighbor of u.
        # And w-z must be an edge.
        # Also u, v, w, z distinct.
        
        # So, number of C4s containing (u, v) is:
        # Sum over w in N(v) \ {u} of Sum over z in N(u) \ {v} of [w in N(z)] and [w != z]
        
        def count_c4_containing_edge(u, v):
            count = 0
            # w is neighbor of v (and not u)
            # z is neighbor of u (and not v)
            # w and z must be connected
            # w != z
            for w in neighbors[v]:
                if w == u:
                    continue
                for z in neighbors[u]:
                    if z == v:
                        continue
                    if w == z:
                        continue
                    if z in neighbors[w]:
                        count += 1
            return count

        c4_removed = count_c4_containing_edge(u_out, v_out)
        
        # To calculate c4_added for (x_in, y_in), we need to simulate the removal of e_out first?
        # If e_out and e_in share vertices, the count might be affected.
        # It's safer to:
        # 1. Remove e_out from graph structures.
        # 2. Calculate c4_added for e_in.
        # 3. Add e_in.
        # 4. Check if total decreased.
        
        # Remove e_out
        edges.discard(e_out)
        neighbors[u_out].discard(v_out)
        neighbors[v_out].discard(u_out)
        
        c4_added = count_c4_containing_edge(x_in, y_in)
        
        new_c4 = current_c4 - c4_removed + c4_added
        
        if new_c4 < current_c4:
            # Accept
            current_c4 = new_c4
            edges.add(e_in)
            neighbors[x_in].add(y_in)
            neighbors[y_in].add(x_in)
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = set(edges)
        else:
            # Revert
            edges.add(e_out)
            neighbors[u_out].add(v_out)
            neighbors[v_out].add(u_out)
            
            # Do NOT add e_in
            
    # Return the best edges found
    # Convert set of tuples to list of lists
    return [list(e) for e in best_edges]

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(len(res))
