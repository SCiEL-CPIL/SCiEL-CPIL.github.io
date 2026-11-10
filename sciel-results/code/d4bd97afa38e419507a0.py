import time
import random
import sys

def construct():
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Bitmask helpers
    def bits_to_list(mask):
        out = []
        while mask:
            b = mask & -mask
            out.append(b.bit_length() - 1)
            mask ^= b
        return out

    def can_add(adj, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 is formed if there is a path u-a-b-c-v of length 4 in the current graph.
        """
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        if nu == 0 or nv == 0:
            return True
            
        a_mask = nu
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # If a is also a neighbor of v, we must handle a==c case carefully
            # Path u-a-b-c-v. If a==c, path is u-a-b-a-v.
            # This requires a-b and b-a.
            # But a is repeated. So it's not a simple path.
            # So if a == c, it's not a valid C6 witness.
            # We can skip c if c == a.
            
            c_mask = nv
            if c_mask & a_bit:
                c_mask ^= a_bit
                
            while c_mask:
                c_bit = c_mask & -c_mask
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                
                # Check if there is a b such that a-b-c is a path
                # b in N(a) and b in N(c)
                # b != u, v, a, c
                common = adj[a] & adj[c]
                # Exclude u, v, a, c
                # u and v are not in N(a) or N(c) necessarily?
                # u is not in N(a) if graph is simple? No, u-a is an edge, so u in N(a).
                # So we must exclude u.
                # v is in N(c) if c-v is an edge. So v in N(c).
                # a is in N(a)? No, simple graph.
                # c is in N(c)? No.
                # So we need to exclude u, v, a, c from the common neighbors.
                
                # Mask to exclude u, v, a, c
                # Note: a and c are distinct (we skipped a==c)
                exclude_mask = (1 << u) | (1 << v) | a_bit | c_bit
                if common & ~exclude_mask:
                    return False
                    
        return True

    # Initialize adj
    adj = [0] * N
    
    def edge_count(adj):
        return sum(bin(x).count('1') for x in adj) // 2

    # Strategy: Start with a random graph, remove C6s, then optimize.
    # Or start empty and add.
    # Starting empty is safer but might get stuck.
    # Let's try a hybrid: Start with a random graph of density ~0.2, clean it, then add.
    
    def random_valid_graph():
        adj_rand = [0] * N
        # Add random edges
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.25:
                    adj_rand[i] |= (1 << j)
                    adj_rand[j] |= (1 << i)
        
        # Remove edges to eliminate C6s
        # We need to find C6s.
        # A C6 exists if there is a cycle.
        # We can just try to remove edges that are part of a C6.
        # How to find a C6 quickly?
        # We can check all pairs (u,v) to see if adding them creates a C6? No, they are already there.
        # We need to detect existing C6s.
        
        # Alternative: Just use the greedy addition from empty.
        # It's proven to work well if we use kicks.
        
        return adj_rand

    # Let's stick to Greedy Addition from Empty, but with better kicks.
    
    # Initial structure: Random
    for i in range(N):
        for j in range(i+1, N):
            if random.random() < 0.1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    current_edges = edge_count(adj)
    best_edges = current_edges
    best_adj = adj[:]
    
    # List of missing edges
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    missing_edges = [p for p in all_pairs if not (adj[p[0]] & (1 << p[1]))]
    
    # Main Loop
    while time.time() < deadline:
        # Try to add edges
        # Shuffle missing edges to avoid bias
        random.shuffle(missing_edges)
        
        added = 0
        new_missing = []
        
        for u, v in missing_edges:
            if time.time() > deadline - 5:
                break
                
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                added += 1
            else:
                new_missing.append((u, v))
                
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
        missing_edges = new_missing
        
        if added == 0:
            # Stuck. Kick.
            # Remove some random edges to create space.
            # Prefer removing edges that are "central" or high degree?
            # Or just random.
            # Let's remove 1-3 random edges.
            existing_edges = []
            for u in range(N):
                for v in range(u+1, N):
                    if adj[u] & (1 << v):
                        existing_edges.append((u, v))
                        
            if not existing_edges:
                break
                
            kick_size = random.randint(1, 3)
            for _ in range(kick_size):
                if not existing_edges:
                    break
                idx = random.randint(0, len(existing_edges) - 1)
                u, v = existing_edges[idx]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
                missing_edges.append((u, v))
                existing_edges.remove((u, v))

    # Final Polish
    # Try to add any remaining missing edges
    if time.time() < deadline - 2:
        random.shuffle(missing_edges)
        for u, v in missing_edges:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

    # Return best
    final_edges = []
    for u in range(N):
        for v in range(u+1, N):
            if best_adj[u] & (1 << v):
                final_edges.append([u, v])
                
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
