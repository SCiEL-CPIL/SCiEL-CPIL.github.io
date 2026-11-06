import random
import time
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices with a high number of edges.
    Uses randomized greedy with fast bitset C6-checks and local search polish.
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    best_edges = []
    best_count = 0

    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 containing (u,v) has the form u-a-x-y-b-v.
        This requires:
        - a in N(u), b in N(v), a != b
        - x in N(a) \ {u,v,a,b}, y in N(b) \ {u,v,a,b,x}
        - x ~ y
        """
        # Fast pre-checks
        nu = adj[u]
        nv = adj[v]
        
        # Remove u and v from consideration for a and b
        nu &= ~(1 << v)
        nv &= ~(1 << u)
        
        # If no common neighbors structure is possible, bail early
        # Actually, we just need to check the specific condition.
        
        # Iterate over pairs (a, b)
        # Optimization: If the intersection of N(a) and N(b) (excluding forbidden) is empty for all pairs, no C6.
        # We can precompute common neighbors of u and v? No, a and b are distinct neighbors.
        
        # Let's try a slightly different angle for speed:
        # For a fixed a in N(u), what are the constraints on b?
        # We need b in N(v), b != a.
        # And we need x in N(a)\{u,v,a,b} and y in N(b)\{u,v,a,b,x} with x~y.
        
        # This is still O(deg(u)*deg(v)*deg^2) in worst case if we iterate x,y.
        # But with bitsets, the inner check is fast.
        
        nu_iter = nu
        while nu_iter:
            a_bit = nu_iter & (-nu_iter)
            a = a_bit.bit_length() - 1
            nu_iter ^= a_bit
            
            # For this a, we need to check all b in N(v) \ {a}
            # Let's precompute the "bad" b's?
            # For a given a, if we add (u,v), and pick b, we create C6 if:
            # exists x in N(a)\{u,v,a,b}, y in N(b)\{u,v,a,b,x} s.t. x~y.
            
            # Let's iterate b
            nv_iter = nv
            if a_bit & nv:
                nv_iter ^= a_bit  # skip b=a
            
            while nv_iter:
                b_bit = nv_iter & (-nv_iter)
                b = b_bit.bit_length() - 1
                nv_iter ^= b_bit
                
                # Forbidden vertices for x and y
                forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << b)
                
                # Candidates for x: neighbors of a, excluding forbidden
                cand_x = adj[a] & ~forbidden
                if cand_x == 0:
                    continue
                
                # Candidates for y: neighbors of b, excluding forbidden
                cand_y = adj[b] & ~forbidden
                if cand_y == 0:
                    continue
                
                # Check if any x in cand_x is adjacent to any y in cand_y (x != y)
                # Since x and y are in different sets (unless they overlap, but x!=y is handled by mask)
                # Actually x and y can be the same vertex? No, x is in N(a), y in N(b).
                # If x == y, then x is adjacent to a and b.
                # The cycle is u-a-x-b-v-u? No, u-a-x-y-b-v.
                # If x=y, the path is u-a-x-b-v. That's a path of length 4.
                # Adding (u,v) makes a 5-cycle u-a-x-b-v-u.
                # We are looking for C6 (6 vertices).
                # So x and y must be distinct.
                
                # Check intersection of neighbors of cand_x with cand_y
                # We need to check if there exists x in cand_x such that adj[x] & cand_y & ~(1<<x) is non-zero.
                
                cx = cand_x
                while cx:
                    x_bit = cx & (-cx)
                    x = x_bit.bit_length() - 1
                    cx ^= x_bit
                    if adj[x] & cand_y & ~(1 << x):
                        return True
        
        return False

    def local_search(adj, edge_set, time_limit):
        """Try to add edges via local search."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(all_edges)
            for u, v in all_edges:
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    improved = True

    # Run multiple restarts
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        adj = [0] * N
        edge_set = set()
        
        # Greedy construction
        for u, v in all_edges:
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        current_count = len(edge_set)
        
        # Local search polish
        # Give it a bit of time, but ensure we have time for at least one more restart
        remaining = deadline - time.time()
        ls_deadline = time.time() + min(5, remaining * 0.5)
        local_search(adj, edge_set, ls_deadline)
        
        current_count = len(edge_set)
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edge_set)
            
            # If we hit the theoretical max or close to it, we could stop, but let's keep going
            # 105 is the best known.
            if best_count >= 105:
                # Try to push further, but don't spend too long if we're already good
                pass

    # Return in the required format: list of [u, v]
    return [[u, v] for u, v in best_edges]
