import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    best_edges = []
    best_count = 0

    # Precompute bit masks for vertices
    bit = [1 << i for i in range(n)]

    def creates_q3(u, v, adj):
        # adj is a list of 16 ints, where adj[x] is bitmask of neighbors
        # Check if adding (u,v) creates a Q3.
        # We assume (u,v) is NOT yet in adj.
        
        # Q3 has vertices that can be partitioned into two sets of 4, 
        # with each set forming a 4-cycle, and specific cross edges.
        # A Q3 is isomorphic to the cube. 
        # Key property: Q3 is bipartite with parts of size 4.
        # Also, Q3 is 3-regular.
        # A subgraph isomorphic to Q3 must be an induced subgraph? No, just a subgraph.
        # But since we are adding edges one by one to a simple graph, 
        # and we want to avoid ANY subgraph isomorphic to Q3.
        
        # Efficient check:
        # If we add edge (u,v), any new Q3 must contain this edge.
        # Let the Q3 be H. H has 8 vertices and 12 edges.
        # The edge (u,v) is one of the 12 edges.
        # In Q3, every edge is in exactly 2 faces (4-cycles).
        # So, we can look for 4-cycles containing (u,v).
        # A 4-cycle containing (u,v) is u-x-v-y-u.
        # x must be neighbor of u, y neighbor of v, x!=y, x!=v, y!=u.
        # Also (x,y) must NOT be an edge? No, in a 4-cycle u-x-v-y-u, 
        # the edges are (u,x), (x,v), (v,y), (y,u).
        # Wait, the cycle is u-x-v-y-u. Edges: (u,x), (x,v), (v,y), (y,u).
        # So x is neighbor of u and v. y is neighbor of u and v.
        # So x, y are in N(u) ∩ N(v).
        # Let S = N(u) ∩ N(v).
        # For each pair (x, y) in S with x < y:
        #   We have a 4-cycle u-x-v-y-u.
        #   To extend this to a Q3, we need to find 4 more vertices and edges.
        #   The Q3 contains this 4-cycle.
        #   In Q3, a 4-cycle is a face. The other 4 vertices form another 4-cycle,
        #   parallel to the first, with matching edges.
        #   Specifically, if the face is u-x-v-y-u, then the opposite face is u'-x'-v'-y'-u',
        #   where u' is the vertex opposite to u, etc.
        #   But we don't know u', x', v', y' yet.
        #   
        # Alternative approach:
        # Enumerate all subsets of 8 vertices containing u and v.
        # There are C(14, 6) = 3003 such subsets.
        # For each subset, check if the induced subgraph contains a Q3.
        # This might be too slow if done naively.
        
        # Better approach:
        # A Q3 is a 3-regular bipartite graph with 8 vertices.
        # If we add (u,v), and (u,v) is in a Q3, then in that Q3,
        # u and v are either adjacent or not? They are adjacent in the Q3.
        # So u and v are in the same part of the bipartition? No, adjacent vertices are in opposite parts.
        # So u is in part A, v is in part B (or vice versa).
        # The Q3 has 4 vertices in A and 4 in B.
        # u is in A, v is in B.
        # The other 3 vertices in A are neighbors of v (in the Q3).
        # The other 3 vertices in B are neighbors of u (in the Q3).
        # Let A' = {u, a1, a2, a3} and B' = {v, b1, b2, b3}.
        # Edges in Q3:
        #   u is connected to b1, b2, b3 (and v is not connected to u in the Q3? Wait, (u,v) IS an edge in the Q3).
        #   So u is connected to v, b1, b2, b3? No, degree is 3.
        #   So u is connected to exactly 3 vertices in B'.
        #   Since (u,v) is an edge, u is connected to v and two others in B'.
        #   Let's say u is connected to v, b1, b2.
        #   Similarly, v is connected to u, a1, a2.
        #   
        # This is getting complicated. Let's use the 4-cycle approach.
        # For each 4-cycle u-x-v-y-u:
        #   The Q3 containing this cycle must have 4 more vertices: x', y', u', v' (opposite to x, y, u, v).
        #   Actually, the opposite face is u'-x'-v'-y'-u'.
        #   u' is connected to x' and y'.
        #   x' is connected to u' and v'.
        #   v' is connected to x' and y'.
        #   y' is connected to u' and v'.
        #   Also, x is connected to x', y is connected to y', u is connected to u', v is connected to v'.
        #   So we need to find vertices x', y', u', v' such that:
        #     (x, x'), (y, y'), (u, u'), (v, v') are edges.
        #     (u', x'), (x', v'), (v', y'), (y', u') are edges.
        #     u', x', v', y' are distinct from u, x, v, y and each other.
        
        # This is still complex. Let's try a simpler heuristic check first, 
        # and only do the full check if necessary.
        
        # Heuristic: If the graph is dense, it's likely to contain a Q3.
        # But we need an exact check.
        
        # Let's try enumerating the 4-cycles containing (u,v).
        # S = N(u) ∩ N(v)
        # For each pair (x, y) in S:
        #   We have a 4-cycle u-x-v-y-u.
        #   Now we need to find the "opposite" 4-cycle.
        #   The vertices u', x', v', y' must be neighbors of u, x, v, y respectively.
        #   Specifically:
        #     u' ∈ N(u) \ {x, y, v}  (and u' != u)
        #     x' ∈ N(x) \ {u, v, y}  (and x' != x)
        #     v' ∈ N(v) \ {x, y, u}  (and v' != v)
        #     y' ∈ N(y) \ {u, v, x}  (and y' != y)
        #   And we need:
        #     (u', x'), (x', v'), (v', y'), (y', u') are edges.
        #     u', x', v', y' are distinct and not in {u, x, v, y}.
        
        # This is a constrained search. Let's implement it.
        
        mask_u = adj[u]
        mask_v = adj[v]
        
        # S = N(u) ∩ N(v), excluding u and v themselves (though they are not in their own neighbor sets)
        mask_s = mask_u & mask_v
        
        # Get list of vertices in S
        s_list = []
        mask = mask_s
        while mask:
            lsb = mask & -mask
            x = lsb.bit_length() - 1
            s_list.append(x)
            mask ^= lsb
        
        for i in range(len(s_list)):
            x = s_list[i]
            for j in range(i + 1, len(s_list)):
                y = s_list[j]
                
                # We have a 4-cycle u-x-v-y-u.
                # Now find u', x', v', y'.
                
                # u' ∈ N(u) \ {x, y, v}
                mask_up = mask_u & ~(bit[x] | bit[y] | bit[v])
                
                # x' ∈ N(x) \ {u, v, y}
                mask_xp = adj[x] & ~(bit[u] | bit[v] | bit[y])
                
                # v' ∈ N(v) \ {x, y, u}
                mask_vp = mask_v & ~(bit[x] | bit[y] | bit[u])
                
                # y' ∈ N(y) \ {u, v, x}
                mask_yp = adj[y] & ~(bit[u] | bit[v] | bit[x])
                
                # We need to find u' in mask_up, x' in mask_xp, v' in mask_vp, y' in mask_yp
                # such that:
                #   (u', x') is an edge
                #   (x', v') is an edge
                #   (v', y') is an edge
                #   (y', u') is an edge
                #   and all 4 are distinct and not in {u, x, v, y}.
                
                # Iterate over u'
                mask_u_iter = mask_up
                while mask_u_iter:
                    lsb_u = mask_u_iter & -mask_u_iter
                    u_prime = lsb_u.bit_length() - 1
                    mask_u_iter ^= lsb_u
                    
                    # x' must be in mask_xp ∩ N(u_prime)
                    mask_xp_cand = mask_xp & adj[u_prime]
                    
                    if mask_xp_cand == 0:
                        continue
                    
                    # Iterate over x'
                    mask_x_iter = mask_xp_cand
                    while mask_x_iter:
                        lsb_x = mask_x_iter & -mask_x_iter
                        x_prime = lsb_x.bit_length() - 1
                        mask_x_iter ^= lsb_x
                        
                        # v' must be in mask_vp ∩ N(x_prime)
                        mask_vp_cand = mask_vp & adj[x_prime]
                        
                        if mask_vp_cand == 0:
                            continue
                        
                        # Iterate over v'
                        mask_v_iter = mask_vp_cand
                        while mask_v_iter:
                            lsb_v = mask_v_iter & -mask_v_iter
                            v_prime = lsb_v.bit_length() - 1
                            mask_v_iter ^= lsb_v
                            
                            # y' must be in mask_yp ∩ N(v_prime) ∩ N(u_prime)
                            mask_yp_cand = mask_yp & adj[v_prime] & adj[u_prime]
                            
                            if mask_yp_cand == 0:
                                continue
                            
                            # Check if there is a y' in mask_yp_cand
                            # such that y' is distinct from u', x', v', and not in {u, x, v, y}
                            mask_y_final = mask_yp_cand & ~(bit[u_prime] | bit[x_prime] | bit[v_prime] | bit[u] | bit[x] | bit[v] | bit[y])
                            
                            if mask_y_final != 0:
                                return True
        
        return False

    # Greedy with Local Search
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [0] * n
        current_edges = []
        
        # Greedy Phase
        for u, v in all_edges:
            if not creates_q3(u, v, adj):
                adj[u] |= bit[v]
                adj[v] |= bit[u]
                current_edges.append((u, v))
        
        # Local Search Phase: Kicks
        for _ in range(10):
            if time.time() > deadline:
                break
            
            # Pick a random edge to remove
            if not current_edges:
                break
            idx = random.randint(0, len(current_edges) - 1)
            u_rem, v_rem = current_edges[idx]
            
            # Remove it
            adj[u_rem] &= ~bit[v_rem]
            adj[v_rem] &= ~bit[u_rem]
            current_edges.pop(idx)
            
            # Try to add edges greedily again
            random.shuffle(all_edges)
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                # Check if (u,v) is already in graph
                if (adj[u] >> v) & 1:
                    continue
                if not creates_q3(u, v, adj):
                    adj[u] |= bit[v]
                    adj[v] |= bit[u]
                    current_edges.append((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()
            
    return [[u, v] for u, v in best_edges]
