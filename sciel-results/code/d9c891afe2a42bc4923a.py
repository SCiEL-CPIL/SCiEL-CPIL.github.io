import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    n = 16
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set(all_edges)
    
    best_edges = 0
    best_graph = set()
    
    # Precompute bit masks for vertices
    vertex_masks = [1 << i for i in range(n)]
    
    def creates_q3(adj, u, v):
        """
        Check if adding edge (u,v) to graph with adjacency bitmasks adj creates a Q3.
        adj already includes the (u,v) edge when called.
        Optimized: use bit operations to find candidate 6-vertex Q3 subgraphs.
        """
        nu = adj[u]
        nv = adj[v]
        
        # In a Q3 containing edge (u,v), u and v have 3 other neighbors each
        # that form the rest of the cube. The structure is:
        # u-v is one edge. u has neighbors {v, a1, a2, a3} and v has neighbors {u, b1, b2, b3}
        # where a_i and b_j are matched such that a_i ~ b_j for specific pairs.
        # Actually, in Q3, each vertex has degree 3. So if u-v is an edge in the Q3,
        # then u has 2 other neighbors in the Q3 (besides v) and v has 2 other neighbors in the Q3 (besides u).
        # Let's call them u's other Q3-neighbors: a, b and v's other Q3-neighbors: c, d.
        # The Q3 structure with edge uv:
        # u is connected to v, a, b
        # v is connected to u, c, d
        # a is connected to u, c, d  (a must connect to v's other neighbors c,d)
        # b is connected to u, c, d  (b must connect to v's other neighbors c,d)
        # Wait, that's not right. Let's think about Q3 structure.
        # Q3 vertices: 000, 001, 010, 011, 100, 101, 110, 111
        # Edges: (000,001), (000,010), (000,100), (001,011), (001,101), (010,011), (010,110), (011,111), (100,101), (100,110), (101,111), (110,111)
        # Let u=000, v=001. Then u's other neighbors in Q3: 010, 100. v's other neighbors in Q3: 011, 101.
        # 010 is connected to 011 (yes) and 110 (not in {011,101}). 
        # So a=010 connects to c=011 and d=101? No, 010 is not connected to 101.
        # Let's re-examine. In Q3, two vertices are adjacent if they differ in exactly 1 bit.
        # u=000, v=001. 
        # u's neighbors: 001, 010, 100. So a=010, b=100.
        # v's neighbors: 000, 011, 101. So c=011, d=101.
        # Now check connections between {a,b} and {c,d}:
        # a=010: neighbors are 000, 011, 110. So a~c (010~011). a not ~ d (010 not ~ 101).
        # b=100: neighbors are 000, 101, 110. So b~d (100~101). b not ~ c (100 not ~ 011).
        # So we need a~c and b~d, OR a~d and b~c.
        # Also, we need a 6th vertex e that is connected to... wait, Q3 has 8 vertices.
        # The 8 vertices are u, v, a, b, c, d, and two more? No, I listed 6: u,v,a,b,c,d.
        # The other two are 110 and 111.
        # 110 is connected to 010(a), 100(b), 111.
        # 111 is connected to 011(c), 101(d), 110.
        # So the full Q3 requires 8 vertices: u,v,a,b,c,d,e,f where e=110, f=111.
        # e is connected to a, b, f.
        # f is connected to c, d, e.
        # So to detect Q3 containing edge (u,v), we need to find:
        # a, b in N(u)\{v} and c, d in N(v)\{u} such that:
        # (a~c and b~d) or (a~d and b~c)
        # AND there exists e in N(a)∩N(b) not in {u,v,a,b,c,d} and e~f where f in N(c)∩N(d) not in {u,v,a,b,c,d,e}
        # AND f~e (which is symmetric)
        # Actually, e must be adjacent to both a and b, and f must be adjacent to both c and d, and e~f.
        
        # Step 1: Find candidate (a,b) from N(u)\{v} and (c,d) from N(v)\{u}
        # with the cross-adjacency condition.
        
        a_candidates = nu & ~(1 << v)
        b_candidates = nv & ~(1 << u)
        
        # We need to pick 2 from a_candidates and 2 from b_candidates
        # This is at most C(deg(u)-1, 2) * C(deg(v)-1, 2) pairs
        # For n=16, max degree is 15, so max C(14,2)=91. 91*91 = 8281 checks.
        
        # Let's list the candidates
        a_list = []
        mask = a_candidates
        while mask:
            bit = mask & (-mask)
            a_list.append(bit.bit_length() - 1)
            mask ^= bit
        
        b_list = []
        mask = b_candidates
        while mask:
            bit = mask & (-mask)
            b_list.append(bit.bit_length() - 1)
            mask ^= bit
        
        if len(a_list) < 2 or len(b_list) < 2:
            return False
        
        # For each pair (a1,a2) from a_list and (b1,b2) from b_list
        for i in range(len(a_list)):
            a1 = a_list[i]
            for j in range(i+1, len(a_list)):
                a2 = a_list[j]
                # Check cross adjacency: a1~b1 and a2~b2 OR a1~b2 and a2~b1
                # b1, b2 are from b_list
                for k in range(len(b_list)):
                    b1 = b_list[k]
                    for l in range(k+1, len(b_list)):
                        b2 = b_list[l]
                        
                        # Check if (a1~b1 and a2~b2) or (a1~b2 and a2~b1)
                        if (adj[a1] & vertex_masks[b1]) and (adj[a2] & vertex_masks[b2]):
                            # Valid pairing 1
                            used = {u, v, a1, a2, b1, b2}
                            used_mask = 0
                            for x in used:
                                used_mask |= vertex_masks[x]
                            # Find e in N(a1)∩N(a2) not in used
                            e_mask = adj[a1] & adj[a2] & ~used_mask
                            if e_mask:
                                # Find f in N(b1)∩N(b2) not in used, not e, and e~f
                                f_mask = adj[b1] & adj[b2] & ~used_mask & ~vertex_masks[e_mask.bit_length()-1] if False else 0
                                # Need to iterate e candidates
                                em = e_mask
                                while em:
                                    e_bit = em & (-em)
                                    e = e_bit.bit_length() - 1
                                    em ^= e_bit
                                    f_mask = adj[b1] & adj[b2] & ~used_mask & ~vertex_masks[e] & adj[e]
                                    if f_mask:
                                        return True
                        elif (adj[a1] & vertex_masks[b2]) and (adj[a2] & vertex_masks[b1]):
                            # Valid pairing 2 (a1~b2, a2~b1)
                            # e must be in N(a1)∩N(a2), f in N(b1)∩N(b2)
                            used = {u, v, a1, a2, b1, b2}
                            used_mask = 0
                            for x in used:
                                used_mask |= vertex_masks[x]
                            e_mask = adj[a1] & adj[a2] & ~used_mask
                            em = e_mask
                            while em:
                                e_bit = em & (-em)
                                e = e_bit.bit_length() - 1
                                em ^= e_bit
                                f_mask = adj[b1] & adj[b2] & ~used_mask & ~vertex_masks[e] & adj[e]
                                if f_mask:
                                    return True
        
        return False

    # Multi-restart greedy with local search
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        random.shuffle(all_edges)
        
        # Build adjacency bitmask
        adj = [0] * n
        current_graph = set()
        
        for edge in all_edges:
            u, v = edge
            # Temporarily add u-v to adj for checking
            adj[u] |= vertex_masks[v]
            adj[v] |= vertex_masks[u]
            
            if creates_q3(adj, u, v):
                # Remove
                adj[u] &= ~vertex_masks[v]
                adj[v] &= ~vertex_masks[u]
            else:
                current_graph.add((u, v))
        
        if len(current_graph) > best_edges:
            best_edges = len(current_graph)
            best_graph = set(current_graph)
        
        # Local search: try to add any missing edge
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for edge in all_edges:
                if edge in current_graph:
                    continue
                u, v = edge
                adj[u] |= vertex_masks[v]
                adj[v] |= vertex_masks[u]
                if not creates_q3(adj, u, v):
                    current_graph.add(edge)
                    improved = True
                else:
                    adj[u] &= ~vertex_masks[v]
                    adj[v] &= ~vertex_masks[u]
        
        if len(current_graph) > best_edges:
            best_edges = len(current_graph)
            best_graph = set(current_graph)
        
        # Edge swap local search: remove one edge, try to add two others
        if time.time() < deadline:
            # Rebuild adj from current_graph
            adj = [0] * n
            for u, v in current_graph:
                adj[u] |= vertex_masks[v]
                adj[v] |= vertex_masks[u]
            
            # Try removing each edge and see if we can add more
            edges_list = list(current_graph)
            random.shuffle(edges_list)
            
            for rem_edge in edges_list:
                if time.time() > deadline:
                    break
                ru, rv = rem_edge
                # Remove
                adj[ru] &= ~vertex_masks[rv]
                adj[rv] &= ~vertex_masks[ru]
                current_graph.discard(rem_edge)
                
                # Try to add any missing edge
                added_any = False
                for add_edge in all_edges:
                    if add_edge in current_graph:
                        continue
                    au, av = add_edge
                    adj[au] |= vertex_masks[av]
                    adj[av] |= vertex_masks[au]
                    if not creates_q3(adj, au, av):
                        current_graph.add(add_edge)
                        added_any = True
                    else:
                        adj[au] &= ~vertex_masks[av]
                        adj[av] &= ~vertex_masks[au]
                
                if len(current_graph) > best_edges:
                    best_edges = len(current_graph)
                    best_graph = set(current_graph)
                
                # If we couldn't add more, put the edge back
                if not added_any or len(current_graph) <= best_edges:
                    adj[ru] |= vertex_masks[rv]
                    adj[rv] |= vertex_masks[ru]
                    current_graph.add(rem_edge)
        
        # Rebuild adj for next iteration
        adj = [0] * n
        for u, v in current_graph:
            adj[u] |= vertex_masks[v]
            adj[v] |= vertex_masks[u]
        
        if num_restarts > 2000:
            break
    
    return [[u, v] for u, v in best_graph]
