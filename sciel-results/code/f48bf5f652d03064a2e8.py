import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    A = {0, 1, 2}
    B = set(range(3, 16))
    
    edges = set()
    for a in A:
        for b in B:
            edges.add((a, b))
            
    missing = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edges:
                missing.append((i, j))
                
    random.shuffle(missing)
    
    # Q3 adjacency
    # 0:000, 1:001, 2:010, 3:011, 4:100, 5:101, 6:110, 7:111
    q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5), (1, 7),
        (2, 3), (2, 6), (2, 7),
        (3, 7), (4, 5), (4, 6),
        (5, 7), (6, 7)
    ]
    # Wait, 12 edges.
    # 0-1, 0-2, 0-4
    # 1-3, 1-5, 1-7
    # 2-3, 2-6, 2-7
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    # Count: 3+3+3+1+2+1+1 = 14?
    # Q3 has 12 edges.
    # Let's list properly.
    # 000 connected to 001, 010, 100
    # 001 connected to 000, 011, 101
    # 010 connected to 000, 011, 110
    # 011 connected to 001, 010, 111
    # 100 connected to 000, 101, 110
    # 101 connected to 001, 100, 111
    # 110 connected to 010, 100, 111
    # 111 connected to 011, 101, 110
    
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    
    def has_q3(current_edges, new_edge):
        u, v = new_edge
        # Check if adding (u,v) creates a Q3
        # Build adj
        adj = {i: set() for i in range(n)}
        for x, y in current_edges:
            adj[x].add(y)
            adj[y].add(x)
        adj[u].add(v)
        adj[v].add(u)
        
        # Iterate over 6 other vertices
        others = [x for x in range(n) if x != u and x != v]
        
        for subset in itertools.combinations(others, 6):
            S = list(subset) + [u, v]
            S_set = set(S)
            
            # Check if Q3 is a subgraph of G[S]
            # We can check by trying to map Q3 vertices to S
            # such that edges are preserved.
            
            # Optimization: 
            # u and v are adjacent in G[S].
            # So in Q3, they must be mapped to adjacent vertices.
            # There are 12 edges in Q3.
            # Let's try each edge (a, b) in Q3 as the image of (u, v).
            
            found = False
            for a, b in q3_edges:
                # Map a -> u, b -> v
                # Remaining Q3 vertices: 6
                # Remaining S vertices: 6
                # We need to map the rest such that edges match.
                
                # This is a backtracking search.
                # Given the small size, it's fast.
                
                def backtrack(remaining_q3_verts, remaining_s_verts, mapping):
                    if not remaining_q3_verts:
                        return True
                    
                    # Pick a vertex from remaining_q3_verts
                    qv = remaining_q3_verts[0]
                    rem_q3 = remaining_q3_verts[1:]
                    
                    for sv in remaining_s_verts:
                        # Check if qv can be mapped to sv
                        # All neighbors of qv in Q3 that are already mapped
                        # must have their images adjacent to sv in G[S]
                        
                        valid = True
                        for q_neighbor in q3_adj[qv]:
                            if q_neighbor in mapping:
                                mapped_neighbor = mapping[q_neighbor]
                                if mapped_neighbor not in adj[sv]:
                                    valid = False
                                    break
                            
                        if valid:
                            # Also check if sv is in S (it is)
                            # And if the edge (mapped, sv) exists in G
                            # adj[sv] contains all neighbors in G[S]
                            
                            new_mapping = mapping.copy()
                            new_mapping[qv] = sv
                            if backtrack(rem_q3, [x for x in remaining_s_verts if x != sv], new_mapping):
                                return True
                    return False

                rem_q3 = [x for x in range(8) if x != a and x != b]
                rem_s = [x for x in S if x != u and x != v]
                
                initial_mapping = {a: u, b: v}
                if backtrack(rem_q3, rem_s, initial_mapping):
                    return True
                    
            # Try swapping u and v
            # i.e., a -> v, b -> u
            for a, b in q3_edges:
                rem_q3 = [x for x in range(8) if x != a and x != b]
                rem_s = [x for x in S if x != u and x != v]
                
                initial_mapping = {a: v, b: u}
                if backtrack(rem_q3, rem_s, initial_mapping):
                    return True

        return False

    added = 0
    for e in missing:
        if time.time() > deadline:
            break
        if not has_q3(edges, e):
            edges.add(e)
            added += 1
            if added >= 20: # Stop early if we have enough
                break
                
    return [list(e) for e in edges]
