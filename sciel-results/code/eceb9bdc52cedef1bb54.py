import sys
import time
import random

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # 1. Build Edge List
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    # 2. Build Conflict Graph
    # Two edges conflict if they are in the same face.
    # We will build the complement graph (non-conflict graph) for Max Clique.
    # adj[i] is a bitmask of edges j that do NOT conflict with i.
    
    # First, find all faces and map edge -> set of conflicting edges
    conflict = [set() for _ in range(n_edges)]
    
    # Enumerate faces
    # Face defined by dimensions i, j and mask of other d-2 bits
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_edges = [e1, e2, e3, e4]
                # All pairs in this face conflict
                for a in range(4):
                    for b in range(a+1, 4):
                        conflict[face_edges[a]].add(face_edges[b])
                        conflict[face_edges[b]].add(face_edges[a])

    # Build adjacency bitsets for the complement graph (non-conflict)
    # adj[i] has bit j set if i and j do NOT conflict and i != j
    adj = [0] * n_edges
    for i in range(n_edges):
        mask = 0
        for j in range(n_edges):
            if i != j and j not in conflict[i]:
                mask |= (1 << j)
        adj[i] = mask

    # 3. Max Clique Solver on Complement Graph
    sys.setrecursionlimit(10000)
    
    # Initial vertex ordering: sort by degree in complement graph (ascending)
    # Low degree in complement = High degree in conflict = More constrained?
    # Actually, Tomita's algorithm usually benefits from ordering vertices by degree in the graph being solved (complement).
    # Let's try ordering by degree in complement (ascending).
    degrees = [bin(adj[i]).count('1') for i in range(n_edges)]
    order = sorted(range(n_edges), key=lambda x: degrees[x])
    
    # Map original indices to ordered indices? 
    # It's easier to just permute the adjacency matrix.
    # new_adj[new_i] corresponds to original vertex order[new_i]
    new_adj = [0] * n_edges
    for new_i, orig_i in enumerate(order):
        mask = 0
        for new_j, orig_j in enumerate(order):
            if (adj[orig_i] >> orig_j) & 1:
                mask |= (1 << new_j)
        new_adj[new_i] = mask
    
    adj = new_adj
    n = n_edges
    best_clique = []
    best_size = 0
    
    # Time limit guard
    start_time = time.time()
    timeout = 145 # 5s margin
    
    def greedy_coloring(C):
        """
        C: integer bitmask of vertices
        Returns: (max_color, color_of_dict)
        color_of_dict maps vertex index to color (1-based)
        """
        P = C
        k = 0
        color_of = {}
        while P:
            k += 1
            Q = P
            while Q:
                # Pick a vertex from Q
                # Heuristic: pick the one with lowest index? Or random?
                # Low index is deterministic and fast.
                # v = (Q & -Q).bit_length() - 1
                v = (Q & -Q).bit_length() - 1
                color_of[v] = k
                # Remove v and its neighbors from Q
                # Neighbors in complement graph are adj[v]
                Q &= ~(adj[v] | (1 << v))
                P &= ~(1 << v)
        return k, color_of

    def expand(C, S):
        nonlocal best_clique, best_size
        
        if time.time() - start_time > timeout:
            return
            
        if C == 0:
            if len(S) > best_size:
                best_size = len(S)
                best_clique = list(S)
            return
            
        k, color_of = greedy_coloring(C)
        
        if len(S) + k <= best_size:
            return
            
        # Order vertices in C by color
        # We want to iterate from highest color to lowest? 
        # Tomita's: "Sort the vertices in C in non-increasing order of their colors"
        # Wait, if we pick a vertex with high color, the remaining C has lower potential.
        # The bound check `if len(S) + color_of[v] <= best_size` implies we should check vertices with *lower* color first?
        # No. The condition `len(S) + color_of[v] <= best_size` is used to BREAK the loop.
        # So we iterate such that `color_of[v]` decreases.
        # If we start with high color, we might not break early.
        # Standard Tomita: Order by color descending.
        # Then for each v:
        #   if |S| + color[v] <= best: break
        #   recurse
        #   C = C - {v}
        
        vertices_in_C = [i for i in range(n) if (C >> i) & 1]
        # Sort by color descending
        vertices_in_C.sort(key=lambda x: color_of[x], reverse=True)
        
        for v in vertices_in_C:
            if time.time() - start_time > timeout:
                return
                
            if len(S) + color_of[v] <= best_size:
                break
                
            # New candidate set: C intersect neighbors of v
            new_C = C & adj[v]
            # Add v to S
            S.add(v)
            expand(new_C, S)
            S.remove(v)
            
            # Remove v from C
            C &= ~(1 << v)
            
            # Optional: Update C for next iteration? 
            # The standard algorithm updates C = C - {v} at the end of the loop.
            # But we are using a pre-sorted list. 
            # The `C &= ~(1 << v)` ensures that subsequent `new_C` calculations 
            # don't include v, which is correct.
            
    # Initial call
    # C is all vertices
    initial_C = (1 << n) - 1
    expand(initial_C, set())
    
    # Map back to original edge indices
    # best_clique contains indices in the `order` permutation
    selected_edges = [order[i] for i in best_clique]
    
    # 4. Final Polish: Greedy Add
    # The B&B finds a maximum clique in the complement, which is a maximum independent set in the conflict graph.
    # This is a valid solution. 
    # However, due to time limits or pruning, it might not be the absolute maximum if the search was cut short.
    # We can try to add more edges greedily to the found solution to see if we can improve.
    
    # Convert selected_edges to a set for fast lookup
    selected_set = set(selected_edges)
    
    # Check validity and try to add
    # We need to check if an edge can be added without creating a full face.
    # An edge e can be added if it is not in selected_set and 
    # for every face containing e, the other 3 edges are not all in selected_set.
    
    # Precompute face membership for each edge
    edge_faces = [[] for _ in range(n_edges)]
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_id = len(faces)
                faces.append([e1, e2, e3, e4])
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
    
    # Greedy add
    improved = True
    while improved:
        improved = False
        for e in range(n_edges):
            if e not in selected_set:
                can_add = True
                for f in edge_faces[e]:
                    # Check if the other 3 edges in this face are selected
                    count = 0
                    for fe in faces[f]:
                        if fe == e:
                            continue
                        if fe in selected_set:
                            count += 1
                    if count == 3:
                        can_add = False
                        break
                if can_add:
                    selected_set.add(e)
                    improved = True
                    if time.time() - start_time > timeout:
                        break
        if time.time() - start_time > timeout:
            break
            
    result = []
    for e in selected_set:
        result.append(list(edges[e]))
        
    return result

