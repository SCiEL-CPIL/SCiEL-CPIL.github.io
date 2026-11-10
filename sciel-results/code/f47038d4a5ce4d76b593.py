import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1)) # 448
    
    # Generate all edges
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx

    # Generate all faces (2D squares)
    # A face is defined by two distinct dimensions i, j and a vertex u 
    # such that the 4 vertices are u, u^(1<<i), u^(1<<j), u^(1<<i)^(1<<j)
    # The 4 edges are:
    # (u, u^(1<<i)), (u^(1<<j), u^(1<<i)^(1<<j))
    # (u, u^(1<<j)), (u^(1<<i), u^(1<<i)^(1<<j))
    
    faces = []
    edge_to_faces = [[] for _ in range(n_edges)]
    
    # To avoid duplicates, we can iterate over all pairs of dimensions and all vertices,
    # but each face is counted 4 times (once per vertex). 
    # We can just generate all and use a set of frozensets of edge indices to deduplicate,
    # or generate uniquely.
    # Unique generation: 
    # For each pair of dims (i, j) with i < j:
    #   For each vertex u:
    #     The face is unique if we fix a canonical representative, e.g., the smallest vertex.
    #     Or simply, the set of 4 edges is unique.
    
    seen_faces = set()
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n_vertices):
                # The 4 vertices
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                # The 4 edges
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v3, v4), max(v3, v4))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v4), max(v2, v4))
                
                # Get indices
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_tuple = tuple(sorted([idx1, idx2, idx3, idx4]))
                
                if face_tuple not in seen_faces:
                    seen_faces.add(face_tuple)
                    faces.append(face_tuple)
                    
                    for idx in face_tuple:
                        edge_to_faces[idx].append(len(faces) - 1)

    n_faces = len(faces) # Should be 672
    
    def greedy_hitting_set(randomize_ties=False):
        R = set()
        unhit_faces = set(range(n_faces))
        
        # Precompute face edges for fast access
        # faces is list of tuples of 4 edge indices
        
        while unhit_faces:
            # Find an unhit face
            # If randomize, pick a random unhit face, else pick first
            if randomize_ties and unhit_faces:
                f_idx = random.choice(list(unhit_faces))
            else:
                f_idx = next(iter(unhit_faces))
                
            face = faces[f_idx]
            
            # Count for each edge in this face, how many OTHER unhit faces it touches
            best_edge = -1
            max_count = -1
            counts = {}
            for e in face:
                count = 0
                for f_idx2 in edge_to_faces[e]:
                    if f_idx2 in unhit_faces:
                        count += 1
                counts[e] = count
                if count > max_count:
                    max_count = count
                    best_edge = e
            
            # Tie breaking: if randomize, pick random among max
            if randomize_ties and max_count > 0:
                candidates = [e for e in face if counts[e] == max_count]
                best_edge = random.choice(candidates)
            
            R.add(best_edge)
            
            # Update unhit faces
            # Remove all faces containing best_edge from unhit_faces
            for f_idx2 in edge_to_faces[best_edge]:
                if f_idx2 in unhit_faces:
                    unhit_faces.remove(f_idx2)
                    
        return R

    def cleanup(R):
        R = set(R)
        changed = True
        while changed:
            changed = False
            for e in list(R):
                # Check if removing e leaves any face uncovered
                # A face is covered if it contains at least one edge from R
                # If we remove e, a face f is uncovered if e was the ONLY edge from R in f.
                # So, check all faces containing e.
                is_redundant = True
                for f_idx in edge_to_faces[e]:
                    face = faces[f_idx]
                    # Check if any other edge in this face is in R
                    has_other = False
                    for e2 in face:
                        if e2 != e and e2 in R:
                            has_other = True
                            break
                    if not has_other:
                        is_redundant = False
                        break
                
                if is_redundant:
                    R.remove(e)
                    changed = True
        return R

    def local_search(R, time_limit):
        # Try to improve R by removing an edge and re-optimizing the gaps
        # This is expensive, so we do it sparingly or with limited depth
        # For now, just do cleanup and maybe some random swaps?
        # Given time constraints, Greedy + Cleanup is a strong baseline.
        # Let's add a simple "swap" heuristic:
        # Pick an edge e in R. Remove it. 
        # This uncovers some faces.
        # Can we add a different edge e' not in R that covers all uncovered faces?
        # If yes, and if e' allows us to remove more edges later?
        # This is complex. Let's stick to multiple random greedy restarts + cleanup.
        pass

    best_R_size = float('inf')
    best_R = set()
    
    # Run multiple random restarts
    num_restarts = 2000 # Adjust based on time
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Occasionally use deterministic, mostly random
        use_random = (i % 10 != 0)
        R = greedy_hitting_set(randomize_ties=use_random)
        R = cleanup(R)
        
        if len(R) < best_R_size:
            best_R_size = len(R)
            best_R = R.copy()
            
        # If we hit the theoretical lower bound (448 - 304 = 144), we can stop early?
        # Best known ex(Q7, C4) is 304, so min removals is 144.
        if best_R_size == 144:
            break

    # Construct the final edge list
    # The selected edges are all edges NOT in best_R
    selected_edges = []
    for idx, (u, v) in enumerate(edges):
        if idx not in best_R:
            selected_edges.append([u, v])
            
    return selected_edges
