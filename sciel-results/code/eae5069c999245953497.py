"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 115.0  # 5 second safety margin
    
    d = 7
    n_vertices = 1 << d
    total_edges = d * (1 << (d - 1))  # 448
    
    # Build edge list with canonical form (u < v)
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    n_edges = len(all_edges)
    
    # Build edge index for O(1) lookup
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
    
    # Precompute: for each edge, which other edges share a 2-face with it
    # A 2-face is determined by two distinct dimensions i, j and two vertices that differ in those bits
    # Edge e = (u, v) where v = u ^ (1 << i). 
    # The other edges on the 2-face containing e are:
    #   For each j != i:
    #     c = u ^ (1 << j), e2 = v ^ (1 << j)
    #     Edge (c, e2) is on the same face
    #     Also edges (u, c) and (v, e2) are on the same face, but those are the "parallel" edges
    #     
    # Actually, let me think more carefully. A 4-cycle (2-face) has 4 edges.
    # Given edge (u, v) with v = u ^ (1 << i), and another dimension j != i:
    # The face has vertices: u, v=u^(1<<i), w=u^(1<<j), z=u^(1<<i)^(1<<j)
    # Edges: (u,v), (u,w), (v,z), (w,z)
    # So the other three edges are: (u,w), (v,z), (w,z)
    # Note: (u,w) has u<w or w<u, (v,z) has v<z or z<v, (w,z) has w<z or z<w
    
    # For checking if adding edge e would create a C4, we need to check all faces containing e
    # A face is complete if all 4 edges are present.
    # When adding e, it would complete a face if the other 3 edges of that face are already in S.
    
    # Precompute for each edge index, the list of (other_edge_idx1, other_edge_idx2, other_edge_idx3)
    # for each face containing it. Actually, we need to check if ALL 3 others are in S.
    # Better: for each edge, precompute the list of faces (as tuples of 4 edge indices) it belongs to.
    # But there are C(7,2) = 21 faces per edge... wait no.
    
    # For edge (u, v) with dimension i, and for each j != i, there's exactly one 2-face containing this edge
    # where the face is spanned by dimensions i and j. So each edge is in (d-1) = 6 faces.
    
    # For edge e_idx with endpoints u, v and dimension i:
    #   For each j != i:
    #     w = u ^ (1 << j)
    #     z = v ^ (1 << j)
    #     Other edges: (u,w), (v,z), (w,z) - need canonical forms
    
    edge_faces = [[] for _ in range(n_edges)]
    for e_idx, (u, v) in enumerate(all_edges):
        # Find dimension
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            w = u ^ (1 << j)
            z = v ^ (1 << j)
            # Get canonical edge indices for the other 3 edges
            ew = (min(u, w), max(u, w))
            evz = (min(v, z), max(v, z))
            ewz = (min(w, z), max(w, z))
            
            ew_idx = edge_index.get(ew)
            evz_idx = edge_index.get(evz)
            ewz_idx = edge_index.get(ewz)
            
            if ew_idx is not None and evz_idx is not None and ewz_idx is not None:
                edge_faces[e_idx].append((ew_idx, evz_idx, ewz_idx))
    
    def greedy_init():
        """Greedy construction."""
        S = set()
        for e_idx in range(n_edges):
            u, v = all_edges[e_idx]
            # Check if adding this edge completes any C4
            blocked = False
            for (a, b, c) in edge_faces[e_idx]:
                if a in S and b in S and c in S:
                    blocked = True
                    break
            if not blocked:
                S.add(e_idx)
        return S
    
    def is_blocked(S, e_idx):
        """Check if edge e_idx can be added to S without creating a C4."""
        for (a, b, c) in edge_faces[e_idx]:
            if a in S and b in S and c in S:
                return True
        return False
    
    def local_search(S, deadline):
        """Local search to improve the edge set."""
        S = set(S)
        best_size = len(S)
        
        # First, try to add any edge that's not blocked
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for e_idx in range(n_edges):
                if e_idx not in S and not is_blocked(S, e_idx):
                    S.add(e_idx)
                    changed = True
                    best_size = len(S)
        
        # Now try 1-for-1 swaps and 1-for-2 moves
        while time.time() < deadline - 10:
            improved = False
            
            # Try 1-for-1 swaps: remove one edge, add another
            for rem_idx in range(n_edges):
                if rem_idx not in S:
                    continue
                S.remove(rem_idx)
                
                # Try to add any edge that's now unblocked
                for add_idx in range(n_edges):
                    if add_idx in S:
                        continue
                    if not is_blocked(S, add_idx):
                        S.add(add_idx)
                        improved = True
                        break
                
                if not improved:
                    # Try to add two edges (1-for-2 net gain of +1)
                    added_any = False
                    for add_idx in range(n_edges):
                        if add_idx in S:
                            continue
                        if not is_blocked(S, add_idx):
                            S.add(add_idx)
                            added_any = True
                            break
                    
                    if added_any:
                        # Try to add another
                        for add_idx2 in range(n_edges):
                            if add_idx2 in S:
                                continue
                            if not is_blocked(S, add_idx2):
                                S.add(add_idx2)
                                improved = True
                                break
                        if not improved:
                            # Revert the single add if it didn't help net
                            # Actually we added 1, removed 1, so net 0. 
                            # But if we can add a second one, net +1
                            # If we only added one, we're back to same size.
                            # Let's keep it for now (might help future moves)
                            pass
                    else:
                        # Couldn't add anything, revert
                        S.add(rem_idx)
                
                if improved:
                    best_size = len(S)
                    break
            
            if not improved:
                # Try random restart of local search: remove a random subset and re-add
                if len(S) > 100:
                    random.seed(int(time.time() * 1000) % (2**31))
                    remove_count = max(5, len(S) // 10)
                    to_remove = random.sample(list(S), min(remove_count, len(S)))
                    for r in to_remove:
                        S.discard(r)
                    # Re-run greedy add
                    for e_idx in range(n_edges):
                        if e_idx not in S and not is_blocked(S, e_idx):
                            S.add(e_idx)
                    best_size = len(S)
                    improved = True  # Trigger another iteration
        
        return S
    
    # Run greedy init
    S = greedy_init()
    
    # Local search
    S = local_search(S, deadline)
    
    # Multiple restarts with different random seeds
    best_S = set(S)
    
    if time.time() < deadline - 30:
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            random.seed(seed)
            
            # Create a perturbed initial solution
            S_trial = set(S)
            
            # Randomly remove some edges
            remove_count = max(10, len(S_trial) // 8)
            to_remove = random.sample(list(S_trial), min(remove_count, len(S_trial)))
            for r in to_remove:
                S_trial.discard(r)
            
            # Re-add greedily
            for e_idx in range(n_edges):
                if e_idx not in S_trial and not is_blocked(S_trial, e_idx):
                    S_trial.add(e_idx)
            
            # Local search
            S_trial = local_search(S_trial, deadline)
            
            if len(S_trial) > len(best_S):
                best_S = set(S_trial)
            
            if time.time() > deadline - 15:
                break
    
    # Convert to list of [u, v] pairs
    result = []
    for e_idx in best_S:
        u, v = all_edges[e_idx]
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
