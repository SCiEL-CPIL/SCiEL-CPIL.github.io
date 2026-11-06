import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    d = 7
    n_vertices = 1 << d
    num_edges_total = n_vertices * d // 2  # 448
    
    # 1. Precompute edges
    all_edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                
    num_edges = len(all_edges)
    
    # 1b. Precompute edge_faces and face_coverers
    edge_faces = [[] for _ in range(num_edges)]
    face_coverers = defaultdict(set)
    
    # Build face -> edges mapping
    face_edges_map = {}  # frozenset of 4 edge indices -> face index
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v0 = u
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v0, v1), max(v0, v1))
                e2 = (min(v0, v2), max(v0, v2))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v3), max(v2, v3))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_key = frozenset([idx1, idx2, idx3, idx4])
                
                if face_key not in face_edges_map:
                    face_idx = len(face_edges_map)
                    face_edges_map[face_key] = face_idx
                    for e_idx in face_key:
                        edge_faces[e_idx].append(face_idx)
                        face_coverers[face_idx].add(e_idx)
    
    num_faces = len(face_edges_map)
    
    # 3. Greedy Initialization of Hitting Set R
    # R is the set of edges NOT in the solution.
    # We want to minimize |R|.
    # Initialize R as empty? No, we need to cover all faces.
    # Greedy: Pick edge that covers max uncovered faces.
    
    # State:
    # in_R[e]: boolean, is edge e in R?
    # cover_count[f]: number of edges in R covering face f.
    
    in_R = [False] * num_edges
    cover_count = [0] * num_faces
    uncovered_faces = set(range(num_faces))
    
    # Greedy step
    while uncovered_faces:
        best_edge = -1
        max_cov = 0
        # To speed up, we can precompute for each edge the list of faces.
        # Iterate through all edges not in R? Or all edges?
        # Since we add to R, we only consider edges not in R.
        # Optimization: Only consider edges that cover at least one uncovered face.
        
        # Simple scan
        for e in range(num_edges):
            if in_R[e]:
                continue
            cov = 0
            for f in edge_faces[e]:
                if f in uncovered_faces:
                    cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_edge = e
                        if max_cov == len(uncovered_faces):
                            break
            if max_cov == len(uncovered_faces):
                break
                
        if best_edge == -1:
            # Should not happen if graph is connected and faces exist
            break
            
        in_R[best_edge] = True
        for f in edge_faces[best_edge]:
            if f in uncovered_faces:
                uncovered_faces.remove(f)
            cover_count[f] += 1
            
    # 4. Local Search on R
    # Objective: Minimize |R| (which is sum(in_R))
    # Moves:
    # 1. Deletion: If e in R, and for all f in edge_faces[e], cover_count[f] >= 2, remove e.
    # 2. Swap: If e in R, v not in R. 
    #    Check if R - {e} + {v} is valid.
    #    Condition: For all f in edge_faces[e], if cover_count[f] == 1, then v must cover f.
    
    current_size = sum(in_R)
    best_size = current_size
    best_in_R = in_R[:]
    
    # Helper to update cover counts
    def add_to_R(e):
        in_R[e] = True
        for f in edge_faces[e]:
            cover_count[f] += 1
            
    def remove_from_R(e):
        in_R[e] = False
        for f in edge_faces[e]:
            cover_count[f] -= 1
            
    # Precompute list of edges in R and not in R for faster iteration?
    # We can maintain lists, but updating them is O(N).
    # Given N=448, scanning is fine.
    
    # Shuffle for randomization
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try deletions
        # Iterate through edges in R
        # To avoid bias, we can shuffle the order of checking deletions
        edges_in_R = [e for e in range(num_edges) if in_R[e]]
        rng.shuffle(edges_in_R)
        
        for e in edges_in_R:
            if time.time() > deadline:
                break
            # Check if e can be removed
            # Condition: min(cover_count[f] for f in edge_faces[e]) >= 2
            can_remove = True
            for f in edge_faces[e]:
                if cover_count[f] < 2:
                    can_remove = False
                    break
            
            if can_remove:
                remove_from_R(e)
                current_size -= 1
                improved = True
                if current_size < best_size:
                    best_size = current_size
                    best_in_R = in_R[:]
        
        # Phase 2: Try swaps
        # If no deletion improved, try swaps to potentially enable deletions
        # Or just try swaps always to explore landscape.
        
        # Get list of edges in R and not in R
        edges_in_R = [e for e in range(num_edges) if in_R[e]]
        edges_not_in_R = [e for e in range(num_edges) if not in_R[e]]
        
        if not edges_in_R or not edges_not_in_R:
            break
            
        rng.shuffle(edges_in_R)
        rng.shuffle(edges_not_in_R)
        
        # Try a limited number of swaps to save time
        swaps_tried = 0
        max_swaps = 2000  # Increased from 1000 since swaps are now faster
        found_swap_improvement = False
        
        for e in edges_in_R:
            if time.time() > deadline:
                break
            if swaps_tried >= max_swaps:
                break
                
            # For a given e in R, find a v not in R that allows swap
            # We can pick random v's
            
            # Optimization: 
            # Let critical_faces = [f for f in edge_faces[e] if cover_count[f] == 1]
            # If critical_faces is empty, we can just delete e (handled in Phase 1)
            # So here critical_faces is not empty.
            # We need v such that for all f in critical_faces, v covers f.
            
            critical_faces = []
            for f in edge_faces[e]:
                if cover_count[f] == 1:
                    critical_faces.append(f)
                    
            if not critical_faces:
                # Should have been caught in Phase 1, but maybe state changed?
                # If cover count changed, re-check? 
                # Actually, if we are in swap phase, we assume no deletions happened 
                # in this specific iteration loop, or we re-evaluated.
                # If critical_faces is empty, we can just delete.
                # But Phase 1 already tried. Maybe it failed due to order?
                # If it failed, it means some cover_count was 1.
                # So critical_faces should not be empty if deletion failed.
                pass
                
            # Try random v from edges_not_in_R
            # To be efficient, we can check if v covers all critical_faces.
            # Use face_coverers for O(1) lookup instead of O(|edge_faces[v]|)
            
            for v in edges_not_in_R:
                if time.time() > deadline:
                    break
                swaps_tried += 1
                
                # Check if v covers all critical faces using face_coverers
                valid = True
                for f in critical_faces:
                    if v not in face_coverers[f]:
                        valid = False
                        break
                        
                if valid:
                    # Perform swap
                    # Remove e
                    remove_from_R(e)
                    # Add v
                    add_to_R(v)
                    # Size remains same
                    improved = True
                    found_swap_improvement = True
                    # After swap, cover counts changed.
                    # This might enable new deletions.
                    # Break to restart the loop? 
                    # Or just continue? 
                    # Restarting Phase 1 is better.
                    break
            
            if found_swap_improvement:
                break
                
        if not improved and not found_swap_improvement:
            # Stuck. Try a random restart or perturbation?
            # Perturbation: Remove a random edge from R and add a random edge not in R, 
            # even if invalid, then repair?
            # Simpler: Just break and rely on best found.
            # Or, do a random swap regardless of validity, then repair by adding edges to R?
            # Adding to R increases size, which we don't want.
            # Removing from R decreases size but might break validity.
            
            # Let's try a "kick": 
            # Pick random e in R, remove it (making state invalid).
            # Then greedily add edges to R to restore validity?
            # This is like a restart from a slightly perturbed state.
            
            if current_size > 0:
                # Randomly remove one edge from R
                e_rand = rng.choice(edges_in_R)
                remove_from_R(e_rand)
                current_size -= 1
                # Now state is invalid. Some faces have cover_count 0.
                # We need to add edges to cover them.
                # Greedy repair:
                # Find all faces with cover_count 0.
                uncovered = set()
                for f in range(num_faces):
                    if cover_count[f] == 0:
                        uncovered.add(f)
                        
                while uncovered:
                    # Find edge covering max uncovered
                    best_e = -1
                    max_cov = 0
                    for e in range(num_edges):
                        if in_R[e]:
                            continue
                        cov = 0
                        for f in edge_faces[e]:
                            if f in uncovered:
                                cov += 1
                        if cov > max_cov:
                            max_cov = cov
                            best_e = e
                    if best_e == -1:
                        break
                    add_to_R(best_e)
                    current_size += 1
                    for f in edge_faces[best_e]:
                        if f in uncovered:
                            uncovered.remove(f)
                            
                if current_size < best_size:
                    best_size = current_size
                    best_in_R = in_R[:]
                    
            else:
                break

    # Construct final result
    # Solution is E \ R
    # We want list of edges [u, v]
    result = []
    for i in range(num_edges):
        if not best_in_R[i]:
            result.append(list(all_edges[i]))
            
    return result

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges")
