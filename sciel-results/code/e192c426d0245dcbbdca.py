import time
import random
import numpy as np

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))  # 448
    
    # Build edge list
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
    
    # Identify faces (2D squares)
    # A face is determined by two dimensions (i, j) and the values of the other d-2 bits
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
                    
    n_faces = len(faces)
    
    # Precompute for each face, the 4 edge indices
    face_edges = np.array(faces, dtype=np.int32)  # shape (n_faces, 4)
    # edge_faces is list of lists, but let's also make a numpy-friendly version if needed
    
    # Strategy: We want to minimize the number of removed edges.
    # This is a hitting set problem.
    # We start with a greedy hitting set, then use local search with multi-edges swaps.
    
    # 1. Greedy Initialization
    removed = set()
    face_counts = np.zeros(n_faces, dtype=np.int32)
    
    def add_to_removed(e):
        removed.add(e)
        for f in edge_faces[e]:
            face_counts[f] += 1
            
    def remove_from_removed(e):
        removed.remove(e)
        for f in edge_faces[e]:
            face_counts[f] -= 1
            
    uncovered_faces = set(range(n_faces))
    
    while uncovered_faces:
        # Pick a random uncovered face
        f = random.choice(list(uncovered_faces))
        
        best_e = -1
        max_cover = -1
        for e in faces[f]:
            if e in removed:
                continue
            cover = 0
            for ff in edge_faces[e]:
                if ff in uncovered_faces:
                    cover += 1
            if cover > max_cover:
                max_cover = cover
                best_e = e
        
        if best_e == -1:
            best_e = faces[f][0]
            
        add_to_removed(best_e)
        
        for ff in edge_faces[best_e]:
            if face_counts[ff] == 1:
                uncovered_faces.discard(ff)
                
    best_removed = set(removed)
    best_size = len(removed)
    
    # 2. Local Search with Multi-Swaps
    start_time = time.time()
    deadline = 115  # Safety margin
    
    # Helper to check if a set of edges is a valid hitting set
    # We maintain face_counts, so we just need to ensure no face has count 0.
    
    # Phase 1: Deletions
    # If an edge is not critical (i.e., all faces it covers have count > 1), remove it.
    def try_deletions():
        improved = False
        for e in list(removed):
            critical = False
            for f in edge_faces[e]:
                if face_counts[f] == 1:
                    critical = True
                    break
            if not critical:
                remove_from_removed(e)
                improved = True
        return improved
    
    # Phase 2: Swaps
    # Try to swap e_out with e_in.
    # e_out must be in removed.
    # e_in must not be in removed.
    # After swap, all faces must still be covered.
    # This is equivalent to: for every face f covered by e_out, if face_counts[f] == 1, then e_in must also cover f.
    # i.e., e_in must be in the intersection of all faces that are critical for e_out.
    
    def try_swaps():
        improved = False
        # Iterate over a copy of removed
        for e_out in list(removed):
            critical_faces = []
            for f in edge_faces[e_out]:
                if face_counts[f] == 1:
                    critical_faces.append(f)
            
            if not critical_faces:
                # Should have been deleted
                continue
                
            # Find candidates for e_in
            # Candidates must cover all critical_faces
            # Start with edges of the first critical face
            candidates = set(faces[critical_faces[0]])
            for f in critical_faces[1:]:
                candidates &= set(faces[f])
            
            candidates.discard(e_out)
            
            valid_candidates = [e for e in candidates if e not in removed]
            
            if valid_candidates:
                # Pick a random candidate
                e_in = random.choice(valid_candidates)
                
                # Check if this swap improves the "potential" or just keeps it valid?
                # Actually, we want to minimize size. A 1-for-1 swap keeps size same.
                # It might allow future deletions.
                # Let's just do it if it's valid.
                remove_from_removed(e_out)
                add_to_removed(e_in)
                improved = True
                
                # Optional: Try deletions immediately after a swap?
                # It might be efficient to do a quick deletion pass.
                # But let's keep the phases separate for now.
                
        return improved
    
    # Phase 3: Multi-Swaps (2-for-1, 3-for-1)
    # This is more complex.
    # 2-for-1: Remove 2 edges, add 1 edge.
    # This reduces size by 1 if valid.
    # Conditions:
    # Let R be the set of 2 edges to remove.
    # Let A be the set of 1 edge to add.
    # For any face f:
    #   If f was covered by k edges in R, and 0 edges in A:
    #     New count = old_count - k. Must be >= 1.
    #     So old_count >= k + 1.
    #   If f was covered by k edges in R, and l edges in A:
    #     New count = old_count - k + l. Must be >= 1.
    
    # Heuristic for 2-for-1:
    # Pick two edges e1, e2 in removed.
    # Identify faces that become uncovered or critical.
    # Specifically, faces where count_R(f) == 1 and f is covered by e1 or e2.
    # Wait, if we remove e1 and e2, any face f where count(f) == 1 and (e1 in f or e2 in f) will become 0.
    # So we need an edge e_in that covers ALL such faces.
    # Let Critical_Faces = { f | count(f) == 1 and (e1 in f or e2 in f) }
    # e_in must be in intersection of all faces in Critical_Faces.
    # Also, we need to ensure that for any face g where count(g) == 1 and e1 not in g and e2 not in g,
    # the count remains 1 (so e_in can cover it or not, doesn't matter as long as it doesn't drop to 0).
    # Actually, if e_in covers g, count becomes 2. If not, count stays 1. Both are valid.
    # The only danger is faces that drop to 0.
    # So the condition is: e_in must cover all faces f such that count(f) == 1 and (e1 in f or e2 in f).
    # Additionally, we need to ensure that removing e1 and e2 doesn't drop any face to 0 that wasn't covered by e_in?
    # Yes, that's exactly the condition above.
    # What about faces with count(f) == 2?
    # If e1 in f and e2 in f, count drops by 2 -> 0. Invalid unless e_in covers f (then 1).
    # If e1 in f and e2 not in f, count drops by 1 -> 1. Valid.
    # So the condition "e_in covers all faces where count(f) == 1 and (e1 in f or e2 in f)" is NOT sufficient.
    # We also need: for any face f where count(f) == 2 and e1 in f and e2 in f, e_in must cover f.
    # Generally: Let delta_f = (1 if e1 in f else 0) + (1 if e2 in f else 0) - (1 if e_in in f else 0).
    # We need count(f) - delta_f >= 1.
    # => count(f) - 1 >= delta_f.
    
    def try_multi_swaps():
        improved = False
        # This is expensive. Limit attempts.
        # Pick random pairs of edges from removed.
        # To make it faster, we can restrict to edges that share faces.
        
        # Let's try a simpler version: 2-for-1 where the two edges share a face.
        # If e1 and e2 share a face f, and count(f) == 2, then removing them drops f to 0.
        # So e_in must cover f.
        # Other faces covered by e1 or e2:
        # If count(g) == 1 and e1 in g (e2 not in g), e_in must cover g.
        # So e_in must cover all faces in (Faces(e1) U Faces(e2)) that have count <= 2 and are affected.
        
        # Let's just try random pairs.
        if len(removed) < 2:
            return improved
            
        # Sample a few pairs
        attempts = 0
        max_attempts = 50  # per call
        
        while attempts < max_attempts and time.time() - start_time < deadline:
            e1, e2 = random.sample(list(removed), 2)
            attempts += 1
            
            # Determine required faces for e_in
            required = set()
            valid = True
            
            # Check all faces covered by e1 or e2
            all_faces = set(edge_faces[e1]) | set(edge_faces[e2])
            
            for f in all_faces:
                cnt = face_counts[f]
                covers_e1 = 1 if e1 in faces[f] else 0 # This is slow, faces[f] is list
                # Better: precompute edge-in-face?
                # For now, let's assume we can check.
                # Actually, faces[f] is a list of 4 edges.
                # So `e1 in faces[f]` is O(1).
                
                cov_e1 = 1 if e1 in faces[f] else 0
                cov_e2 = 1 if e2 in faces[f] else 0
                removals = cov_e1 + cov_e2
                
                if cnt - removals < 1:
                    # We need e_in to cover this face to restore count
                    # New count = cnt - removals + 1 (if e_in covers f)
                    # We need cnt - removals + 1 >= 1 => cnt - removals >= 0
                    # Since cnt - removals < 1, it must be 0 or negative.
                    # If cnt - removals == 0, e_in MUST cover f.
                    # If cnt - removals < 0, even with e_in, count is < 1. Invalid pair.
                    if cnt - removals < 0:
                        valid = False
                        break
                    else: # cnt - removals == 0
                        required.add(f)
                        
            if not valid:
                continue
                
            # Find e_in that covers all faces in `required`
            if not required:
                # Any edge not in removed works?
                # We want to minimize size, so 2-for-1 reduces size by 1.
                # We can pick any valid e_in.
                # To be safe, pick one that doesn't create new critical issues?
                # Actually, adding e_in increases counts. It never hurts validity.
                # So we just need e_in not in removed.
                # But we also need to ensure e_in doesn't "waste" capacity?
                # No, we just want to reduce size.
                # So pick any e_in not in removed.
                # But we must ensure e_in is a valid edge.
                # And ideally, we want e_in to be such that it doesn't prevent future moves?
                # For now, random choice.
                
                # However, if required is empty, it means removing e1, e2 doesn't drop any face to 0.
                # So we can just remove them? No, we must add 1 to keep it a "swap" in the context of hitting set?
                # Wait, the problem is to find a MINIMAL hitting set? No, just a hitting set.
                # If we can remove 2 edges and add 0, that's even better (2-for-0).
                # But a hitting set must cover all faces.
                # If removing e1, e2 leaves all faces covered, then we can just remove them.
                # This is a "deletion" of 2 edges.
                # My `try_deletions` only removes 1 at a time.
                # So this is a valid improvement.
                
                # Let's check if we can just remove them.
                # If required is empty, it means for all f, cnt(f) - removals(f) >= 1.
                # So yes, we can remove e1 and e2 without adding anything.
                remove_from_removed(e1)
                remove_from_removed(e2)
                improved = True
                continue # Restart loop? Or continue to find more?
                
            # If required is not empty, find e_in
            # e_in must be in intersection of all faces in `required`
            candidates = set(faces[list(required)[0]])
            for f in list(required)[1:]:
                candidates &= set(faces[f])
                
            candidates.discard(e1)
            candidates.discard(e2)
            
            valid_candidates = [e for e in candidates if e not in removed]
            
            if valid_candidates:
                e_in = random.choice(valid_candidates)
                # Perform swap
                remove_from_removed(e1)
                remove_from_removed(e2)
                add_to_removed(e_in)
                improved = True
                
        return improved
    
    # Main Loop
    while time.time() - start_time < deadline:
        improved = False
        
        # 1. Deletions
        if try_deletions():
            improved = True
            if len(removed) < best_size:
                best_size = len(removed)
                best_removed = set(removed)
            continue # Restart to catch new deletions
        
        # 2. Swaps (1-for-1)
        if try_swaps():
            improved = True
            # Size unchanged, but structure changed.
            # Try deletions again?
            continue
        
        # 3. Multi-Swaps (2-for-1 or 2-for-0)
        if try_multi_swaps():
            improved = True
            if len(removed) < best_size:
                best_size = len(removed)
                best_removed = set(removed)
            continue
        
        # If no improvement, break or perturb
        # Let's add a simple perturbation: add a random edge, then run deletions.
        # This increases size, but might lead to a better local minimum.
        if time.time() - start_time < deadline - 10:
            # Perturb
            # Pick a random edge not in removed
            non_removed = [e for e in range(n_edges) if e not in removed]
            if non_removed:
                e_perturb = random.choice(non_removed)
                add_to_removed(e_perturb)
                # Run deletions to see if we can get back to a smaller size
                # We don't update best here because size increased
                # But the structure might be better.
                # Let's just let the main loop handle it.
                # To avoid infinite loop of perturbations, we can count them.
                # Or just break if we've been stuck for a while.
                # For now, let's just break if no improvement in a few iterations.
                # Actually, the loop above `continue`s on improvement.
                # So if we get here, we tried deletions, swaps, multi-swaps and failed.
                # Let's do one perturbation and then break if we don't improve soon?
                # Or just break.
                pass
        else:
            break
            
    # Final Polish
    face_counts_final = np.zeros(n_faces, dtype=np.int32)
    for e in best_removed:
        for f in edge_faces[e]:
            face_counts_final[f] += 1
            
    changed = True
    while changed:
        changed = False
        for e in list(best_removed):
            critical = False
            for f in edge_faces[e]:
                if face_counts_final[f] == 1:
                    critical = True
                    break
            if not critical:
                best_removed.remove(e)
                for f in edge_faces[e]:
                    face_counts_final[f] -= 1
                changed = True
                
    result = []
    for i in range(n_edges):
        if i not in best_removed:
            result.append(list(edges[i]))
            
    return result
