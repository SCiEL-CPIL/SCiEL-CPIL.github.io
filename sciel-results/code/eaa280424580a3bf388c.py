<<<<<<< SEARCH
    def local_search(removed, face_counts, deadline):
        """Local search to minimize removed set. Returns (best_removed, best_size)."""
        current_size = len(removed)
        best_removed = set(removed)
        best_size = current_size
        best_face_counts = face_counts[:]
        
        while time.time() < deadline:
            improved = False
            
            # Phase 1: Delete non-critical edges
            for e in list(removed):
                critical = False
                for f in edge_faces[e]:
                    if face_counts[f] == 1:
                        critical = True
                        break
                if not critical:
                    removed.discard(e)
                    for f in edge_faces[e]:
                        face_counts[f] -= 1
                    current_size -= 1
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_removed = set(removed)
                        best_face_counts = face_counts[:]
            
            # Phase 2: Try swaps
            if not improved:
                # Try multiple swap attempts
                swap_attempts = 0
                max_swaps = 500
                while swap_attempts < max_swaps and time.time() < deadline:
                    swap_attempts += 1
                    if not removed:
                        break
                    
                    e_out = random.choice(list(removed))
                    
                    # Find critical faces for e_out
                    critical_faces = [f for f in edge_faces[e_out] if face_counts[f] == 1]
                    
                    if not critical_faces:
                        # Deletion candidate - should have been caught in Phase 1
                        continue
                    
                    # Find candidate edges in intersection of critical faces
                    candidate_edges = set(faces[critical_faces[0]])
                    for f in critical_faces[1:]:
                        candidate_edges &= set(faces[f])
                    
                    candidate_edges.discard(e_out)
                    valid_candidates = [e for e in candidate_edges if e not in removed]
                    
                    if valid_candidates:
                        e_in = random.choice(valid_candidates)
                        removed.discard(e_out)
                        removed.add(e_in)
                        for f in edge_faces[e_out]:
                            face_counts[f] -= 1
                        for f in edge_faces[e_in]:
                            face_counts[f] += 1
                        improved = True
                        # Size unchanged, but may enable deletions
            
            if not improved:
                # Stuck - try perturbation or break
                # Try a few random valid swaps to escape local minimum
                perturbed = False
                for _ in range(50):
                    if time.time() >= deadline:
                        break
                    if not removed:
                        break
                    e_out = random.choice(list(removed))
                    critical_faces = [f for f in edge_faces[e_out] if face_counts[f] == 1]
                    if not critical_faces:
                        continue
                    candidate_edges = set(faces[critical_faces[0]])
                    for f in critical_faces[1:]:
                        candidate_edges &= set(faces[f])
                    candidate_edges.discard(e_out)
                    valid_candidates = [e for e in candidate_edges if e not in removed]
                    if valid_candidates:
                        e_in = random.choice(valid_candidates)
                        removed.discard(e_out)
                        removed.add(e_in)
                        for f in edge_faces[e_out]:
                            face_counts[f] -= 1
                        for f in edge_faces[e_in]:
                            face_counts[f] += 1
                        perturbed = True
                        break
                if not perturbed:
                    break
        
        return best_removed, best_size
=======
    def local_search(removed, face_counts, deadline):
        """Local search to minimize removed set. Returns (best_removed, best_size)."""
        current_size = len(removed)
        best_removed = set(removed)
        best_size = current_size
        best_face_counts = face_counts[:]
        
        while time.time() < deadline:
            improved = False
            
            # Phase 1: Delete non-critical edges
            for e in list(removed):
                critical = False
                for f in edge_faces[e]:
                    if face_counts[f] == 1:
                        critical = True
                        break
                if not critical:
                    removed.discard(e)
                    for f in edge_faces[e]:
                        face_counts[f] -= 1
                    current_size -= 1
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_removed = set(removed)
                        best_face_counts = face_counts[:]
            
            # Phase 2: Try swaps - improved strategy
            if not improved:
                swap_attempts = 0
                max_swaps = 1000
                while swap_attempts < max_swaps and time.time() < deadline:
                    swap_attempts += 1
                    if not removed:
                        break
                    
                    e_out = random.choice(list(removed))
                    
                    # Find critical faces for e_out
                    critical_faces = [f for f in edge_faces[e_out] if face_counts[f] == 1]
                    
                    if not critical_faces:
                        continue
                    
                    # Strategy: Find edges that cover the most critical faces
                    # and don't create new uncovered faces
                    best_e_in = -1
                    best_score = -1
                    
                    # Get all edges in any critical face
                    candidate_pool = set()
                    for f in critical_faces:
                        candidate_pool.update(faces[f])
                    candidate_pool.discard(e_out)
                    
                    for e_in in candidate_pool:
                        if e_in in removed:
                            continue
                        
                        # Check if this swap is valid: after removing e_out and adding e_in,
                        # no face should become uncovered (count goes to 0)
                        # and we should cover as many critical faces as possible
                        new_critical_covered = 0
                        creates_uncovered = False
                        
                        for f in edge_faces[e_out]:
                            if face_counts[f] == 1:  # critical face for e_out
                                if e_in in faces[f]:
                                    new_critical_covered += 1
                            else:
                                # face_counts[f] >= 2, after removing e_out it's still >= 1
                                pass
                        
                        for f in edge_faces[e_in]:
                            if face_counts[f] == 0:
                                creates_uncovered = True
                                break
                        
                        if creates_uncovered:
                            continue
                        
                        # Prefer swaps that cover more critical faces
                        score = new_critical_covered
                        if score > best_score:
                            best_score = score
                            best_e_in = e_in
                    
                    if best_e_in != -1:
                        removed.discard(e_out)
                        removed.add(best_e_in)
                        for f in edge_faces[e_out]:
                            face_counts[f] -= 1
                        for f in edge_faces[e_in]:
                            face_counts[f] += 1
                        improved = True
            
            if not improved:
                # Stuck - try perturbation
                perturbed = False
                for _ in range(100):
                    if time.time() >= deadline:
                        break
                    if not removed:
                        break
                    
                    # Random perturbation: remove 1-3 random edges and re-add them
                    num_remove = random.randint(1, 3)
                    to_remove = random.sample(list(removed), min(num_remove, len(removed)))
                    
                    saved_removed = set(removed)
                    saved_face_counts = face_counts[:]
                    saved_size = current_size
                    
                    for e in to_remove:
                        removed.discard(e)
                        for f in edge_faces[e]:
                            face_counts[f] -= 1
                    
                    # Try to re-add edges greedily
                    for e in to_remove:
                        # Only add back if it doesn't create uncovered faces
                        valid = True
                        for f in edge_faces[e]:
                            if face_counts[f] == 0:
                                valid = False
                                break
                        if valid:
                            removed.add(e)
                            for f in edge_faces[e]:
                                face_counts[f] += 1
                    
                    # Run deletion pass
                    for e in list(removed):
                        critical = False
                        for f in edge_faces[e]:
                            if face_counts[f] == 1:
                                critical = True
                                break
                        if not critical:
                            removed.discard(e)
                            for f in edge_faces[e]:
                                face_counts[f] -= 1
                            current_size -= 1
                            if current_size < best_size:
                                best_size = current_size
                                best_removed = set(removed)
                                best_face_counts = face_counts[:]
                    
                    perturbed = True
                    break
                
                if not perturbed:
                    break
        
        return best_removed, best_size
>>>>>>> REPLACE

