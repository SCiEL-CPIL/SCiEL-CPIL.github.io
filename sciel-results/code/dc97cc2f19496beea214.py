"""Combined: complementary deletion + face-based local search for C4-free Q7 edges."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    d = 7
    n_vertices = 1 << d  # 128
    n_edges = n_vertices * d // 2  # 448
    
    # Build edge list and mapping
    edge_to_idx = {}
    idx_to_edge = []
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(idx_to_edge)
                idx_to_edge.append((u, v))
                edge_to_idx[(u, v)] = idx
    
    # Precompute faces: each face is a set of 4 edge indices
    faces = []
    face_edges = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for x in range(n_vertices):
                if (x & mask_i) or (x & mask_j):
                    continue
                v0 = x
                v1 = x ^ mask_i
                v2 = x ^ mask_j
                v3 = x ^ mask_i ^ mask_j
                
                e01 = (min(v0, v1), max(v0, v1))
                e02 = (min(v0, v2), max(v0, v2))
                e13 = (min(v1, v3), max(v1, v3))
                e23 = (min(v2, v3), max(v2, v3))
                
                idx01 = edge_to_idx[e01]
                idx02 = edge_to_idx[e02]
                idx13 = edge_to_idx[e13]
                idx23 = edge_to_idx[e23]
                
                face_idx = len(faces)
                faces.append([idx01, idx02, idx13, idx23])
                
                for ei in [idx01, idx02, idx13, idx23]:
                    face_edges[ei].append(face_idx)
    
    num_faces = len(faces)
    
    # Precompute for each edge, the 3 other edges in each face it belongs to
    # edge_face_others[edge_idx] = list of (face_idx, [other_e1, other_e2, other_e3])
    edge_face_others = [[] for _ in range(n_edges)]
    for fidx, face in enumerate(faces):
        for pos, e in enumerate(face):
            others = [face[k] for k in range(4) if k != pos]
            edge_face_others[e].append((fidx, others))
    
    deadline = time.time() + 115
    
    # Precompute face membership matrices for vectorized checks
    # face_adj[edge_idx] = list of face indices containing this edge
    # For each edge, store the "others" as a flat array for vectorized counting
    edge_others_flat = []  # edge_others_flat[edge_idx] = np.array of other edges in each face
    edge_face_ids = []     # edge_face_ids[edge_idx] = np.array of face indices
    
    for e in range(n_edges):
        flist = []
        fidlist = []
        for fidx, others in edge_face_others[e]:
            fidlist.append(fidx)
            flist.extend(others)
        edge_others_flat.append(np.array(flist, dtype=np.int32) if flist else np.array([], dtype=np.int32))
        edge_face_ids.append(np.array(fidlist, dtype=np.int32) if fidlist else np.array([], dtype=np.int32))
    
    def can_add(idx, sel):
        """Check if edge idx can be added without creating a C4 (vectorized)."""
        others = edge_others_flat[idx]
        if len(others) == 0:
            return True
        # Reshape to (num_faces, 3)
        n_f = len(others) // 3
        others_reshaped = others.reshape(n_f, 3)
        sel_others = sel[others_reshaped]
        # If any face has all 3 others selected, can't add
        return not np.any(np.all(sel_others, axis=1))
    
    def can_swap(idx_rm, idx_add, sel):
        """Check if removing idx_rm and adding idx_add keeps C4-free (vectorized)."""
        others = edge_others_flat[idx_add]
        if len(others) == 0:
            return True
        n_f = len(others) // 3
        others_reshaped = others.reshape(n_f, 3)
        sel_others = sel[others_reshaped].astype(np.int8)
        # For each face, count how many of the 3 others are selected,
        # but if idx_rm is among them, it doesn't count (since we're removing it)
        # Create mask: 1 where the edge is idx_rm
        rm_mask = (others_reshaped == idx_rm).astype(np.int8)
        # Effective selected: sel_others but set to 0 where rm_mask is 1
        effective = sel_others * (1 - rm_mask)
        # If any face has all 3 effective selected, can't swap
        return not np.any(np.all(effective == 1, axis=1))
    
    def count_edges(sel):
        return int(np.sum(sel))
    
    def greedy_fill(sel, order=None, max_time=None):
        """Add back edges that can be added without creating C4."""
        if order is None:
            order = list(range(n_edges))
            random.shuffle(order)
        for idx in order:
            if max_time is not None and time.time() > max_time:
                break
            if not sel[idx]:
                if can_add(idx, sel):
                    sel[idx] = True
    
    def local_search_1flip(sel, rng, max_iters=200):
        """Try to add random missing edges."""
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            for _ in range(80):
                idx = rng.randint(0, n_edges - 1)
                if sel[idx]:
                    continue
                if can_add(idx, sel):
                    sel[idx] = True
                    improved = True
                    break
        return sel
    
    def local_search_2swap(sel, rng, max_iters=150):
        """Try 2-swap: remove one edge, add another. Targeted: for a missing edge,
        pick removal candidates from its blocking faces."""
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            for _ in range(60):
                # Pick a random missing edge to add
                idx_add = rng.randint(0, n_edges - 1)
                if sel[idx_add]:
                    continue
                others = edge_others_flat[idx_add]
                if len(others) == 0:
                    sel[idx_add] = True
                    improved = True
                    break
                n_f = len(others) // 3
                others_reshaped = others.reshape(n_f, 3)
                sel_others = sel[others_reshaped]
                # Find blocking faces (all 3 others selected)
                blocking = np.where(np.all(sel_others, axis=1))[0]
                if len(blocking) == 0:
                    sel[idx_add] = True
                    improved = True
                    break
                # Collect candidate removals from blocking faces
                candidates = set()
                for bf in blocking:
                    for e in others_reshaped[bf]:
                        if sel[e]:
                            candidates.add(int(e))
                if not candidates:
                    continue
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                for idx_rm in cand_list[:6]:
                    if can_swap(idx_rm, idx_add, sel):
                        sel[idx_rm] = False
                        sel[idx_add] = True
                        improved = True
                        break
                if improved:
                    break
        return sel
    
    def local_search_3flip(sel, rng, max_iters=100):
        """Try 3-flip: remove two edges, add one. Targeted: pick a missing edge,
        find blocking faces, pick 2 removals from those faces."""
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            # Pick a random missing edge to add
            idx_add = rng.randint(0, n_edges - 1)
            if sel[idx_add]:
                continue
            others = edge_others_flat[idx_add]
            if len(others) == 0:
                # No blocking, just add (shouldn't happen if 1flip already ran)
                sel[idx_add] = True
                improved = True
                break
            n_f = len(others) // 3
            others_reshaped = others.reshape(n_f, 3)
            sel_others = sel[others_reshaped]
            blocking = np.where(np.all(sel_others, axis=1))[0]
            if len(blocking) == 0:
                sel[idx_add] = True
                improved = True
                break
            # Collect candidate removals from blocking faces
            candidates = set()
            for bf in blocking:
                for e in others_reshaped[bf]:
                    if sel[e]:
                        candidates.add(int(e))
            if len(candidates) < 2:
                continue
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            # Try pairs of removals (limit to keep fast)
            tried = 0
            for i in range(min(len(cand_list), 8)):
                for j in range(i + 1, min(len(cand_list), 8)):
                    tried += 1
                    if tried > 12:
                        break
                    idx_rm1 = cand_list[i]
                    idx_rm2 = cand_list[j]
                    sel_others_i8 = sel[others_reshaped].astype(np.int8)
                    rm_mask = ((others_reshaped == idx_rm1) | (others_reshaped == idx_rm2)).astype(np.int8)
                    effective = sel_others_i8 * (1 - rm_mask)
                    if not np.any(np.all(effective == 1, axis=1)):
                        sel[idx_rm1] = False
                        sel[idx_rm2] = False
                        sel[idx_add] = True
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        return sel

    def local_search_4swap(sel, rng, max_iters=50):
        """Try 4-swap: remove 2 edges, add 2 edges. Net 0 but can escape local optima."""
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            # Pick two random missing edges to add
            idx_add1 = rng.randint(0, n_edges - 1)
            idx_add2 = rng.randint(0, n_edges - 1)
            if sel[idx_add1] or sel[idx_add2] or idx_add1 == idx_add2:
                continue
            # Get blocking candidates for both
            others1 = edge_others_flat[idx_add1]
            others2 = edge_others_flat[idx_add2]
            if len(others1) == 0 or len(others2) == 0:
                continue
            n_f1 = len(others1) // 3
            n_f2 = len(others2) // 3
            or1 = others1.reshape(n_f1, 3)
            or2 = others2.reshape(n_f2, 3)
            sel1 = sel[or1]
            sel2 = sel[or2]
            blk1 = np.where(np.all(sel1, axis=1))[0]
            blk2 = np.where(np.all(sel2, axis=1))[0]
            if len(blk1) == 0 or len(blk2) == 0:
                # One is not blocked, just add it (should be caught by 1flip)
                if len(blk1) == 0:
                    sel[idx_add1] = True
                    improved = True
                    break
                if len(blk2) == 0:
                    sel[idx_add2] = True
                    improved = True
                    break
                continue
            # Collect removal candidates
            cands1 = set()
            for bf in blk1:
                for e in or1[bf]:
                    if sel[e]:
                        cands1.add(int(e))
            cands2 = set()
            for bf in blk2:
                for e in or2[bf]:
                    if sel[e]:
                        cands2.add(int(e))
            if not cands1 or not cands2:
                continue
            cl1 = list(cands1)[:5]
            cl2 = list(cands2)[:5]
            rng.shuffle(cl1)
            rng.shuffle(cl2)
            for idx_rm1 in cl1:
                for idx_rm2 in cl2:
                    if idx_rm1 == idx_rm2:
                        continue
                    # Check if removing idx_rm1 and idx_rm2 unblocks both
                    sel1_i8 = sel[or1].astype(np.int8)
                    sel2_i8 = sel[or2].astype(np.int8)
                    rm1_mask = ((or1 == idx_rm1) | (or1 == idx_rm2)).astype(np.int8)
                    rm2_mask = ((or2 == idx_rm1) | (or2 == idx_rm2)).astype(np.int8)
                    eff1 = sel1_i8 * (1 - rm1_mask)
                    eff2 = sel2_i8 * (1 - rm2_mask)
                    if not np.any(np.all(eff1 == 1, axis=1)) and not np.any(np.all(eff2 == 1, axis=1)):
                        sel[idx_rm1] = False
                        sel[idx_rm2] = False
                        sel[idx_add1] = True
                        sel[idx_add2] = True
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        return sel
    
    def complementary_deletion(rng):
        """Start with all edges, remove to break all C4s."""
        sel = np.ones(n_edges, dtype=bool)
        
        # Build list of currently violating faces
        face_order = list(range(num_faces))
        rng.shuffle(face_order)
        
        for fidx in face_order:
            if time.time() > deadline:
                break
            face = faces[fidx]
            # Check if all 4 edges are still selected
            if sel[face[0]] and sel[face[1]] and sel[face[2]] and sel[face[3]]:
                # Remove the edge that appears in the most other violating faces
                best_to_remove = None
                best_score = -1
                for e in face:
                    if not sel[e]:
                        continue
                    # Count how many other faces this edge is in that are still fully selected
                    conflict_count = 0
                    for ofidx, others in edge_face_others[e]:
                        if ofidx == fidx:
                            continue
                        if all(sel[oe] for oe in others):
                            conflict_count += 1
                    if conflict_count > best_score:
                        best_score = conflict_count
                        best_to_remove = e
                if best_to_remove is not None:
                    sel[best_to_remove] = False
        
        return sel
    
    def full_local_search(sel, rng):
        """Iterative 1-flip, 2-swap, 3-flip, 4-swap until no improvement."""
        prev_count = -1
        for _ in range(25):
            if time.time() > deadline - 2:
                break
            cnt = count_edges(sel)
            if cnt == prev_count:
                break
            prev_count = cnt
            
            sel = local_search_1flip(sel, rng, max_iters=100)
            if time.time() > deadline - 2:
                break
            sel = local_search_2swap(sel, rng, max_iters=100)
            if time.time() > deadline - 2:
                break
            # Try 3-flip to escape local optima
            if rng.random() < 0.4:
                sel = local_search_3flip(sel, rng, max_iters=60)
                if time.time() > deadline - 2:
                    break
            # Try 4-swap less frequently
            if rng.random() < 0.25:
                sel = local_search_4swap(sel, rng, max_iters=40)
                if time.time() > deadline - 2:
                    break
            # After escape moves, re-run 1flip and 2swap to harvest gains
            sel = local_search_1flip(sel, rng, max_iters=80)
            if time.time() > deadline - 2:
                break
            sel = local_search_2swap(sel, rng, max_iters=60)
        
        return sel
    
    def polish_pass(sel, rng):
        """Final polish: multiple passes of targeted 1-flip and 2-swap."""
        for _pass in range(3):
            if time.time() > deadline - 1:
                break
            # Try adding every missing edge in order
            for idx in range(n_edges):
                if time.time() > deadline - 1:
                    break
                if not sel[idx] and can_add(idx, sel):
                    sel[idx] = True
            
            # Targeted 2-swap: for each missing edge, find if any single removal enables it
            missing = [i for i in range(n_edges) if not sel[i]]
            rng.shuffle(missing)
            for idx_add in missing[:150]:
                if time.time() > deadline - 1:
                    break
                if sel[idx_add]:
                    continue
                others = edge_others_flat[idx_add]
                if len(others) == 0:
                    sel[idx_add] = True
                    continue
                n_f = len(others) // 3
                others_reshaped = others.reshape(n_f, 3)
                sel_others = sel[others_reshaped]
                blocking = np.where(np.all(sel_others, axis=1))[0]
                if len(blocking) == 0:
                    sel[idx_add] = True
                    continue
                for bf in blocking:
                    if time.time() > deadline - 1:
                        break
                    face_others = others_reshaped[bf]
                    for e in face_others:
                        if sel[e] and can_swap(e, idx_add, sel):
                            sel[e] = False
                            sel[idx_add] = True
                            break
                    if sel[idx_add]:
                        break
            # Final 1-flip pass
            for idx in range(n_edges):
                if time.time() > deadline - 1:
                    break
                if not sel[idx] and can_add(idx, sel):
                    sel[idx] = True
        
        return sel
    
    best_sel = None
    best_count = 0
    rng = random.Random(42)
    
    # Multiple restarts
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        rng_seed = 42 + restart * 137
        rng = random.Random(rng_seed)
        
        # Phase 1: Complementary deletion
        sel = complementary_deletion(rng)
        cnt = count_edges(sel)
        
        # Phase 2: Greedy fill
        if time.time() < deadline - 8:
            greedy_fill(sel, max_time=deadline - 8)
            cnt = count_edges(sel)
        
        # Phase 3: Full local search
        if time.time() < deadline - 5:
            sel = full_local_search(sel, rng)
            cnt = count_edges(sel)
        
        if cnt > best_count:
            best_count = cnt
            best_sel = sel.copy()
        
        # If we're doing well, do additional local search on best
        if best_count >= 300 and time.time() < deadline - 10:
            sel2 = best_sel.copy()
            sel2 = full_local_search(sel2, rng)
            cnt2 = count_edges(sel2)
            if cnt2 > best_count:
                best_count = cnt2
                best_sel = sel2
            # Second round of refinement on best
            if best_count >= 304 and time.time() < deadline - 8:
                sel3 = best_sel.copy()
                sel3 = full_local_search(sel3, rng)
                cnt3 = count_edges(sel3)
                if cnt3 > best_count:
                    best_count = cnt3
                    best_sel = sel3
    
    # Convert to edge list
    if best_sel is None:
        return []
    
    # Final polish pass on the best solution
    if time.time() < deadline - 2:
        rng_polish = random.Random(12345)
        best_sel = polish_pass(best_sel, rng_polish)
    
    edges = []
    for i in range(n_edges):
        if best_sel[i]:
            u, v = idx_to_edge[i]
            edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
