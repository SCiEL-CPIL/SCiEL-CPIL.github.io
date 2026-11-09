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
    
    def can_add(idx, sel):
        """Check if edge idx can be added without creating a C4."""
        for fidx, others in edge_face_others[idx]:
            if sel[others[0]] and sel[others[1]] and sel[others[2]]:
                return False
        return True
    
    def can_swap(idx_rm, idx_add, sel):
        """Check if removing idx_rm and adding idx_add keeps C4-free."""
        for fidx, others in edge_face_others[idx_add]:
            cnt = 0
            for e in others:
                if e == idx_rm:
                    continue
                if sel[e]:
                    cnt += 1
            if cnt >= 3:
                return False
        return True
    
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
        """Try 2-swap: remove one edge, add another."""
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            for _ in range(40):
                # Pick a random existing edge to potentially remove
                idx_rm = rng.randint(0, n_edges - 1)
                if not sel[idx_rm]:
                    continue
                # Pick a random missing edge to add
                idx_add = rng.randint(0, n_edges - 1)
                if sel[idx_add]:
                    continue
                if can_swap(idx_rm, idx_add, sel):
                    sel[idx_rm] = False
                    sel[idx_add] = True
                    improved = True
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
        """Iterative 1-flip and 2-swap until no improvement."""
        prev_count = -1
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            cnt = count_edges(sel)
            if cnt == prev_count:
                break
            prev_count = cnt
            
            sel = local_search_1flip(sel, rng, max_iters=100)
            if time.time() > deadline - 2:
                break
            sel = local_search_2swap(sel, rng, max_iters=80)
        
        return sel
    
    best_sel = None
    best_count = 0
    rng = random.Random(42)
    
    # Multiple restarts
    restart = 0
    while time.time() < deadline - 3:
        restart += 1
        rng_seed = 42 + restart * 137
        rng = random.Random(rng_seed)
        
        # Phase 1: Complementary deletion
        sel = complementary_deletion(rng)
        cnt = count_edges(sel)
        
        # Phase 2: Greedy fill
        if time.time() < deadline - 5:
            greedy_fill(sel, max_time=deadline - 5)
            cnt = count_edges(sel)
        
        # Phase 3: Full local search
        if time.time() < deadline - 3:
            sel = full_local_search(sel, rng)
            cnt = count_edges(sel)
        
        if cnt > best_count:
            best_count = cnt
            best_sel = sel.copy()
        
        # If we're doing well, do additional local search on best
        if best_count >= 300 and time.time() < deadline - 5:
            sel2 = best_sel.copy()
            sel2 = full_local_search(sel2, rng)
            cnt2 = count_edges(sel2)
            if cnt2 > best_count:
                best_count = cnt2
                best_sel = sel2
    
    # Convert to edge list
    if best_sel is None:
        return []
    
    edges = []
    for i in range(n_edges):
        if best_sel[i]:
            u, v = idx_to_edge[i]
            edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
