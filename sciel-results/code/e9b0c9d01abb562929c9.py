import time
import random

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 115  # Safety margin

    # Precompute edges
    all_edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_idx[(x, y)] = idx
    m = len(all_edges)  # 448

    # Precompute faces
    face_edges = []
    edge_faces = [[] for _ in range(m)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, d2), max(b, d2))]
                    e3 = edge_idx[(min(d2, c), max(d2, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]
                    
                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)
                        
    n_faces = len(face_edges)  # 672

    best_removed = None
    best_size = m  # Minimize removed edges

    def local_search(R, time_limit):
        """
        Local search: try to add back removed edges (increase selected count)
        by checking if they don't create a C4. Use swaps to escape local optima.
        R is a set of removed edge indices. We want to minimize |R|.
        """
        start_time = time.time()
        R = set(R)
        
        # Build face coverage: for each face, how many of its 4 edges are in R (removed)
        # A face is "satisfied" (no C4 in selected set) if at least 1 of its 4 edges is removed
        # i.e., not all 4 edges are selected.
        
        # For each edge e in R, check if we can add it back (remove from R)
        # without violating any face constraint.
        # Adding e back means: for every face f containing e, we need at least one other edge in f to be in R.
        
        # Precompute: for each face, the set of edges in R that cover it
        # face_covered_by_R[f] = list of edges in R that are in face f
        
        def is_safe_to_add(e, R_set):
            """Check if adding edge e back (removing from R) keeps all faces valid."""
            for f in edge_faces[e]:
                # Face f has edges face_edges[f]
                # After removing e from R, the remaining removed edges in f are:
                # {e2 for e2 in face_edges[f] if e2 in R_set and e2 != e}
                # We need at least one such edge to remain.
                has_other_removed = False
                for e2 in face_edges[f]:
                    if e2 != e and e2 in R_set:
                        has_other_removed = True
                        break
                if not has_other_removed:
                    return False
            return True
        
        # Phase 1: Greedy add-back
        # Try to add back edges that are safe, prioritizing those that "free up" more faces
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Find all edges in R that are safe to add back
            safe_edges = []
            for e in R:
                if is_safe_to_add(e, R):
                    # Score: number of faces where this edge is the ONLY removed edge
                    # (i.e., faces that would become invalid if we add it back... wait no)
                    # Actually, adding e back is safe only if no face loses its last removed edge.
                    # So all safe edges are equally "safe". Let's just pick one.
                    safe_edges.append(e)
            
            if safe_edges:
                # Pick one to add back
                e = random.choice(safe_edges)
                R.remove(e)
                changed = True
        
        # Phase 2: Swap-based local search
        # Try swapping: remove e1 from R (add back), add e2 to R (remove from selected)
        # This is valid if after the swap, all faces are still hit.
        # Net effect: |R| unchanged, but we might enable further add-backs.
        # Actually, for minimizing |R|, swaps don't help directly unless they enable add-backs.
        
        # Phase 3: Iterative improvement with random restarts within local search
        # Try removing random subsets and re-optimizing
        for _ in range(20):
            if time.time() > time_limit:
                break
            
            # Try to add back a random safe edge
            safe_edges = [e for e in R if is_safe_to_add(e, R)]
            if safe_edges:
                e = random.choice(safe_edges)
                R.remove(e)
            
            # Now try to add back more
            changed = True
            while changed and time.time() < time_limit:
                changed = False
                safe_edges = [e for e in R if is_safe_to_add(e, R)]
                if safe_edges:
                    e = random.choice(safe_edges)
                    R.remove(e)
                    changed = True
        
        return R
    
    # Run Greedy Hitting Set with random tie-breaking multiple times
    # to potentially escape bad local minima of the greedy order
    iterations = 50
    for it in range(iterations):
        if time.time() > deadline - 30:
            break
            
        # Initialize
        R = set()
        uncovered_faces = set(range(n_faces))
        
        # Greedy
        while uncovered_faces:
            counts = [0] * m
            for f in uncovered_faces:
                for e in face_edges[f]:
                    counts[e] += 1
            
            max_count = max(counts)
            if max_count == 0:
                break
                
            candidates = [e for e in range(m) if counts[e] == max_count]
            best_e = random.choice(candidates)
            
            R.add(best_e)
            
            # Update uncovered
            to_remove = []
            for f in edge_faces[best_e]:
                if f in uncovered_faces:
                    to_remove.append(f)
            for f in to_remove:
                uncovered_faces.remove(f)
        
        # Redundancy Removal
        changed = True
        while changed:
            changed = False
            cover_count = [0] * n_faces
            for e in R:
                for f in edge_faces[e]:
                    cover_count[f] += 1
            
            for e in list(R):
                redundant = True
                for f in edge_faces[e]:
                    if cover_count[f] == 1:
                        redundant = False
                        break
                
                if redundant:
                    R.remove(e)
                    changed = True
                    break
        
        # Local search to further minimize |R|
        time_left = deadline - 35 - time.time()
        if time_left > 1:
            R = local_search(R, time.time() + min(time_left, 5))
        
        # Check size
        if len(R) < best_size:
            best_size = len(R)
            best_removed = R.copy()
            # Early stop if we hit a good bound
            if best_size <= 140:
                break

    # If no valid solution found (should not happen), fallback to simple linear construction
    if best_removed is None:
        best_removed = set(range(m))
        best_size = m

    # Construct final answer
    selected_edges = []
    for e in range(m):
        if e not in best_removed:
            x, y = all_edges[e]
            selected_edges.append([x, y])
            
    return selected_edges

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Edges: {len(edges)}")
    
    d = 7
    n = 1 << d
    all_edges_set = set()
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges_set.add((x, y))
                
    selected_set = set()
    for u, v in edges:
        selected_set.add((min(u, v), max(u, v)))
        
    valid = True
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(a, b), max(a, b))
                    e2 = (min(b, d2), max(b, d2))
                    e3 = (min(d2, c), max(d2, c))
                    e4 = (min(c, a), max(c, a))
                    
                    cnt = 0
                    if e1 in selected_set: cnt += 1
                    if e2 in selected_set: cnt += 1
                    if e3 in selected_set: cnt += 1
                    if e4 in selected_set: cnt += 1
                    
                    if cnt == 4:
                        print(f"Invalid face found at {x}, dims {i},{j}")
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        print("Valid!")
    else:
        print("Invalid!")
