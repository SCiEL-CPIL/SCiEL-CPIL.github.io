import time
import random
import numpy as np

def construct():
    d = 7
    n = 2 ** d
    deadline = time.time() + 110.0
    
    # 1. Generate edges
    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
                edge_idx[(y, x)] = idx
    m = len(edges) # 448

    # 2. Generate faces
    # A face is defined by two distinct bit positions b, c and a base vertex a
    # such that bits b and c are 0 in a.
    faces = []
    edge_faces = [set() for _ in range(m)]
    
    for b in range(d):
        for c in range(b + 1, d):
            for a in range(n):
                if (a & (1 << b)) or (a & (1 << c)):
                    continue
                v0 = a
                v1 = a ^ (1 << b)
                v2 = a ^ (1 << c)
                v3 = a ^ (1 << b) ^ (1 << c)
                
                e0 = edge_idx[(min(v0, v1), max(v0, v1))]
                e1 = edge_idx[(min(v1, v3), max(v1, v3))]
                e2 = edge_idx[(min(v2, v3), max(v2, v3))]
                e3 = edge_idx[(min(v0, v2), max(v0, v2))]
                
                face_edges = frozenset([e0, e1, e2, e3])
                faces.append(face_edges)
                f_idx = len(faces) - 1
                for e in face_edges:
                    edge_faces[e].add(f_idx)
    
    F = len(faces) # 672

    def solve_once(seed):
        rng = random.Random(seed)
        
        # Greedy Hitting Set
        removed = set()
        unhit = set(range(F))
        
        while unhit:
            # Find edge with max coverage in unhit
            best_e = -1
            max_cov = -1
            # Optimization: only check edges that are in at least one unhit face
            # But checking all 448 is fast enough
            for e in range(m):
                cov = 0
                for f in edge_faces[e]:
                    if f in unhit:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_e = e
            
            if best_e == -1:
                break # Should not happen if unhit is not empty
                
            removed.add(best_e)
            for f in edge_faces[best_e]:
                if f in unhit:
                    unhit.remove(f)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try deletions
            for e in list(removed):
                redundant = True
                for f in edge_faces[e]:
                    # Check if there is another edge in removed \cap face[f]
                    # face[f] is a set of 4 edges
                    count = 0
                    for ee in faces[f]:
                        if ee in removed:
                            count += 1
                    if count == 1:
                        redundant = False
                        break
                if redundant:
                    removed.discard(e)
                    improved = True
            
            if time.time() >= deadline - 5.0:
                break

            # 2. Try swaps
            # For each e in removed, see if we can swap with e' not in removed
            # such that hitting property is maintained.
            # Condition: For all f in edge_faces[e] with |removed \cap face[f]| == 1, e' must be in face[f].
            
            # To make it efficient, we can pick random e from removed and random e' from candidates
            # Or iterate.
            
            # Let's try random swaps to escape local minima
            if removed:
                e = rng.choice(list(removed))
                # Determine critical faces for e
                critical_faces = []
                for f in edge_faces[e]:
                    count = 0
                    for ee in faces[f]:
                        if ee in removed:
                            count += 1
                    if count == 1:
                        critical_faces.append(f)
                
                if not critical_faces:
                    # Can just delete, which is handled in step 1
                    continue
                
                # Find intersection of critical faces
                # Intersection of sets of edges
                inter = set(faces[critical_faces[0]])
                for f in critical_faces[1:]:
                    inter.intersection_update(faces[f])
                
                # Candidates are in inter but not in removed
                candidates = inter - removed
                if candidates:
                    e_new = rng.choice(list(candidates))
                    # Perform swap
                    removed.discard(e)
                    removed.add(e_new)
                    improved = True
                    # After swap, we might have new deletions possible
                    # The loop will restart and check deletions again
        
        return len(removed)

    best_removed_size = m # Max possible removed is m (keep 0)
    
    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        r = solve_once(i)
        if r < best_removed_size:
            best_removed_size = r
            
    # The result is the number of edges KEPT
    kept = m - best_removed_size
    
    # We need to return the list of edges, not just the count.
    # The problem asks for `list of edges [u, v]`.
    # My `solve_once` only returns the size. I need to return the actual set of kept edges.
    
    # Let's modify solve_once to return the set of removed edges, or kept edges.
    
    def solve_once_full(seed):
        rng = random.Random(seed)
        
        # Greedy Hitting Set
        removed = set()
        unhit = set(range(F))
        
        while unhit:
            best_e = -1
            max_cov = -1
            for e in range(m):
                cov = 0
                for f in edge_faces[e]:
                    if f in unhit:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_e = e
            
            if best_e == -1:
                break
                
            removed.add(best_e)
            for f in edge_faces[best_e]:
                if f in unhit:
                    unhit.remove(f)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try deletions
            for e in list(removed):
                redundant = True
                for f in edge_faces[e]:
                    count = 0
                    for ee in faces[f]:
                        if ee in removed:
                            count += 1
                    if count == 1:
                        redundant = False
                        break
                if redundant:
                    removed.discard(e)
                    improved = True
            
            if time.time() >= deadline - 5.0:
                break

            # 2. Try swaps
            if removed:
                e = rng.choice(list(removed))
                critical_faces = []
                for f in edge_faces[e]:
                    count = 0
                    for ee in faces[f]:
                        if ee in removed:
                            count += 1
                    if count == 1:
                        critical_faces.append(f)
                
                if not critical_faces:
                    continue
                
                inter = set(faces[critical_faces[0]])
                for f in critical_faces[1:]:
                    inter.intersection_update(faces[f])
                
                candidates = inter - removed
                if candidates:
                    e_new = rng.choice(list(candidates))
                    removed.discard(e)
                    removed.add(e_new)
                    improved = True
        
        # Return the set of KEPT edges
        kept_edges = [edges[i] for i in range(m) if i not in removed]
        return kept_edges

    best_kept = []
    best_size = 0
    
    for i in range(100):
        if time.time() > deadline - 10.0:
            break
        kept = solve_once_full(i)
        if len(kept) > best_size:
            best_size = len(kept)
            best_kept = kept
            
    return best_kept
