import time
import random
import math

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
                
    # 2. Build Face List
    face_edges = [] # list of lists of 4 edge IDs
    edge_faces = [[] for _ in range(n_edges)]
    
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
                
                face_id = len(face_edges)
                face_edges.append([e1, e2, e3, e4])
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
                    
    n_faces = len(face_edges)
    
    start_time = time.time()
    timeout = 115 # Safety margin
    
    def greedy_init():
        removed_mask = 0
        face_hit = [0] * n_faces
        unhit = n_faces
        
        # Randomize the order of edges to get different greedy solutions
        edge_order = list(range(n_edges))
        random.shuffle(edge_order)
        
        # Greedy: Pick edge that covers max unhit faces.
        # To speed up, we can just iterate through shuffled edges and add if it covers any unhit face?
        # No, standard greedy set cover picks the one with max gain.
        # With 448 edges and 672 faces, scanning all edges to find max gain is 448 * (avg faces per edge) ~ 448 * 3 = 1300 ops per step.
        # Steps <= 448. Total ~ 600k ops. Very fast.
        
        while unhit > 0:
            best_gain = -1
            best_e = -1
            for e in edge_order:
                if (removed_mask >> e) & 1:
                    continue
                gain = 0
                for f in edge_faces[e]:
                    if face_hit[f] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_e = e
            if best_e == -1:
                break
            removed_mask |= (1 << best_e)
            for f in edge_faces[best_e]:
                if face_hit[f] == 0:
                    unhit -= 1
                face_hit[f] += 1
        return removed_mask

    # Run multiple greedy initializations and keep the best
    best_removed_mask = 0
    best_removed_count = n_edges # Worst case
    
    # Do a few random greedy runs
    for _ in range(20):
        if time.time() - start_time > timeout:
            break
        mask = greedy_init()
        cnt = bin(mask).count('1')
        if cnt < best_removed_count:
            best_removed_count = cnt
            best_removed_mask = mask
            
    # Local Search on the best mask
    current_mask = best_removed_mask
    current_count = bin(current_mask).count('1')
    
    # Precompute face_hit for current mask
    face_hit = [0] * n_faces
    for f, es in enumerate(face_edges):
        for e in es:
            if (current_mask >> e) & 1:
                face_hit[f] += 1
                
    # Helper to check if a move is valid
    # Move: remove set `rem`, add set `add`
    # We need to check if any face becomes unhit.
    # A face becomes unhit if:
    # 1. It had exactly 1 hit, and that hit was in `rem`, and it has no edge in `add`.
    
    def try_move(rem_edges, add_edges):
        """
        rem_edges: list of edge indices to remove from mask
        add_edges: list of edge indices to add to mask
        Returns True if the move is valid (all faces still hit) and improves count.
        """
        if len(rem_edges) <= len(add_edges):
            return False # No improvement in count
            
        # Check validity
        # For each face, count how many of its edges are in rem_edges and add_edges.
        # We only need to check faces incident to rem_edges.
        # Let's use a temporary array or dict for face hits change.
        # Since rem_edges is small (1 or 2), we can just check directly.
        
        # Create a set of faces to check
        faces_to_check = set()
        for e in rem_edges:
            faces_to_check.update(edge_faces[e])
            
        for f in faces_to_check:
            # Count hits in this face for current mask
            # We know face_hit[f] is the current count.
            # We remove rem_edges and add add_edges.
            # New count = face_hit[f] - (count of rem_edges in face) + (count of add_edges in face)
            # We need New count >= 1.
            
            rem_count = 0
            add_count = 0
            for e in face_edges[f]:
                if e in rem_edges:
                    rem_count += 1
                if e in add_edges:
                    add_count += 1
                    
            new_count = face_hit[f] - rem_count + add_count
            if new_count < 1:
                return False
        return True

    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        
        # 1. Try single removals
        for e in range(n_edges):
            if not ((current_mask >> e) & 1):
                continue
            if try_move([e], []):
                # Perform move
                current_mask ^= (1 << e)
                for f in edge_faces[e]:
                    face_hit[f] -= 1
                current_count -= 1
                improved = True
                # Don't break, try to find more? Or break to restart?
                # Restarting is safer to avoid inconsistent state if we don't update carefully.
                # But we update face_hit. So we can continue.
                # However, removing one might enable removing another.
                # Let's break and restart the loop to be safe and simple.
                break
                
        if improved:
            continue
            
        # 2. Try 1-for-1 swaps
        # Optimize: Only try swaps if single removals failed.
        # To make it faster, we can precompute for each removed edge e1, which non-removed edges e2 could swap with.
        # But let's just iterate.
        for e1 in range(n_edges):
            if not ((current_mask >> e1) & 1):
                continue
            # Check if e1 is "critical" (face_hit == 1 for some face)
            # If not critical, it could have been removed in step 1.
            # So if we are here, e1 is likely critical or step 1 missed it (unlikely if step 1 was thorough).
            # Actually, step 1 removes ANY non-critical edge.
            # So if we are here, all removed edges are critical (part of a face with hit=1).
            
            for e2 in range(n_edges):
                if (current_mask >> e2) & 1:
                    continue
                if try_move([e1], [e2]):
                    # Perform move
                    current_mask ^= (1 << e1)
                    current_mask ^= (1 << e2)
                    # Update face_hit
                    for f in edge_faces[e1]:
                        face_hit[f] -= 1
                    for f in edge_faces[e2]:
                        face_hit[f] += 1
                    # Count unchanged
                    improved = True
                    break
            if improved:
                break
                
        if improved:
            continue
            
        # 3. Try 2-for-1 swaps
        # Remove e1, e2; Add e3.
        # This is O(N^3) if naive. 448^3 ~ 90M. Too slow in Python.
        # Optimization:
        # Only consider e1, e2 that are in the same face? Or adjacent?
        # If e1 and e2 are in different faces, removing them might unhit both faces.
        # e3 must hit both faces.
        # So e3 must be in intersection of faces(e1) and faces(e2)?
        # No, e3 just needs to be in any face that was unhit by e1/e2 removal.
        
        # Let's restrict the search space.
        # Pick a random removed edge e1.
        # Pick a random removed edge e2 != e1.
        # Find all e3 that could potentially fix the unhit faces.
        
        # To avoid O(N^3), we can do:
        # For each pair (e1, e2) of removed edges that share a face?
        # Number of pairs is ~448*2/2 = 448 (if each edge in ~3 faces, pairs per face is 6, total pairs ~ 672*6/2 ~ 2000).
        # This is small!
        
        # Let's iterate over faces.
        # For each face F, let R_F be removed edges in F.
        # If |R_F| >= 2, we can try to remove two of them.
        
        # Actually, simpler:
        # Iterate over all pairs of removed edges (e1, e2).
        # There are ~200 removed edges. Pairs ~ 20000.
        # For each pair, check if valid.
        # try_move([e1, e2], [e3])
        # We need to find e3.
        
        # Let's collect candidate e3s.
        # Faces unhit by removing e1, e2:
        # U = { f | face_hit[f] == 1 and (e1 in F or e2 in F) and (e1 not in F or e2 not in F) }?
        # No.
        # A face f is unhit if:
        # 1. face_hit[f] == 1
        # 2. The single hit is e1 or e2.
        # 3. e3 is NOT in f (so it doesn't compensate).
        
        # So, for a pair (e1, e2), the set of "dangerous" faces is:
        # D = { f | face_hit[f] == 1 and (e1 in F_f or e2 in F_f) }
        # Wait, if face_hit[f] == 2 and e1, e2 are the hits, then removing them makes it 0.
        # So D = { f | face_hit[f] <= 2 and (e1 in F_f or e2 in F_f) and (face_hit[f] - count(e1,e2 in F_f) == 0) }
        
        # If D is empty, the move is valid for any e3 (even empty add, but we need add 1 for 2-for-1? No, 2-for-0 is better if valid).
        # But step 1 would have caught 2-for-0 if possible?
        # Step 1 removes ONE edge. If removing e1 is valid, it does it.
        # If removing e1 is NOT valid, but removing e1 AND e2 IS valid (and e3 is not needed), that means e1 and e2 together covered a face, but individually they were critical?
        # No. If removing e1 is invalid, there is a face f with hit=1 and e1 in f.
        # If we remove e1 and e2, f is still unhit unless e2 is in f.
        # So if e2 is in f, f might still be hit?
        # If f has hit=1 (only e1), and e2 is not in f, then removing e1, e2 leaves f unhit.
        # If f has hit=1 (only e1), and e2 IS in f... wait, hit=1 means only 1 removed edge. So e2 cannot be in f if e1 is the only one.
        # So if face_hit[f] == 1, e2 is NOT in f.
        # Therefore, removing e1 makes f unhit. Adding e2 doesn't help f.
        # So 2-for-0 is only valid if no face has hit=1 involving e1 or e2.
        # But step 1 would have removed e1 if it wasn't critical.
        # So if we are at step 3, e1 is critical (has a face with hit=1).
        # So 2-for-0 is never valid?
        # Unless... the "critical" face for e1 is also critical for e2? No, hit=1 means unique.
        
        # So we MUST add an edge e3 to compensate for the faces that lost their only hit.
        
        # Let's implement 2-for-1.
        
        # To speed up, we can limit the number of pairs we check.
        # Or just check all pairs of removed edges.
        
        removed_list = [i for i in range(n_edges) if (current_mask >> i) & 1]
        for i1 in range(len(removed_list)):
            if time.time() - start_time > timeout: break
            e1 = removed_list[i1]
            for i2 in range(i1 + 1, len(removed_list)):
                e2 = removed_list[i2]
                
                # Identify faces that will become unhit
                # Faces f where face_hit[f] == 1 and e1 in f?
                # Or face_hit[f] == 2 and e1, e2 in f?
                
                # Let's just use try_move with a dummy add? No, we need to find e3.
                
                # Find candidate e3s:
                # e3 must be in ALL faces that become unhit.
                # Unhit faces U = { f | (face_hit[f] == 1 and e1 in F_f) or (face_hit[f] == 2 and e1 in F_f and e2 in F_f) }
                # Actually, if face_hit[f] == 1 and e1 in F_f, e2 is not in F_f. So removing e1, e2 leaves it 0.
                # If face_hit[f] == 2 and e1, e2 in F_f, removing them leaves it 0.
                # If face_hit[f] == 1 and e2 in F_f, removing e1, e2 leaves it 0.
                
                U = set()
                for f in edge_faces[e1]:
                    if face_hit[f] == 1:
                        U.add(f)
                    elif face_hit[f] == 2 and e2 in face_edges[f]:
                        U.add(f)
                for f in edge_faces[e2]:
                    if face_hit[f] == 1:
                        U.add(f)
                    # The "2 and e1 in f" case is covered by e1's loop if e2 in f?
                    # If e1 in f and e2 in f and hit=2, it's added in e1's loop.
                    # If e2 in f and hit=1, it's added here.
                
                if not U:
                    # Valid 2-for-0?
                    # This implies removing e1, e2 doesn't unhit any face.
                    # But step 1 would have removed e1 if it was safe.
                    # So this case shouldn't happen if step 1 was perfect.
                    # But let's handle it.
                    if try_move([e1, e2], []):
                         # Perform 2-for-0
                         current_mask ^= (1 << e1)
                         current_mask ^= (1 << e2)
                         for f in edge_faces[e1]: face_hit[f] -= 1
                         for f in edge_faces[e2]: face_hit[f] -= 1
                         current_count -= 2
                         improved = True
                         break
                         
                # Find e3 in intersection of edges of all faces in U
                # And e3 not in current_mask
                candidates = set(face_edges[f] for f in U) # This is union, we need intersection!
                # Intersection of sets of edges in faces U.
                # Start with edges of first face in U.
                if U:
                    first_face = next(iter(U))
                    cand_set = set(face_edges[first_face])
                    for f in U:
                        if f == first_face: continue
                        cand_set &= set(face_edges[f])
                else:
                    cand_set = set()
                    
                # Filter candidates: must not be in current_mask
                for e3 in cand_set:
                    if (current_mask >> e3) & 1:
                        continue
                    # Check if e3 is actually in all faces in U?
                    # The intersection logic above ensures e3 is in all faces in U.
                    # So adding e3 will hit all faces in U.
                    # Are there other faces that become unhit?
                    # No, U contains ALL faces that become unhit.
                    # So the move is valid.
                    
                    # Perform 2-for-1
                    current_mask ^= (1 << e1)
                    current_mask ^= (1 << e2)
                    current_mask ^= (1 << e3)
                    
                    for f in edge_faces[e1]: face_hit[f] -= 1
                    for f in edge_faces[e2]: face_hit[f] -= 1
                    for f in edge_faces[e3]: face_hit[f] += 1
                    
                    current_count -= 1
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break

    # Construct result
    result = []
    for i, (u, v) in enumerate(edges):
        if not ((current_mask >> i) & 1):
            result.append([u, v])
            
    return result
