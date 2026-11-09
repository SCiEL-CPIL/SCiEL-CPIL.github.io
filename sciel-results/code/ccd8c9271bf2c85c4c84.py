import time
import random
import numpy as np

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290 # Safety margin
    
    # Precompute coordinates
    coords = np.array([(x, y) for y in range(N) for x in range(N)])
    
    # Precompute squared distances
    # dist2[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
    # This is 4096 x 4096 = 16M floats. ~128MB. Acceptable.
    # We can optimize by computing on the fly if memory is tight, but 128MB is fine.
    # Let's use int32 to save memory. Max dist is 2*63^2 = 7938.
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Calculate pairwise squared distances
    # dx = x_i - x_j
    # dist2 = dx^2 + dy^2
    # Using broadcasting
    # This might be slow to compute initially but worth it.
    # 4096x4096 matrix.
    # Let's try to compute it efficiently.
    
    # Actually, we can just compute dist2 on the fly using numpy vectorization for a single point against a set.
    # That avoids the large matrix.
    
    best_set = []
    best_len = 0
    
    # Helper to check if point q can be added to S
    # S: list of indices
    # dists: list of sets, dists[i] has dists from S[i] to others
    def can_add(q, S, dists):
        if q in S: # S should be a set for O(1) lookup, but we need list for order?
            return False, None
            
        # We need to check:
        # 1. Distances from q to all p in S are unique.
        # 2. For each p in S, dist(q, p) is not in dists[index_of_p].
        
        # To make this fast, we need to map index in S to index in dists.
        # Let's maintain S as a set of indices, and a parallel list for order?
        # Or just iterate S.
        
        seen_dists_from_q = set()
        
        # Get coordinates of q
        qx, qy = coords[q]
        
        for p in S:
            px, py = coords[p]
            d2 = (qx - px)**2 + (qy - py)**2
            
            # Check uniqueness from q
            if d2 in seen_dists_from_q:
                return False, None
            seen_dists_from_q.add(d2)
            
            # Check if d2 is already in dists for p
            # We need to know which dists entry corresponds to p.
            # If S is a list, we can zip(S, dists).
            
        # This function is called very often. We need it to be fast.
        # Using Python sets and loops is slow.
        # Let's precompute dist2 matrix?
        
        return True, seen_dists_from_q

    # Let's use the precomputed matrix for speed.
    # dist2[i, j]
    
    # Compute dist2 matrix
    # This takes a few seconds.
    if time.time() > deadline - 250:
         return []
         
    # Efficient computation of dist2
    # x: (4096, 1), y: (4096, 1)
    # dist2 = (x - x.T)**2 + (y - y.T)**2
    # This creates a 4096x4096 array.
    
    x = coords[:, 0].reshape(-1, 1)
    y = coords[:, 1].reshape(-1, 1)
    
    # This might blow up memory if not careful, but 4096^2 is 16M.
    # 16M * 4 bytes (int32) = 64MB. OK.
    
    # However, computing (x - x.T) creates a 4096x4096 float array.
    # Let's just do it.
    
    try:
        dist2_mat = ((x - x.T)**2 + (y - y.T)**2).astype(np.int32)
    except MemoryError:
        # Fallback to on-the-fly if memory fails (unlikely)
        dist2_mat = None

    # Local Search
    # State: S (set of indices), dists (list of sets corresponding to S)
    # We need to keep S and dists synchronized.
    
    def add_point(q, S, dists):
        # q: index
        # S: set of indices
        # dists: list of sets. dists[i] is for S[i].
        # This is hard to keep in sync if S is a set.
        # Let's use S as a list, and a set for O(1) membership check.
        
        pass

    # Better representation:
    # S_list: list of indices
    # S_set: set of indices
    # dists: list of sets, same length as S_list. dists[i] corresponds to S_list[i].
    
    # To check if q can be added:
    # 1. q not in S_set
    # 2. For each i, p in enumerate(S_list):
    #    d = dist2_mat[q, p]
    #    if d in dists[i]: return False
    #    if d in seen_dists: return False
    #    seen_dists.add(d)
    # 3. If valid:
    #    S_list.append(q)
    #    S_set.add(q)
    #    dists.append(seen_dists)
    #    For each i, p in enumerate(old_S_list):
    #       dists[i].add(dist2_mat[q, p])
        
    # This update is O(|S|).
    
    # Let's run a loop of random restarts with local search.
    
    for restart in range(1000):
        if time.time() > deadline - 10:
            break
            
        random.seed(restart)
        
        S_list = []
        S_set = set()
        dists = []
        
        # Greedy Initialization
        # Pick random order
        order = list(range(TOTAL_POINTS))
        random.shuffle(order)
        
        for q in order:
            if time.time() > deadline - 5:
                break
            
            # Check if q can be added
            if q in S_set:
                continue
                
            seen_dists = set()
            valid = True
            
            # We need to iterate S_list
            for i, p in enumerate(S_list):
                d = dist2_mat[q, p]
                
                if d in dists[i]:
                    valid = False
                    break
                if d in seen_dists:
                    valid = False
                    break
                seen_dists.add(d)
            
            if valid:
                S_list.append(q)
                S_set.add(q)
                dists.append(seen_dists)
                
                # Update dists for existing points
                for i, p in enumerate(S_list[:-1]):
                    dists[i].add(dist2_mat[q, p])
        
        current_len = len(S_list)
        
        if current_len > best_len:
            best_len = current_len
            best_set = S_list.copy()
            # We should probably deep copy dists too if we want to resume, 
            # but we are restarting from scratch or doing local search from here.
            
        # Local Search / Iterated Local Search
        # Try to add more points
        # If stuck, remove some and try again
        
        local_improved = True
        while local_improved and time.time() < deadline - 5:
            local_improved = False
            
            # Try random adds
            # Pick random point not in S
            # If valid, add it.
            
            # To avoid scanning all points, we can keep a list of candidates?
            # Or just pick random indices.
            
            for _ in range(100): # Max tries per step
                if time.time() > deadline - 5:
                    break
                    
                q = random.randint(0, TOTAL_POINTS - 1)
                if q in S_set:
                    continue
                
                seen_dists = set()
                valid = True
                
                for i, p in enumerate(S_list):
                    d = dist2_mat[q, p]
                    if d in dists[i]:
                        valid = False
                        break
                    if d in seen_dists:
                        valid = False
                        break
                    seen_dists.add(d)
                
                if valid:
                    S_list.append(q)
                    S_set.add(q)
                    dists.append(seen_dists)
                    for i, p in enumerate(S_list[:-1]):
                        dists[i].add(dist2_mat[q, p])
                    local_improved = True
                    break # Restart the "try random adds" loop? Or continue?
                    # Breaking allows us to check time and update best.
                    # Let's update best immediately.
                    if len(S_list) > best_len:
                        best_len = len(S_list)
                        best_set = S_list.copy()
            
            if not local_improved:
                # Stuck. Perturb.
                # Remove 1-3 random points
                if len(S_list) > 0:
                    num_remove = min(3, len(S_list))
                    for _ in range(num_remove):
                        if not S_list: break
                        idx = random.randint(0, len(S_list) - 1)
                        p = S_list[idx]
                        S_set.remove(p)
                        
                        # Remove p from S_list
                        S_list.pop(idx)
                        
                        # Remove dists entry for p
                        dists.pop(idx)
                        
                        # Remove p from dists of other points
                        # dists[i] contains dist to p. We need to remove dist2_mat[other, p]
                        for i in range(len(S_list)):
                            other = S_list[i]
                            d = dist2_mat[other, p]
                            if d in dists[i]:
                                dists[i].discard(d)
                
                # After perturbation, continue to try adds
                # The while loop will continue if we set local_improved = True?
                # No, we want to try adds again.
                # So we don't break the while loop, we just continue to the next iteration 
                # which will try to add.
                # But the `for _ in range(100)` loop above finished without adding.
                # So we go to the next iteration of the `while` loop.
                # It will try to add again.
                # This is fine.

    return [coords[i] for i in best_set]
