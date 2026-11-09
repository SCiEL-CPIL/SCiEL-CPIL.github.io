import time
import random
import numpy as np
from collections import defaultdict

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 295

    # Precompute points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute distance matrix D[i,j] = squared distance
    # M=4096, D is 4096x4096 int32 = 67MB, fine
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2).astype(np.int32)

    # Precompute for each point p, the list of (distance, other_point) pairs
    # This allows O(1) lookup: given p and a distance d, find all q in S with D[p,q]==d
    # Actually we need: for point p, for each distance value d, which points q have D[p,q]=d
    # Build: dist_map[p] = dict {d: list of q indices}
    # But M=4096, each row has 4096 entries. 4096 dicts with ~4096 entries each.
    # Alternative: for each p, sort points by distance and use binary search.
    # Better: precompute for each p an array of (distances sorted, corresponding point indices)
    # Then for a given d, we can find all q with D[p,q]==d via searchsorted.
    
    # Actually, the key insight for O(|S|) check:
    # To check if p can be added to S:
    #   1. All D[p, s] for s in S must be distinct.
    #   2. For each s in S, D[p, s] must not equal D[s, t] for any t in S, t != s.
    #      Equivalently: for each s in S, D[p, s] is not in {D[s, t] : t in S, t != s}.
    #      But D[s, s] = 0, and D[p, s] > 0 (since p not in S), so we can include t=s.
    #      So: D[p, s] not in {D[s, t] : t in S}.
    
    # For condition 2, we need to check if D[p, s] appears in the row D[s, S].
    # If we maintain for each s in S the set of distances {D[s, t] : t in S}, we can check in O(1).
    # But maintaining that set incrementally is complex.
    
    # Alternative structural approach: Use a different representation.
    # Instead of checking each s individually, note that the condition is:
    #   For all pairs (s, t) in S x S, D[p, s] != D[s, t] unless... wait no.
    #   The isosceles condition: triangle (p, s, t) is isosceles if:
    #     D[p,s] == D[p,t] (p is apex) -> caught by condition 1
    #     D[p,s] == D[s,t] (s is apex) -> condition 2 for pair (s,t)
    #     D[p,t] == D[s,t] (t is apex) -> condition 2 for pair (t,s), i.e., D[p,t] == D[t,s]
    #   So we need: for all s in S, D[p,s] not in {D[s,t] : t in S}
    #   AND: for all t in S, D[p,t] not in {D[t,s] : s in S}  -- but this is the same set!
    #   So condition 2 is: for all s in S, D[p,s] not in {D[s,t] : t in S}.
    
    # Precompute: for each point s, and for each possible distance d, the count of points t with D[s,t]=d.
    # dist_count[s][d] = number of t such that D[s,t] = d
    # Then when checking if p can be added: for each s in S, check if dist_count[s][D[p,s]] >= 1
    #   BUT this counts ALL t in the grid, not just t in S.
    #   We need: number of t IN S with D[s,t] = D[p,s].
    
    # So we need a dynamic structure: for each s in S, maintain a dict dist_in_S[s][d] = count of t in S with D[s,t]=d.
    # When adding a new point p to S:
    #   For each s in S: dist_in_S[s][D[s,p]] += 1
    #   Also: dist_in_S[p][D[p,s]] = 1 for each s in S (p's distances to existing S)
    #   And we need to check: for each s in S, D[p,s] is not already in dist_in_S[s] with count >= 1
    #   AND: all D[p,s] are distinct.
    
    # When removing a point p from S:
    #   For each s in S: dist_in_S[s][D[s,p]] -= 1
    #   Remove dist_in_S[p]
    #   For each s in S: if dist_in_S[s][D[s,p]] becomes 0, we could delete the key (optional)
    
    # This gives O(|S|) per add/remove check and O(|S|) per can_add.
    
    # Let's implement this.
    
    # dist_in_S: dict mapping point index -> dict {distance: count}
    # Only for points currently in S.
    
    # For fast access, we'll use a list of dicts, indexed by point index.
    # dist_in_S[p] = dict, but only valid if p is in S.
    
    dist_in_S = [None] * M  # dist_in_S[p] is a dict if p in S, else None
    
    # Also maintain S as a list for iteration.
    
    def can_add(p, S_list):
        """Check if point p can be added to set S_list (list of point indices)."""
        if not S_list:
            return True
        
        # Condition 1: All distances from p to points in S must be distinct
        seen_dists = set()
        for s in S_list:
            d = D[p, s]
            if d in seen_dists:
                return False
            seen_dists.add(d)
        
        # Condition 2: For each s in S, D[p,s] must not be a distance from s to any point in S
        for s in S_list:
            d = D[p, s]
            # Check if d is in dist_in_S[s]
            if dist_in_S[s] is not None and d in dist_in_S[s]:
                return False
        
        return True

    def add_point(p, S_list):
        """Add point p to S_list and update dist_in_S."""
        S_list.append(p)
        dist_in_S[p] = {}
        for s in S_list[:-1]:  # all existing points
            d_sp = D[s, p]
            d_ps = D[p, s]  # same as d_sp since symmetric
            # Update dist_in_S[s]
            if d_sp not in dist_in_S[s]:
                dist_in_S[s][d_sp] = 0
            dist_in_S[s][d_sp] += 1
            # Set dist_in_S[p]
            dist_in_S[p][d_ps] = 1  # only one point s at this distance? No, could be multiple.
        
        # Wait, dist_in_S[p] should count distances from p to all other points in S.
        # Since we just added p, for each s in S (excluding p), dist_in_S[p][D[p,s]] should be incremented.
        # But we already set it to 1 above, which is wrong if multiple s have the same distance.
        # Let me redo this.
        
        # Actually, let's rebuild dist_in_S[p] properly.
        dist_in_S[p] = {}
        for s in S_list:
            if s == p:
                continue
            d = D[p, s]
            if d not in dist_in_S[p]:
                dist_in_S[p][d] = 0
            dist_in_S[p][d] += 1
    
    def remove_point(p, S_list):
        """Remove point p from S_list and update dist_in_S."""
        # First, decrement dist_in_S for all other points
        for s in S_list:
            if s == p:
                continue
            d = D[s, p]
            if dist_in_S[s] is not None:
                if d in dist_in_S[s]:
                    dist_in_S[s][d] -= 1
                    if dist_in_S[s][d] == 0:
                        del dist_in_S[s][d]
        
        # Remove p from S_list
        S_list.remove(p)
        # Clear dist_in_S[p]
        dist_in_S[p] = None
    
    # Let me rewrite add_point and remove_point more carefully.
    
    def add_point_correct(p, S_list):
        """Add point p to S_list and update dist_in_S. Assumes can_add(p, S_list) is True."""
        # For each existing s in S_list, update dist_in_S[s]
        for s in S_list:
            d = D[s, p]
            if dist_in_S[s] is None:
                dist_in_S[s] = {}
            if d not in dist_in_S[s]:
                dist_in_S[s][d] = 0
            dist_in_S[s][d] += 1
        
        # Now add p to S_list and initialize dist_in_S[p]
        S_list.append(p)
        dist_in_S[p] = {}
        for s in S_list:
            if s == p:
                continue
            d = D[p, s]
            if d not in dist_in_S[p]:
                dist_in_S[p][d] = 0
            dist_in_S[p][d] += 1
    
    def remove_point_correct(p, S_list):
        """Remove point p from S_list and update dist_in_S."""
        # For each other s in S_list, decrement dist_in_S[s] for distance D[s,p]
        for s in S_list:
            if s == p:
                continue
            d = D[s, p]
            if dist_in_S[s] is not None and d in dist_in_S[s]:
                dist_in_S[s][d] -= 1
                if dist_in_S[s][d] == 0:
                    del dist_in_S[s][d]
        
        # Remove p from S_list
        S_list.remove(p)
        dist_in_S[p] = None
    
    # Use the corrected versions
    add_point = add_point_correct
    remove_point = remove_point_correct

    best_S = []
    best_size = 0
    
    # Greedy construction with random order
    S = []
    candidates = list(range(M))
    random.shuffle(candidates)
    for p in candidates:
        if time.time() > deadline - 5:
            break
        if can_add(p, S):
            add_point(p, S)
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
    
    # Local search: try to swap points or add/remove
    while time.time() < deadline - 5:
        # Strategy: take current best, remove a random point, try to add a different one
        if best_size == 0:
            break
        
        # Copy best_S and its dist_in_S state
        # We need to reconstruct dist_in_S for best_S
        # This is expensive. Instead, let's do local search on the current S.
        
        # Alternative: do a series of random restarts with the fast can_add.
        # Since can_add is now O(|S|), we can do many more restarts.
        
        # Let's do: random restart with greedy
        S = []
        # Reinitialize dist_in_S
        for i in range(M):
            dist_in_S[i] = None
        
        candidates = list(range(M))
        random.shuffle(candidates)
        for p in candidates:
            if time.time() > deadline - 5:
                break
            if can_add(p, S):
                add_point(p, S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
        
        # Also try: perturb best_S
        if best_size > 0 and random.random() < 0.5:
            S2 = best_S.copy()
            # Rebuild dist_in_S for S2
            for i in range(M):
                dist_in_S[i] = None
            for p in S2:
                dist_in_S[p] = {}
            for i, p in enumerate(S2):
                for j, q in enumerate(S2):
                    if i == j:
                        continue
                    d = D[p, q]
                    if d not in dist_in_S[p]:
                        dist_in_S[p][d] = 0
                    dist_in_S[p][d] += 1
            
            # Remove 1-3 random points
            n_remove = random.randint(1, min(3, len(S2)))
            to_remove = random.sample(S2, n_remove)
            for p in to_remove:
                remove_point(p, S2)
            
            # Try to add points
            candidates = [p for p in range(M) if p not in S2]
            random.shuffle(candidates)
            for p in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add(p, S2):
                    add_point(p, S2)
            
            if len(S2) > best_size:
                best_size = len(S2)
                best_S = S2.copy()

    return [tuple(points[i]) for i in best_S]
