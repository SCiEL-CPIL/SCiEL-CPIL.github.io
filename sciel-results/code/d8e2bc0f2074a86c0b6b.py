import time
import random

N = 9
NUM_VERTICES = 1 << N  # 512

def construct():
    """
    Returns a list of distinct integers in [0, 2**N) representing an induced path
    in Q_N. Consecutive vertices differ in exactly one bit; non-consecutive
    vertices differ in at least two bits.
    """
    start_time = time.time()
    TIME_LIMIT = 115  # Leave 5s buffer for safety
    
    # Precompute neighbors for each vertex as sets for fast operations
    neighbors = [set() for _ in range(NUM_VERTICES)]
    for v in range(NUM_VERTICES):
        for b in range(N):
            neighbors[v].add(v ^ (1 << b))
    
    best_path = None
    best_len = 0
    
    def backtrack(path, used, blocked):
        """
        path: list of vertices in the path
        used: set of vertices in the path
        blocked: set of vertices that are adjacent to any vertex in path[:-1]
                 (i.e., cannot be the next vertex)
        """
        nonlocal best_path, best_len, start_time
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        cur_len = len(path)
        if cur_len > best_len:
            best_len = cur_len
            best_path = path[:]
        
        cur = path[-1]
        
        # Candidates are neighbors of cur that are not used and not blocked
        # blocked contains neighbors of path[:-1].
        # cur is not in blocked (unless path len > 2 and cur is adj to path[-2]? 
        # No, cur is path[-1]. blocked is adj to path[:-1]. 
        # If cur is adj to path[-2], cur is in blocked. But cur is in used.
        # So we just check not in used and not in blocked.
        
        candidates = []
        for nb in neighbors[cur]:
            if nb not in used and nb not in blocked:
                candidates.append(nb)
        
        if not candidates:
            return
        
        # Heuristic: MRV (Minimum Remaining Values)
        # For each candidate c, calculate how many valid next moves would exist after adding c.
        # If we add c:
        # new_used = used | {c}
        # new_blocked = blocked | neighbors[cur]  (because cur becomes part of path[:-1] for the next step)
        # valid_nexts = neighbors[c] - new_used - new_blocked
        
        def score_candidate(c):
            new_blocked = blocked | neighbors[cur]
            # We don't need to update used for the count of *next* moves strictly, 
            # but c is used, so c is not a candidate for itself.
            # The candidates for the step after c are neighbors of c.
            # They must not be in new_used (which is used + {c}) and not in new_blocked.
            # Note: c is in neighbors[cur], so c is in new_blocked? 
            # No, new_blocked is neighbors of path[:-1] plus neighbors of cur.
            # Wait, the definition of blocked:
            # "blocked: set of vertices that are adjacent to any vertex in path[:-1]"
            # When we add c to path, the new path is path + [c].
            # The new "prefix" is path + [cur] (which is path[:-1] of the new path? No.)
            # Let's trace:
            # Path: [v0, v1, ..., vk]
            # blocked = N(v0) U ... U N(v{k-1})
            # Candidates for v{k+1} must be in N(vk) \ used \ blocked.
            # If we pick c = v{k+1}:
            # New Path: [v0, ..., vk, c]
            # New blocked = N(v0) U ... U N(v{k-1}) U N(vk) = blocked U N(vk).
            # Next candidates must be in N(c) \ (used U {c}) \ new_blocked.
            
            count = 0
            for nb2 in neighbors[c]:
                if nb2 in used: continue
                if nb2 == c: continue # c is in used now, so this is redundant if we check used|{c}
                if nb2 in new_blocked: continue
                count += 1
            return count
        
        # Sort by score: try most constrained (lowest score) first
        candidates.sort(key=score_candidate)
        
        for c in candidates:
            # Forward check: if score is 0, we can skip (though sort puts them first, 
            # we can explicitly check to avoid recursion overhead if 0)
            # But score_candidate already computed it. 
            # To avoid double computation, we could store scores, but list is small.
            # Let's just recurse. If it fails immediately, it's fast.
            
            new_used = used | {c}
            new_blocked = blocked | neighbors[cur]
            
            path.append(c)
            backtrack(path, new_used, new_blocked)
            path.pop()
            
            if time.time() - start_time > TIME_LIMIT:
                return
    
    # Try multiple starting configurations
    # By symmetry, start=0 is good.
    # Initial path: [0]
    # used: {0}
    # blocked: empty (no vertices in path[:-1])
    
    initial_path = [0]
    initial_used = {0}
    initial_blocked = set()
    
    backtrack(initial_path, initial_used, initial_blocked)
    
    # If we have time, try a few more restarts? 
    # With strong pruning, one deep search might be better than many shallow ones.
    # But if the first branch fails quickly, we might have time for others.
    # However, the symmetry of the hypercube means starting with 0 and any neighbor is equivalent 
    # up to isomorphism? 
    # Actually, the first step determines the "direction". 
    # Due to symmetry of Q_N, all edges are equivalent. 
    # So starting with [0, 1] covers the same structural class as [0, 2].
    # Therefore, one search from [0] exploring all neighbors is sufficient to cover the space 
    # reachable from start 0.
    # However, if the search gets stuck in a local optimum due to heuristic ordering, 
    # different random orderings or restarts might help.
    # But MRV is deterministic. 
    # Let's stick to one deep search first. If it finishes fast, we can try randomizing the tie-breaking.
    
    if best_path:
        return best_path
    else:
        # Fallback to greedy if search fails or finds nothing better
        path, used = [0], {0}
        blocked = set()
        while True:
            cur = path[-1]
            nxt = None
            # Candidates from neighbors[cur] not in used and not in blocked
            for nb in neighbors[cur]:
                if nb not in used and nb not in blocked:
                    nxt = nb
                    break
            if nxt is None:
                return path
            path.append(nxt)
            used.add(nxt)
            blocked = blocked | neighbors[cur]
