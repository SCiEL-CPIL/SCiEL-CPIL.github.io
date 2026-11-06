"""
Snake-in-the-box in Q_9 using constraint-based backtracking with heuristic ordering.
Goal: find an induced path (snake) longer than the greedy seed (104 edges).
"""
import numpy as np

N = 9
NUM_VERTICES = 1 << N  # 512

def construct():
    """
    Returns a list of distinct integers in [0, 2**N) representing an induced path
    in Q_N. Consecutive vertices differ in exactly one bit; non-consecutive
    vertices differ in at least two bits.
    """
    # Precompute neighbors for each vertex
    # neighbors[v] = list of vertices at Hamming distance 1 from v
    neighbors = [[] for _ in range(NUM_VERTICES)]
    for v in range(NUM_VERTICES):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # We want to build a path. The path must be induced, meaning no chords.
    # A chord exists if two non-consecutive vertices in the path are adjacent.
    
    # Strategy: Backtracking search.
    # State: current path (list of vertices), set of used vertices.
    # Constraint: When adding a new vertex v to the end of path P, 
    #   1. v must not be in P.
    #   2. v must be adjacent to P[-1] (always true by construction).
    #   3. v must NOT be adjacent to any vertex in P[:-1] (to maintain induced property).
    
    # Heuristic for ordering candidates:
    # We want to minimize future branching factor.
    # A good proxy is to prefer vertices that have fewer "safe" neighbors available 
    # for the next step, or simply try to extend as long as possible.
    # However, standard greedy fails. 
    # Let's try a depth-first search with a limit on the depth or time, 
    # trying to find a path longer than 104.
    
    # To make this efficient within 120s, we need strong pruning.
    # One key insight: If we are at vertex v, and the only unused neighbors of v 
    # are all "blocked" (adjacent to some previous vertex in the path), we are stuck.
    
    # Let's use a recursive backtracking with a global time check.
    import time
    start_time = time.time()
    TIME_LIMIT = 110  # Leave 10s buffer
    
    best_path = None
    best_len = 0
    
    # Initial path: start at 0
    # To break symmetry, we can fix the first step. 
    # By symmetry of the hypercube, all vertices are equivalent, and all first steps are equivalent.
    # So we can fix start=0, and first step = 1 (flip bit 0).
    
    def is_safe(v, path, used_set):
        """Check if v can be added to the end of path."""
        if v in used_set:
            return False
        # Check if v is adjacent to any vertex in path[:-1]
        # Note: v is adjacent to path[-1] by construction, which is allowed.
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                return False
        return True
    
    def backtrack(path, used_set, depth):
        nonlocal best_path, best_len, start_time
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        cur = path[-1]
        
        # Get candidates: neighbors of cur that are not used
        candidates = []
        for nb in neighbors[cur]:
            if nb not in used_set:
                if is_safe(nb, path, used_set):
                    candidates.append(nb)
        
        if not candidates:
            return
        
        # Heuristic: Order candidates by "constraint"
        # A vertex that leaves fewer options for the next step is more constrained.
        # Let's estimate the number of valid next steps for each candidate.
        def score_candidate(c):
            # Count how many valid extensions exist from c
            count = 0
            for nb2 in neighbors[c]:
                if nb2 not in used_set and nb2 != path[-1]:
                    # Check if nb2 is safe relative to path + [c]
                    # nb2 must not be adjacent to any in path
                    # And nb2 must not be adjacent to c? No, nb2 is neighbor of c.
                    # The constraint is: nb2 not in used, and nb2 not adjacent to path[:-1] (and path[-1] is c's parent, wait)
                    # Path is P. We add c. New path P' = P + [c].
                    # Next step w must be neighbor of c, not in P, and not adjacent to P[:-1].
                    # Note: w is adjacent to c. c is adjacent to P[-1].
                    # So w must not be adjacent to any p in P[:-1].
                    # Also w != P[-1] is already covered by "not in used" if P[-1] is in used.
                    # Wait, P[-1] is in used. So "not in used" covers it.
                    
                    # Optimization: Pre-compute? No, too expensive.
                    # Just check bit_count.
                    safe = True
                    for p in path:
                        if (w ^ p).bit_count() == 1:
                            safe = False
                            break
                    if safe:
                        count += 1
            return count
        
        # This scoring is O(N * len(path)) per candidate.
        # With len(path) up to ~100, and N=9, it's ~900 ops per candidate.
        # Candidates are at most 9. So ~8100 ops per node.
        # If we visit 10,000 nodes, that's 81M ops. Might be too slow in Python.
        
        # Let's try a simpler heuristic first: just random or fixed order?
        # Random might help escape local minima but is slow to converge.
        # Fixed order (lowest bit first) is what the greedy does, and it fails.
        
        # Let's try ordering by "degree of freedom" in the remaining graph?
        # Or just shuffle candidates to avoid deterministic traps?
        # Given the time limit, a pure DFS with good pruning is key.
        
        # Let's try sorting by the number of unused neighbors (simple heuristic)
        # Fewer unused neighbors -> more constrained -> try first? Or last?
        # Usually, "fail first" is good: try the most constrained variable first.
        
        # Let's calculate a simple constraint score: number of unused neighbors.
        def simple_score(c):
            return sum(1 for nb in neighbors[c] if nb not in used_set)
            
        candidates.sort(key=simple_score)
        
        for c in candidates:
            path.append(c)
            used_set.add(c)
            backtrack(path, used_set, depth + 1)
            path.pop()
            used_set.remove(c)
            
            if time.time() - start_time > TIME_LIMIT:
                return
    
    # Start the search
    # Fix start to 0, first step to 1 to break symmetry
    initial_path = [0, 1]
    initial_used = {0, 1}
    
    # Verify initial step is safe (it is, distance 1)
    
    backtrack(initial_path, initial_used, 2)
    
    if best_path:
        return best_path
    else:
        # Fallback to greedy if search fails or finds nothing better
        path, used = [0], {0}
        while True:
            cur, nxt = path[-1], None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                return path
            path.append(nxt)
            used.add(nxt)
