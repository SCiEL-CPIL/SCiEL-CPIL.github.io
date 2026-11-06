"""Snake-in-the-box in Q_9: Constraint Propagation Backtracking Solver."""
import time
import random
import sys

N = 9
MASK = (1 << N) - 1
SIZE = 1 << N

# Precompute neighbors
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def _solve_backtrack(start, deadline, rng):
    """
    Depth-first search with backtracking and Warnsdorff's rule.
    Returns the longest path found within the deadline.
    """
    best_path = [start]
    best_len = 1
    
    # State:
    # path: list of vertices
    # used: set of vertices in path
    # blocked: set of vertices that are adjacent to any vertex in path (cannot be added)
    # Note: 'blocked' helps quickly check validity for new additions.
    # A vertex v is valid to add if v not in used and v not in blocked.
    # Wait, blocked is defined as neighbors of used.
    # If v is in blocked, it is adjacent to some u in path.
    # If v is added, it must NOT be adjacent to any u in path[:-1].
    # So we need to track which vertices are "forbidden" for the next step.
    
    # Let's use a simpler but effective heuristic:
    # 1. Maintain `used` set.
    # 2. Maintain `forbidden` set: vertices that are adjacent to any vertex in `used`.
    #    Actually, if we add v, v must not be adjacent to any u in `used` except the last one.
    #    So `forbidden` should be neighbors of `used` minus the last vertex's neighbors? No.
    #    Let's just check on the fly but use Warnsdorff.
    
    path = [start]
    used = {start}
    
    # To speed up "is v valid", we can precompute for each v, the set of neighbors.
    # Valid(v) means:
    # 1. v not in used
    # 2. For all u in path[:-1], (v^u).bit_count() != 1
    
    # We will use recursion or iterative deepening? 
    # Given the depth (~100+), recursion might be fine with sys.setrecursionlimit.
    # But iterative with explicit stack is safer for time limits.
    
    # Let's try a recursive DFS with a hard time limit check.
    # To avoid deep recursion overhead, we can also just use a greedy approach with backtracking.
    
    # Heuristic: Warnsdorff's rule.
    # At each step, choose the neighbor v such that the number of valid neighbors of v is minimized.
    # This forces the path into tight corners, potentially leaving more open space for the rest.
    
    def get_valid_neighbors(v, path, used):
        """
        Get list of (neighbor, onward_count) for v.
        A neighbor w is valid if:
        1. w not in used
        2. w is not adjacent to any u in path[:-1]
        """
        valids = []
        last_u = path[-1]
        # The vertices in path[:-1] are the "forbidden" set for adjacency.
        # Let's compute the set of vertices adjacent to path[:-1] once? 
        # No, it changes.
        
        # Optimization: 
        # If we maintain a set `blocked` which is the union of neighbors of all vertices in path[:-1],
        # then a candidate w is valid if w not in used and w not in blocked.
        
        # Let's maintain `blocked` explicitly.
        # Initially blocked = neighbors of path[0]
        # When we add v, we add neighbors of v to blocked.
        # When we backtrack, we remove neighbors of v from blocked.
        # But we need to be careful: if a neighbor is shared, we need counts or sets.
        # Using a set is fine for N=9 (512 vertices).
        
        pass # This logic will be in the main loop

    # We will implement the iterative version with explicit stack for backtracking.
    
    # Stack items: (path_copy, used_copy, blocked_copy)
    # This is expensive to copy. 
    # Alternative: Recursive with global state and undo operations.
    
    # Let's use recursion with global state variables for speed.
    
    sys.setrecursionlimit(10000)
    
    def dfs(path, used, blocked):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        cur = path[-1]
        
        # Find candidates
        candidates = []
        for nb in NEIGHBORS[cur]:
            if nb in used:
                continue
            if nb in blocked:
                continue
            # Check if nb is valid. 
            # Since blocked contains neighbors of path[:-1], if nb not in blocked, 
            # it is not adjacent to any vertex in path[:-1].
            # So this is sufficient!
            
            # Calculate Warnsdorff score: number of valid neighbors of nb
            # A neighbor x of nb is valid if x not in used and x not in (blocked U {cur} U neighbors of nb? No)
            # If we add nb to path, the new blocked set will be blocked U neighbors(nb).
            # The new used set will be used U {nb}.
            # So valid neighbors of nb in the next step are neighbors of nb that are not in used U {nb} and not in blocked U neighbors(nb)?
            # Actually, we just need to count how many choices we have from nb.
            # Let next_blocked = blocked | set(NEIGHBORS[nb])
            # Let next_used = used | {nb}
            # Count x in NEIGHBORS[nb] such that x not in next_used and x not in next_blocked.
            
            score = 0
            # Optimization: precompute neighbors of nb
            for x in NEIGHBORS[nb]:
                if x in used:
                    continue
                if x == nb:
                    continue
                if x in blocked:
                    continue
                # x is also in neighbors(nb), so it will be in next_blocked.
                # Wait, if x is in neighbors(nb), then x is in next_blocked.
                # So x is INVALID for the step after adding nb?
                # Yes! If we add nb, then any neighbor of nb becomes blocked.
                # So the "onward moves" from nb are neighbors of nb that are NOT in used 
                # AND NOT in the current blocked set 
                # AND NOT in neighbors(nb) (which is trivially true for distinct x? No, x IS in neighbors(nb)).
                
                # Let's re-evaluate.
                # If we add nb, the new path is path + [nb].
                # The new blocked set is blocked_old | neighbors(nb).
                # The new used set is used_old | {nb}.
                # We want to choose nb such that the number of valid next steps is small.
                # Valid next step y must be:
                # 1. y in NEIGHBORS[nb]
                # 2. y not in new_used (so y not in used_old and y != nb)
                # 3. y not in new_blocked (so y not in blocked_old and y not in neighbors(nb))
                
                # Condition 3: y not in neighbors(nb). But y IS in neighbors(nb). 
                # So NO neighbor of nb is a valid next step?
                # That implies we can never extend the path?
                # My definition of `blocked` must be wrong.
                
                # Let's redefine `blocked`.
                # `blocked` should be the set of vertices that CANNOT be added to the end of the path 
                # because they are adjacent to some vertex in the path (excluding the last one).
                # If v is in `blocked`, it means there exists u in path[:-1] such that dist(v,u)=1.
                # If we add v to the end, we require dist(v, u) >= 2 for all u in path[:-1].
                # So yes, `blocked` = union of neighbors of path[:-1].
                
                # If we add nb, the new `blocked` will be `blocked` U neighbors(nb).
                # The next vertex y must be in NEIGHBORS[nb].
                # It must NOT be in `new_blocked`.
                # But y is in NEIGHBORS[nb], so y IS in `new_blocked`.
                # Therefore, y is always invalid?
                
                # This logic is flawed.
                # If y is in NEIGHBORS[nb], then y is adjacent to nb.
                # The condition for y to be valid is that y is NOT adjacent to any u in (path + [nb])[:-1] = path.
                # So y must not be adjacent to any u in path.
                # y IS adjacent to nb. nb is in path.
                # So y is adjacent to a vertex in path (namely nb).
                # So y is INVALID?
                
                # Wait. The condition is: non-consecutive vertices differ in at least 2 bits.
                # Consecutive vertices: v_i, v_{i+1}. dist=1.
                # Non-consecutive: v_i, v_j with |i-j| > 1. dist >= 2.
                
                # If we have path P = [v1, ..., vk].
                # We want to add v_{k+1}.
                # v_{k+1} must be adjacent to vk.
                # v_{k+1} must NOT be adjacent to v1, ..., v_{k-1}.
                
                # So `blocked` should be the set of vertices adjacent to v1, ..., v_{k-1}.
                # v_{k+1} must NOT be in `blocked`.
                # v_{k+1} must NOT be in `used`.
                
                # If we add nb (as v_{k+1}), then for the NEXT step v_{k+2}:
                # v_{k+2} must be adjacent to nb.
                # v_{k+2} must NOT be adjacent to v1, ..., vk.
                # So the new `blocked` for the next step should be `blocked` U neighbors(nb).
                # And v_{k+2} must NOT be in `new_blocked`.
                
                # But v_{k+2} IS adjacent to nb.
                # So v_{k+2} IS in neighbors(nb).
                # So v_{k+2} IS in `new_blocked`.
                # So v_{k+2} is always invalid?
                
                # This implies that NO snake of length > 2 is possible?
                # Clearly wrong.
                
                # Where is the error?
                # "v_{k+2} must NOT be adjacent to v1, ..., vk."
                # nb is v_{k+1}.
                # The set of previous vertices is v1, ..., vk.
                # nb is NOT in this set.
                # So v_{k+2} does NOT need to be non-adjacent to nb?
                # It DOES need to be adjacent to nb (consecutive).
                # It needs to be NON-adjacent to v1, ..., vk.
                
                # So the new `blocked` set (vertices forbidden for the next addition) 
                # should be the set of vertices adjacent to v1, ..., vk.
                # This is exactly the OLD `blocked` set!
                # We do NOT add neighbors(nb) to the blocked set for the NEXT step's validity check 
                # in the sense of "forbidden by previous vertices".
                # The "forbidden" set only grows by adding the neighbors of the vertex we JUST LEAVE BEHIND?
                # No, the "forbidden" set is defined by ALL previous vertices.
                
                # Let's trace:
                # Path: [v1, v2]
                # Used: {v1, v2}
                # Forbidden for v3: Neighbors of v1. (v2 is last, so its neighbors are allowed, provided they are not used).
                # Actually, v3 must be neighbor of v2.
                # v3 must NOT be neighbor of v1.
                # So Forbidden = Neighbors(v1).
                
                # Add v3. Path: [v1, v2, v3]
                # Forbidden for v4: Neighbors(v1) U Neighbors(v2).
                # v4 must be neighbor of v3.
                # v4 must NOT be in Forbidden.
                # v4 must NOT be in Used.
                
                # So, `blocked` accumulates neighbors of all vertices EXCEPT the last one.
                # When we add a new vertex v_new, the old last vertex v_last becomes a "previous" vertex.
                # So we ADD Neighbors(v_last) to `blocked`.
                
                # So, Warnsdorff score for candidate nb:
                # If we add nb, the new last vertex is nb.
                # The new `blocked` will be `blocked` U Neighbors(cur) (where cur is current last).
                # Wait, `blocked` currently contains Neighbors(path[:-1]).
                # If we add nb, the new path is path + [nb].
                # The new `blocked` for the NEXT step (v_{k+2}) will be Neighbors(path) = Neighbors(path[:-1]) U Neighbors(cur).
                # So `new_blocked` = `blocked` | set(NEIGHBORS[cur]).
                
                # We want to count valid next steps from nb.
                # A next step y is valid if:
                # 1. y in NEIGHBORS[nb]
                # 2. y not in `used` (which is `used` | {nb})
                # 3. y not in `new_blocked`
                
                # So for candidate nb:
                # new_blocked = blocked | set(NEIGHBORS[cur])
                # new_used = used | {nb}
                # Count y in NEIGHBORS[nb] such that y not in new_used and y not in new_blocked.
                
                # This is O(N) per candidate. N=9. Candidates <= 9. Total O(81) per step. Very fast.
                
                # Let's implement this.
                
                pass # Logic corrected in actual code below

    # Re-implementing DFS with correct logic
    
    def dfs(path, used, blocked):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        cur = path[-1]
        
        # Prepare new_blocked for the next step IF we were to add a candidate.
        # The "forbidden" set for the step AFTER adding a candidate nb 
        # is the set of vertices adjacent to all vertices in (path + [nb])[:-1] = path.
        # So new_blocked = blocked | set(NEIGHBORS[cur]).
        # Note: `blocked` is neighbors of path[:-1].
        # `NEIGHBORS[cur]` are neighbors of the current last vertex.
        # So `new_blocked` is neighbors of path.
        
        new_blocked = blocked | set(NEIGHBORS[cur])
        
        candidates = []
        for nb in NEIGHBORS[cur]:
            if nb in used:
                continue
            if nb in blocked:
                continue
            
            # Calculate Warnsdorff score
            # Count valid next steps from nb
            # Next step y must be in NEIGHBORS[nb]
            # y not in (used | {nb})
            # y not in new_blocked
            
            score = 0
            for y in NEIGHBORS[nb]:
                if y in used:
                    continue
                if y == nb:
                    continue
                if y in new_blocked:
                    continue
                score += 1
            
            candidates.append((score, nb))
        
        if not candidates:
            return
        
        # Sort by score (Warnsdorff)
        candidates.sort(key=lambda x: x[0])
        
        # Add some randomness to break ties or explore alternatives
        # If we always pick the min, we might get stuck.
        # We can pick the min, but if it fails, backtrack.
        # Or pick randomly from the lowest few?
        # Standard Warnsdorff picks the min. Let's try that first.
        # To allow backtracking to find longer paths, we need to try alternatives if the best fails.
        
        for score, nb in candidates:
            # Try nb
            path.append(nb)
            used.add(nb)
            # blocked does NOT change for the validity of nb itself, 
            # but for the NEXT call, we need to pass the updated state.
            # The `blocked` passed to the next dfs call should be `new_blocked`?
            # No.
            # In dfs(path, used, blocked), `blocked` is neighbors of path[:-1].
            # If we call dfs(path+[nb], used|{nb}, new_blocked), 
            # then in the next call, `blocked` is neighbors of (path+[nb])[:-1] = neighbors of path.
            # This matches `new_blocked`.
            
            dfs(path, used, new_blocked)
            
            # Backtrack
            path.pop()
            used.remove(nb)
            
            if time.time() > deadline:
                return
            if best_len >= len(path) + 50: # Heuristic: if we found a very good path, stop?
                # Actually, just let it run or break if we hit a global target?
                # Let's just rely on time.
                pass

    dfs(path, used, set())

    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    # Try multiple random starts
    # Also try start 0
    starts = [0] + [rng.randint(0, SIZE - 1) for _ in range(50)]
    
    for s in starts:
        if time.time() > deadline:
            break
        
        # Allocate a small time slice for each start
        # Or just run until deadline and keep best?
        # DFS can be very fast or very slow.
        # Let's limit each run to 10s? Or just run and let the global deadline handle it.
        # If one run takes too long, it might miss better starts.
        # Let's use a sub-deadline for each start.
        
        sub_deadline = min(deadline, time.time() + 10)
        
        path = _solve_backtrack(s, sub_deadline, rng)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
        # If we found a very long path (e.g., > 150), we can stop?
        # The max is ~190.
        # If we get 180+, we can be more conservative with time.
        
    return best_path
