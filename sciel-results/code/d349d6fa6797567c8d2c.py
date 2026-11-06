"""Snake-in-the-box Q_9: Warnsdorff Backtracking + Splicing."""
import time
import random
import sys

N = 9
TIME_LIMIT = 145  # Safety margin below 150s hard kill
MASK = (1 << (1 << N)) - 1  # 512-bit mask for bit-parallel exclusion

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return TIME_LIMIT - elapsed()

    def get_neighbors(v):
        """Generate all N neighbors of v."""
        for b in range(N):
            yield v ^ (1 << b)

    def warnsdorff_search(start_vertex, bit_order, max_depth=512):
        """
        Depth-first search using Warnsdorff's heuristic.
        Returns the longest path found.
        """
        path = [start_vertex]
        used_mask = 1 << start_vertex
        # forbidden_mask: neighbors of all vertices in path except the last one
        # This is a standard optimization for snake-in-the-box: 
        # A vertex is invalid if it is used OR if it is a neighbor of any non-terminal vertex.
        # For the last vertex, its neighbors are candidates (unless used).
        
        # We maintain a 'forbidden' set which is the union of neighbors of path[0..-2].
        # A candidate v is valid if:
        # 1. v is not in used_mask
        # 2. v is not in forbidden_mask
        # Note: v might be a neighbor of path[-1]. That is allowed.
        
        forbidden_mask = 0
        
        # Helper to get valid candidates
        def get_candidates(cur, used, forb):
            cands = []
            for b in bit_order:
                v = cur ^ (1 << b)
                v_bit = 1 << v
                if (used | forb) & v_bit == 0:
                    # Calculate forward degree: how many neighbors of v are also valid?
                    # A neighbor w of v is valid if w is not used and not forbidden.
                    # Note: w == cur is not valid because cur is used.
                    # So we count neighbors w of v such that w is not in (used | forb).
                    # Actually, w is in forb if w is neighbor of path[0..-2].
                    # w is in used if w is in path.
                    # So we just check (used | forb).
                    score = 0
                    for b2 in bit_order:
                        w = v ^ (1 << b2)
                        if w == cur:
                            continue
                        w_bit = 1 << w
                        if (used | forb) & w_bit == 0:
                            score += 1
                    cands.append((score, v))
            return cands

        # Iterative DFS with explicit stack to handle deep recursion
        # Stack elements: (path, used_mask, forbidden_mask, candidates_list, next_index)
        # We store the state of the search.
        
        # To be efficient, we don't store the full path in the stack, but the length and the array.
        # Since max depth is 512, storing a list of 512 ints is fine.
        
        # Let's use a simple recursive approach with sys.setrecursionlimit if needed, 
        # but iterative is safer for memory if we branch. 
        # However, Warnsdorff is usually a single path with backtracking.
        
        # We will implement a manual stack for backtracking.
        
        best_path = list(path)
        
        # State: 
        # path: list of vertices
        # used_mask: int
        # forbidden_mask: int
        # stack of (vertex, candidates, index)
        
        # Let's restructure to a standard DFS
        # We'll use a stack of states.
        # State: (current_vertex, used_mask, forbidden_mask, path_tuple)
        # This is memory heavy. 
        # Alternative:
        # Since N=9, max depth 512.
        # We can use a global path array and backtrack in place.
        
        path_arr = [0] * 512
        path_arr[0] = start_vertex
        current_len = 1
        used_mask = 1 << start_vertex
        forbidden_mask = 0
        
        # We need to track the choices made at each step to backtrack.
        # choices[i] = list of candidates available at step i, and which one we picked.
        # Or simply, at each step, we sort candidates and try them one by one.
        
        # Let's use a recursive function for clarity, with a depth limit.
        # Python recursion limit is usually 1000, so 512 is safe.
        
        def dfs(cur, used, forb, depth):
            nonlocal best_path
            if depth > len(best_path):
                best_path = list(path_arr[:depth])
            
            if depth >= 512:
                return
            
            if time_left() < 1.0:
                return
            
            cur = path_arr[depth-1]
            
            # Generate and sort candidates
            cands = []
            for b in bit_order:
                v = cur ^ (1 << b)
                v_bit = 1 << v
                if (used | forb) & v_bit == 0:
                    # Score: forward degree
                    score = 0
                    for b2 in bit_order:
                        w = v ^ (1 << b2)
                        if w == cur:
                            continue
                        w_bit = 1 << w
                        if (used | forb) & w_bit == 0:
                            score += 1
                    cands.append((score, v))
            
            if not cands:
                return
            
            # Sort by score (Warnsdorff: fewest onward moves first)
            cands.sort(key=lambda x: x[0])
            
            for score, v in cands:
                if time_left() < 1.0:
                    return
                
                # Move to v
                path_arr[depth] = v
                new_used = used | (1 << v)
                
                # Update forbidden: 
                # The new forbidden mask is old_forbidden | neighbors(cur)
                # Because cur is no longer the end, its neighbors become forbidden.
                new_forb = forb
                for nb in get_neighbors(cur):
                    new_forb |= (1 << nb)
                
                dfs(v, new_used, new_forb, depth + 1)
                
                # Backtrack is implicit in recursion stack

        dfs(start_vertex, used_mask, forbidden_mask, 1)
        
        return best_path

    def try_splice(path1, path2):
        """
        Try to splice two disjoint paths into one.
        If path1 ends at v1 and path2 starts at v2, and v1, v2 are adjacent,
        we can merge them if the merge is valid (no conflicts).
        This is complex to verify quickly. 
        Given time, we will focus on the strength of Warnsdorff DFS first.
        If DFS yields short paths, we might need this.
        For now, return path1 if longer.
        """
        if len(path1) >= len(path2):
            return path1
        return path2

    best_path = []
    best_len = 0
    
    # Phase 1: Warnsdorff DFS from various starts
    # We run multiple instances with different random bit orders to break symmetry/ties.
    
    # Determine number of restarts based on time
    # Each DFS run might take a few seconds if it goes deep.
    # We will run as many as we can.
    
    attempts = 0
    while time_left() > 5.0:
        sv = random.randint(0, 2**N - 1)
        bo = list(range(N))
        random.shuffle(bo)
        
        # Run DFS
        p = warnsdorff_search(sv, bo)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            # If we find a very long path, we might stop early? 
            # No, keep searching for improvements.
        
        attempts += 1
        
        # If we are taking too long per attempt, we might need to limit depth.
        # But Warnsdorff is usually fast.
        
    # Phase 2: Local Improvement / Splicing (Simple version)
    # Try to extend the best path from both ends greedily.
    if best_path and time_left() > 2.0:
        # Extend forward
        cur = best_path[-1]
        used = set(best_path)
        forbidden = set()
        for i in range(len(best_path) - 1):
            for nb in get_neighbors(best_path[i]):
                forbidden.add(nb)
        
        improved = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and v not in forbidden:
                best_path.append(v)
                improved = True
                # Recalculate forbidden for the new end? 
                # The new end is v. The previous end cur becomes internal.
                # So neighbors of cur become forbidden.
                for nb in get_neighbors(cur):
                    forbidden.add(nb)
                used.add(v)
                cur = v
                # Continue extending?
                # We can do a loop, but let's keep it simple for now.
                break
        
        # Extend backward
        cur = best_path[0]
        used = set(best_path)
        forbidden = set()
        for i in range(1, len(best_path)):
            for nb in get_neighbors(best_path[i]):
                forbidden.add(nb)
        
        improved = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and v not in forbidden:
                best_path.insert(0, v)
                improved = True
                for nb in get_neighbors(cur):
                    forbidden.add(nb)
                used.add(v)
                cur = v
                break

    return best_path
# EVOLVE-BLOCK-END
