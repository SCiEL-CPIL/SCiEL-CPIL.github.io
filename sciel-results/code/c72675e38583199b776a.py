import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 256
    
    # Perfect squares up to 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
        
    # Diagonal constraint: 2a is a perfect square
    # a = 2 * m^2
    diagonal_bad = set()
    m = 1
    while 2 * m * m <= n:
        diagonal_bad.add(2 * m * m)
        m += 1
        
    # Available vertices
    available_vertices = [i for i in range(1, n + 1) if i not in diagonal_bad]
    available_set = set(available_vertices)
    
    # Build adjacency lists
    conflicts = {v: set() for v in available_vertices}
    
    for a in available_vertices:
        for b in available_vertices:
            if a < b and (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    # Greedy Initialization using Minimum Degree Heuristic
    def greedy_init():
        current_set = set()
        remaining = set(available_vertices)
        
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            current_set.add(best_v)
            to_remove = {best_v} | (conflicts[best_v] & remaining)
            remaining -= to_remove
            
        return current_set

    def local_search(s, time_limit):
        current = set(s)
        in_set = set(current)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try adding any vertex that is not in set and has no conflicts with set
            for v in available_vertices:
                if v in in_set:
                    continue
                if not (conflicts[v] & in_set):
                    current.add(v)
                    in_set.add(v)
                    improved = True
                    break
            
            if not improved:
                # 2. Try 1-for-2 swap
                swap_done = False
                for u in list(current):
                    candidates = []
                    for v in available_vertices:
                        if v in in_set:
                            continue
                        if u not in conflicts[v]:
                            continue
                        common = conflicts[v] & in_set
                        if len(common) == 1 and u in common:
                            candidates.append(v)
                    
                    if len(candidates) >= 2:
                        found_swap = False
                        for i in range(len(candidates)):
                            if found_swap: break
                            for j in range(i + 1, len(candidates)):
                                if candidates[i] not in conflicts[candidates[j]]:
                                    current.remove(u)
                                    in_set.remove(u)
                                    current.add(candidates[i])
                                    in_set.add(candidates[i])
                                    current.add(candidates[j])
                                    in_set.add(candidates[j])
                                    improved = True
                                    swap_done = True
                                    found_swap = True
                                    break
                        if swap_done:
                            break
                
                if not swap_done:
                    # 3. Perturbation
                    if time.time() > time_limit - 2:
                        break
                        
                    k_remove = random.randint(1, 3)
                    to_remove = random.sample(list(current), min(k_remove, len(current)))
                    for v in to_remove:
                        current.remove(v)
                        in_set.remove(v)
                    
                    refill_improved = True
                    while refill_improved:
                        refill_improved = False
                        for v in available_vertices:
                            if v in in_set:
                                continue
                            if not (conflicts[v] & in_set):
                                current.add(v)
                                in_set.add(v)
                                refill_improved = True
                                break
                                
        return current

    def branch_and_bound_mis(vertices, time_limit, current_best):
        """
        Branch and bound for maximum independent set.
        vertices: set of candidate vertices to consider
        time_limit: absolute time limit
        current_best: current best solution size
        Returns: best independent set found within time limit
        """
        best_set = set()
        best_size = 0
        
        # Precompute: for quick degree checks
        def get_neighbors(v, candidate_set):
            return conflicts[v] & candidate_set
        
        def clique_upper_bound(candidate_set):
            """
            Compute a lower bound on the clique number of the conflict graph
            restricted to candidate_set. This gives an upper bound on the MIS size.
            Uses greedy coloring to bound the clique number.
            """
            if not candidate_set:
                return 0
            
            # Greedy coloring of the conflict graph on candidate_set
            # The number of colors used is an upper bound on the clique number
            # which is a lower bound on the chromatic number of the complement
            # which is an upper bound on the independence number
            uncolored = set(candidate_set)
            colors = {}
            color_count = 0
            
            while uncolored:
                color_count += 1
                # Pick a vertex with minimum degree in uncolored
                min_deg = float('inf')
                best_v = None
                for v in uncolored:
                    deg = len(conflicts[v] & uncolored)
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                
                if best_v is None:
                    break
                    
                # Assign this color to best_v and all non-adjacent uncolored vertices
                # that can take this color
                this_color_set = set()
                this_color_set.add(best_v)
                uncolored.remove(best_v)
                
                # Add other vertices that don't conflict with best_v
                for v in list(uncolored):
                    if v not in conflicts[best_v]:
                        this_color_set.add(v)
                        uncolored.remove(v)
                
                for v in this_color_set:
                    colors[v] = color_count
            
            return color_count
        
        def dfs(candidate_set, current_set, current_size):
            nonlocal best_set, best_size
            
            if time.time() > time_limit:
                return False
            
            # Update best if improved
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            
            if not candidate_set:
                return True
            
            # Upper bound: current_size + clique_upper_bound(candidate_set)
            # If this doesn't beat best_size, prune
            ub = current_size + clique_upper_bound(candidate_set)
            if ub <= best_size:
                return True  # Pruned
            
            # Branch on a vertex
            # Choose vertex with minimum degree in candidate_set
            min_deg = float('inf')
            branch_v = None
            for v in candidate_set:
                deg = len(conflicts[v] & candidate_set)
                if deg < min_deg:
                    min_deg = deg
                    branch_v = v
            
            if branch_v is None:
                return True
            
            # Option 1: Include branch_v
            new_candidate = candidate_set - {branch_v} - conflicts[branch_v]
            new_current = current_set | {branch_v}
            
            if dfs(new_candidate, new_current, current_size + 1):
                return True
            
            # Option 2: Exclude branch_v
            new_candidate = candidate_set - {branch_v}
            
            return dfs(new_candidate, current_set, current_size)
        
        # Start DFS
        dfs(set(vertices), set(), 0)
        
        return best_set

    best_set = set()
    best_size = 0
    
    # Initial greedy
    s = greedy_init()
    best_set = set(s)
    best_size = len(s)
    
    # Local search on initial
    s = local_search(s, deadline - 10)
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)
    
    # Multiple restarts
    random.seed(42)
    
    while time.time() < deadline - 3:
        # Randomized Greedy Init
        order = available_vertices[:]
        random.shuffle(order)
        
        current = set()
        in_set = set()
        for v in order:
            if not (conflicts[v] & in_set):
                current.add(v)
                in_set.add(v)
                
        # Local search
        current = local_search(current, deadline - 5)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
        
        # Try branch-and-bound on a subset around the current best
        # Only if we have time
        if time.time() < deadline - 10 and len(available_vertices) <= 80:
            # Use B&B on the full problem if small enough
            bb_result = branch_and_bound_mis(set(available_vertices), deadline - 5, best_size)
            if len(bb_result) > best_size:
                best_size = len(bb_result)
                best_set = set(bb_result)
            
    return list(best_set)
