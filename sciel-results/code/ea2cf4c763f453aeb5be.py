import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
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
    N = len(available_vertices)
    
    # Build adjacency lists
    # conflicts[v] = set of u such that u+v is square and u != v
    conflicts = {v: set() for v in available_vertices}
    
    for a in available_vertices:
        for b in available_vertices:
            if a < b and (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    # Map vertex to index for faster bitset operations
    v_to_idx = {v: i for i, v in enumerate(available_vertices)}
    idx_to_v = {i: v for i, v in enumerate(available_vertices)}
    
    # Bitset representation of conflicts
    # conflict_bits[i] is a bitmask where bit j is set if idx_to_v[i] conflicts with idx_to_v[j]
    conflict_bits = [0] * N
    for i, v in enumerate(available_vertices):
        bits = 0
        for u in conflicts[v]:
            bits |= (1 << v_to_idx[u])
        conflict_bits[i] = bits
    
    # Greedy Initialization using Minimum Degree Heuristic
    def greedy_init():
        current_set = set()
        remaining = set(available_vertices)
        
        while remaining:
            # Find vertex with minimum degree in the induced subgraph of 'remaining'
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
            # Remove best_v and its neighbors from remaining
            to_remove = {best_v} | (conflicts[best_v] & remaining)
            remaining -= to_remove
            
        return current_set

    # Local Search with 1-for-2 swaps
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
                # Check if v has any neighbor in current
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
                    
                    # Refill greedily
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

    # Branch and Bound Maximum Independent Set solver
    def branch_and_bound(initial_set, time_limit):
        """
        Branch and bound MIS solver using bitsets.
        Returns the maximum independent set found.
        """
        # Convert initial_set to a bitset of vertices to include
        # We'll work with the complement: vertices NOT in initial_set that we might add
        # Actually, let's work directly: we want to maximize the independent set.
        
        # Start with the initial set as a lower bound
        best_set = set(initial_set)
        best_size = len(best_set)
        
        # Build the graph of vertices we can potentially add
        # Vertices not in initial_set
        candidate_vertices = [v for v in available_vertices if v not in initial_set]
        
        if not candidate_vertices:
            return best_set
        
        # Build conflict graph among candidate vertices
        # But we also need to consider conflicts with initial_set
        # A vertex v can be added if it has no conflicts with initial_set
        # Let's filter out vertices that conflict with initial_set
        valid_candidates = []
        for v in candidate_vertices:
            if not (conflicts[v] & initial_set):
                valid_candidates.append(v)
        
        if not valid_candidates:
            return best_set
        
        # Build bitset for valid_candidates
        cand_to_idx = {v: i for i, v in enumerate(valid_candidates)}
        M = len(valid_candidates)
        
        # conflict_bits among candidates
        cand_conflict_bits = [0] * M
        for i, v in enumerate(valid_candidates):
            bits = 0
            for u in conflicts[v]:
                if u in cand_to_idx:
                    bits |= (1 << cand_to_idx[u])
            cand_conflict_bits[i] = bits
        
        # Now solve MIS on the candidate graph
        # Using a recursive branch and bound approach
        
        def solve_mis(vertices, current_set, current_size):
            """
            vertices: list of candidate indices
            current_set: set of candidate indices selected so far
            current_size: number selected so far
            """
            nonlocal best_set, best_size
            
            if time.time() > time_limit:
                return
            
            if not vertices:
                if current_size > best_size:
                    best_size = current_size
                    # Reconstruct best_set
                    new_best = set(initial_set)
                    for idx in current_set:
                        new_best.add(valid_candidates[idx])
                    best_set = new_best
                return
            
            # Upper bound: current_size + len(vertices)
            if current_size + len(vertices) <= best_size:
                return
            
            # Select a pivot vertex
            # Use the vertex with maximum degree in the remaining graph
            pivot = max(vertices, key=lambda i: bin(cand_conflict_bits[i] & ((1 << M) - 1)).count('1') if i in vertices else -1)
            
            # Branch 1: Include pivot
            new_current_set = set(current_set)
            new_current_set.add(pivot)
            
            # Remove pivot and its neighbors from vertices
            new_vertices = []
            for i in vertices:
                if i == pivot:
                    continue
                if cand_conflict_bits[pivot] & (1 << i):
                    continue
                new_vertices.append(i)
            
            solve_mis(new_vertices, new_current_set, current_size + 1)
            
            # Branch 2: Exclude pivot
            new_vertices = [i for i in vertices if i != pivot]
            solve_mis(new_vertices, current_set, current_size)
        
        # Initial call
        all_indices = list(range(M))
        solve_mis(all_indices, set(), 0)
        
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
    
    # Try branch and bound on the local search result
    if time.time() < deadline - 5:
        s = branch_and_bound(best_set, deadline - 5)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    # Multiple restarts
    random.seed(42)
    
    while time.time() < deadline - 2:
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
        current = local_search(current, deadline - 2)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
            
    return list(best_set)
