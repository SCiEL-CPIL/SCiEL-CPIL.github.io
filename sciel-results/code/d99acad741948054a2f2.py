import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    # Precompute edge indices
    # Edge (i, j) with 0 <= i < j < 16
    # Index of (i, j) = sum_{k=0}^{i-1} (15-k) + (j - i - 1)
    # Sum_{k=0}^{i-1} (15-k) = 15*i - i*(i-1)/2
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return 15*i - i*(i-1)//2 + (j - i - 1)

    # Precompute all K4s
    vertices = list(range(16))
    k4s = []
    for quad in itertools.combinations(vertices, 4):
        v0, v1, v2, v3 = quad
        e1 = get_edge_idx(v0, v1)
        e2 = get_edge_idx(v0, v2)
        e3 = get_edge_idx(v0, v3)
        e4 = get_edge_idx(v1, v2)
        e5 = get_edge_idx(v1, v3)
        e6 = get_edge_idx(v2, v3)
        k4s.append((e1, e2, e3, e4, e5, e6))
        
    num_k4s = len(k4s)
    
    # Map edge to list of K4 indices
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edges in enumerate(k4s):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    # Helper to check validity of a full assignment
    def is_valid(colors):
        for edges in k4s:
            if len(set(colors[e] for e in edges)) < 5:
                return False
        return True

    # Generate deterministic seeds based on algebraic structures
    # We want to minimize colors. Let's try to construct solutions with C colors.
    
    def generate_algebraic_seed(C):
        # Try (i + j) % C
        # If C is small, collisions might be bad, but it's a good start.
        # We can also try (i * j) % C or other linear combos.
        # For now, stick to (i+j) % C as it's a standard 1-factorization related structure.
        return [(i + j) % C for i in range(16) for j in range(i + 1, 16)]
    
    def generate_random_seed(C):
        return [random.randint(0, C - 1) for _ in range(120)]

    # Backtracking solver with forward checking
    # We will solve for a specific number of colors C.
    # State: colors[120], some might be -1 (unassigned)
    # We assign edges one by one. To make it efficient, we don't just pick random order.
    # We pick the edge that is part of the most "tight" constraints? 
    # Actually, simple DFS on edges is 120 deep. Branching factor C.
    # With C=13, 13^120 is huge. We need strong pruning.
    
    # Pruning: For every K4, if the number of distinct colors among assigned edges + (number of unassigned edges in K4) < 5, then it's impossible.
    # Also, if the number of distinct colors among assigned edges is already >= 5, that K4 is "satisfied" (we don't need to check it further for this constraint, but we still need to ensure other K4s are satisfied).
    # Wait, the constraint is "at least 5". So if we have 5, we are safe for that K4.
    # If we have < 5, we MUST be able to fill the rest with enough new colors.
    
    # Let's implement a specialized backtracking.
    # We maintain:
    # - colors: list of 120 ints, -1 for unassigned
    # - k4_status: for each K4, count of distinct colors present in assigned edges.
    #   Actually, we need to know WHICH colors are present to count distinct ones correctly upon assignment.
    #   So for each K4, maintain a set of colors present in assigned edges.
    #   And count of unassigned edges in that K4.
    
    # Optimization: Order edges dynamically? 
    # Static order might be okay if we are clever.
    # Let's use a static order but check constraints aggressively.
    
    # To speed up, we can use a heuristic: assign edges in an order that covers K4s quickly.
    # Or just random order of edges.
    
    # Let's try a recursive solver.
    # To avoid recursion depth issues or slow Python recursion, we might use iterative or careful recursion.
    # 120 depth is fine.
    
    # We will try to solve for C = 13, 14, 15.
    
    # Precompute edge order. Let's shuffle it for variety in different runs/restarts.
    # But for backtracking, a fixed order is better for consistency within a run.
    
    # We'll use a "Most Constrained Variable" heuristic if possible, but static is simpler.
    # Let's try static order: 0..119.
    
    # To make it faster, we can precompute for each edge, which K4s it belongs to. (Already done: edge_to_k4s)
    
    # State for backtracking:
    # colors: list
    # k4_colors: list of sets (colors present in assigned edges of that K4)
    # k4_unassigned: list of ints (count of unassigned edges in that K4)
    
    # Initial state for a given seed (partial or full):
    # If we start with a full seed, we check if valid. If not, we "unassign" some edges to fix?
    # No, backtracking usually starts from scratch or a partial assignment.
    # If we start from a full assignment that is invalid, we can identify violating K4s and "unassign" edges in them to create a partial assignment, then backtrack.
    
    # Strategy:
    # 1. Try algebraic seeds for C=13, 14, 15. Check validity.
    # 2. If valid, return.
    # 3. If not, use the seed as a "hint" for backtracking?
    #    Actually, let's just run backtracking from scratch for C=13.
    #    If it finds a solution, return.
    #    If not (timeout or stuck), try C=14.
    
    # To make backtracking efficient, we need good pruning.
    # Pruning Rule:
    # For any K4, let S be the set of colors in assigned edges.
    # Let U be the number of unassigned edges.
    # Max possible distinct colors = len(S) + U.
    # If len(S) + U < 5, FAIL.
    
    # Also, we can order edges such that we assign edges in K4s that are "critical" first.
    # Let's just use a random permutation of edges for the search order to avoid bias.
    
    def solve_backtracking(C, seed_colors=None):
        # seed_colors: if provided, use as initial assignment (must be valid colors 0..C-1)
        # We will try to fix it or complete it.
        # If seed_colors is None, start with all -1.
        
        colors = [-1] * 120
        if seed_colors:
            colors = seed_colors[:]
            # Check if any are invalid? We assume valid.
        
        # Initialize K4 states
        k4_colors = [set() for _ in range(num_k4s)]
        k4_unassigned = [6] * num_k4s # 6 edges per K4
        
        for e in range(120):
            if colors[e] != -1:
                c = colors[e]
                for k4_idx in edge_to_k4s[e]:
                    k4_colors[k4_idx].add(c)
                    k4_unassigned[k4_idx] -= 1
            # else: unassigned, k4_unassigned stays as is (or decremented if we track assigned count differently)
            # Wait, k4_unassigned should be the count of UNASSIGNED edges.
            # Initially 6. If an edge is assigned, decrement.
        
        # Check initial state for immediate failures
        for i in range(num_k4s):
            if len(k4_colors[i]) + k4_unassigned[i] < 5:
                return None
        
        # Determine search order
        # We pick edges that are unassigned.
        # To be efficient, we can sort unassigned edges by some heuristic.
        # Heuristic: Edges in K4s that are "close to failure" (low len(S) + U) are more critical.
        # However, calculating this dynamically is expensive.
        # Let's use a static random order of edges.
        edge_order = list(range(120))
        random.shuffle(edge_order)
        
        # We need to skip already assigned edges in the order.
        # Or better, build a list of unassigned edges and shuffle that.
        unassigned_edges = [e for e in range(120) if colors[e] == -1]
        random.shuffle(unassigned_edges)
        
        # If all assigned, check validity
        if not unassigned_edges:
            # Check if valid
            for i in range(num_k4s):
                if len(k4_colors[i]) < 5:
                    return None
            return colors[:]

        # Recursive DFS
        # We use a stack to simulate recursion to avoid overhead? 
        # Python recursion limit is usually 1000, so 120 is fine.
        # But we need to pass state or use globals/closures.
        # Using a class or closure with nonlocal is cleaner.
        
        # Let's use a standard recursive function.
        # To speed up, we can use "fail-first" ordering: 
        # At each step, pick the unassigned edge that is in the K4 with the fewest "available" slots?
        # This is complex. Let's stick to random order first.
        
        # To allow early termination, we check time in the recursion.
        
        def dfs(idx):
            if time.time() > deadline:
                return None
            
            if idx == len(unassigned_edges):
                # All edges assigned
                for i in range(num_k4s):
                    if len(k4_colors[i]) < 5:
                        return None
                return colors[:]
            
            e = unassigned_edges[idx]
            
            # Try each color 0..C-1
            # Heuristic: Try colors that are already in the K4s? Or new colors?
            # For "at least 5", adding a NEW color is often better than repeating.
            # Let's shuffle colors to avoid bias.
            colors_to_try = list(range(C))
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                # Apply change
                colors[e] = c
                
                # Update K4 states
                affected_k4s = edge_to_k4s[e]
                old_states = []
                valid = True
                
                for k4_idx in affected_k4s:
                    old_set = k4_colors[k4_idx]
                    old_unassigned = k4_unassigned[k4_idx]
                    
                    # New state
                    new_set = set(old_set)
                    new_set.add(c)
                    new_unassigned = old_unassigned - 1
                    
                    # Pruning check
                    if len(new_set) + new_unassigned < 5:
                        valid = False
                        break
                    
                    # Store for rollback
                    old_states.append((k4_idx, old_set, old_unassigned))
                    
                    # Tentatively update
                    k4_colors[k4_idx] = new_set
                    k4_unassigned[k4_idx] = new_unassigned
                
                if valid:
                    res = dfs(idx + 1)
                    if res is not None:
                        return res
                
                # Rollback
                for k4_idx, old_set, old_unassigned in old_states:
                    k4_colors[k4_idx] = old_set
                    k4_unassigned[k4_idx] = old_unassigned
                colors[e] = -1
                
            return None
        
        return dfs(0)

    # Main logic
    best_solution = None
    best_color_count = 100
    
    # Try C = 13, 14, 15
    # We allocate time to each.
    # 13: 40s, 14: 40s, 15: 30s
    
    targets = [13, 14, 15]
    time_alloc = {13: 40, 14: 40, 15: 30}
    
    for C in targets:
        if time.time() > deadline - 5:
            break
            
        slice_end = min(deadline - 2, time.time() + time_alloc[C])
        
        # Try multiple restarts with different seeds/orders
        found = False
        restarts = 0
        while time.time() < slice_end and not found:
            restarts += 1
            
            # Generate a seed
            # 50% algebraic, 50% random
            if random.random() < 0.5:
                seed = generate_algebraic_seed(C)
            else:
                seed = generate_random_seed(C)
            
            # Quick check if seed is already valid
            if is_valid(seed):
                best_solution = seed
                best_color_count = C
                found = True
                break
            
            # Run backtracking
            # We pass the seed to backtracking. 
            # Note: Backtracking will unassign some edges? 
            # My current solve_backtracking implementation takes a seed and treats assigned edges as FIXED.
            # If the seed has violations, the "fixed" edges might make it impossible to solve.
            # So, if we use a seed, we should only FIX edges that are NOT in violating K4s?
            # Or better: Just use the seed to initialize the *order* or *color preference*, but start with all unassigned?
            # Starting with all unassigned is safer for finding ANY solution.
            # The seed is just for color preference in the DFS?
            # Let's modify solve_backtracking to accept a "hint" for color ordering, but start empty.
            
            # Actually, let's just run solve_backtracking(C) without seed (starts empty).
            # The randomness in edge order and color order provides the restart diversity.
            
            sol = solve_backtracking(C)
            
            if sol is not None:
                best_solution = sol
                best_color_count = C
                found = True
                break
            
            # If we found a solution for C, we can break and try C-1?
            # The loop structure handles this: if found, break inner while, then outer for continues to next C (lower).
            # Wait, targets are [13, 14, 15]. So we try 13 first. If found, we return 13.
            # If not found in time, we try 14.
            
        # If found, we can return immediately if it's the minimum possible (13)
        if found and C == 13:
            return best_solution

    # If no solution found with backtracking (unlikely if C=15 is easy, but possible if timeout),
    # fallback to the seed program's logic or a simple valid solution.
    
    # Fallback: 17 colors (i+j)%17 is known to be valid? 
    # Let's verify (i+j)%17 for K4.
    # In K4 with vertices a,b,c,d.
    # Edges: a+b, a+c, a+d, b+c, b+d, c+d.
    # Mod 17.
    # It is known that (i+j) mod p for prime p >= n+1 works?
    # For n=16, p=17.
    # Let's just use the seed from the current program as a last resort.
    
    if best_solution is None:
        # Use the seed solution from the original program (17 colors)
        # (i+j) % 17
        best_solution = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        best_color_count = 17
        
        # Verify it
        if not is_valid(best_solution):
            # Panic: use 16 colors?
            alt = [(i + j) % 16 for i in range(16) for j in range(i + 1, 16)]
            if is_valid(alt):
                best_solution = alt
                best_color_count = 16
            else:
                # 15 colors?
                alt2 = [(i + j) % 15 for i in range(16) for j in range(i + 1, 16)]
                if is_valid(alt2):
                    best_solution = alt2
                    best_color_count = 15
                else:
                    # Distinct colors for all edges? 120 colors.
                    # This is always valid.
                    best_solution = list(range(120))
                    best_color_count = 120

    return best_solution
