"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

def construct():
    """
    Returns 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
    Every set of four vertices must span edges of at least five distinct colours.
    Minimize the number of colours used.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline to be safe within 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120 edges
    
    # Precompute all K4s and their edge indices
    # Map (i,j) to index
    idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            idx_map[(i,j)] = idx
            idx += 1
    
    # Precompute K4 structure
    k4_list = list(itertools.combinations(range(n), 4))
    k4_edges = []  # list of 6 edge indices for each K4
    edge_to_k4 = [[] for _ in range(num_edges)]  # for each edge, list of K4 indices
    
    for k4_idx, v in enumerate(k4_list):
        a, b, c, d = v
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        e_indices = []
        for e in edges:
            i, j = e
            if i > j:
                i, j = j, i
            ei = idx_map[(i,j)]
            e_indices.append(ei)
            edge_to_k4[ei].append(k4_idx)
        k4_edges.append(e_indices)
    
    def count_violations(colors):
        """Count number of K4s with < 5 distinct colours."""
        violations = 0
        for e_indices in k4_edges:
            col_set = set()
            for ei in e_indices:
                col_set.add(colors[ei])
            if len(col_set) < 5:
                violations += 1
        return violations
    
    def is_valid(colors):
        return count_violations(colors) == 0
    
    def num_colours_used(colors):
        return len(set(colors))
    
    # Start with the base solution: (i+j) mod 17
    base_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = base_colors.copy()
    best_num = num_colours_used(base_colors)  # should be 17
    
    # Verify base is valid
    if not is_valid(base_colors):
        raise ValueError("Base solution invalid")
    
    # Local search: try to reduce number of colours
    # Strategy: Start from base, try to eliminate colours one by one
    
    def try_eliminate_colour(colors, target_col):
        """Try to eliminate target_col by recoloring all edges with that colour.
        Returns (new_colors, success) where success indicates if target_col was eliminated."""
        cur = colors.copy()
        edges_with_col = [i for i in range(num_edges) if cur[i] == target_col]
        
        if not edges_with_col:
            return cur, True
        
        # Get available colours (excluding target_col)
        available_cols = [c for c in set(cur) if c != target_col]
        
        if not available_cols:
            return cur, False
        
        # For each edge with target_col, try to recolor to another colour
        # Use backtracking with forward checking
        for ei in edges_with_col:
            found = False
            # Try available colours in random order
            random.shuffle(available_cols)
            for new_col in available_cols:
                old = cur[ei]
                cur[ei] = new_col
                # Check if validity is maintained (only affected K4s)
                ok = True
                for k4i in edge_to_k4[ei]:
                    e_indices = k4_edges[k4i]
                    col_set = set()
                    for ej in e_indices:
                        col_set.add(cur[ej])
                    if len(col_set) < 5:
                        ok = False
                        break
                if ok:
                    found = True
                    break
                cur[ei] = old
            
            if not found:
                # Revert all changes
                for ej in edges_with_col:
                    if cur[ej] != target_col:
                        # This is tricky - we need to track original values
                        pass
                return colors, False
        
        # Verify target_col is no longer used
        if target_col not in set(cur):
            return cur, True
        else:
            return colors, False
    
    # More robust approach: local search that minimizes violations, then colours
    def local_search(colors, max_time=None):
        """Local search to reduce number of colours while maintaining validity."""
        cur = colors.copy()
        cur_viol = count_violations(cur)
        cur_cols = set(cur)
        cur_num = len(cur_cols)
        best = cur.copy()
        best_num = cur_num
        best_viol = cur_viol
        
        # Precompute for faster violation checking
        # For each K4, cache the set of colours
        k4_col_sets = []
        for e_indices in k4_edges:
            col_set = set()
            for ei in e_indices:
                col_set.add(cur[ei])
            k4_col_sets.append(col_set)
        
        def update_k4_col_set(k4i, edge_idx, old_col, new_col):
            """Update the colour set for K4 k4i when edge edge_idx changes from old_col to new_col."""
            e_indices = k4_edges[k4i]
            if edge_idx not in e_indices:
                return
            cs = k4_col_sets[k4i]
            # Remove old_col if no other edge in this K4 has it
            count_old = sum(1 for ej in e_indices if cur[ej] == old_col)
            if count_old == 1:  # This was the only edge with old_col
                cs.discard(old_col)
            # Add new_col
            cs.add(new_col)
        
        def get_violation_count():
            """Count violations from cached colour sets."""
            v = 0
            for cs in k4_col_sets:
                if len(cs) < 5:
                    v += 1
            return v
        
        cur_viol = get_violation_count()
        
        # Simulated annealing style local search
        temp = 1.0
        cooling = 0.995
        
        while time.time() < (max_time or deadline - 5):
            if temp < 0.01:
                break
            
            # Pick a random edge
            ei = random.randint(0, num_edges - 1)
            old_col = cur[ei]
            
            # Pick a random new colour (from current palette or a new one)
            if random.random() < 0.8:
                # Choose from current palette
                new_col = random.choice(list(cur_cols))
            else:
                # Choose a random colour (might be new)
                new_col = random.randint(0, best_num)
            
            if new_col == old_col:
                continue
            
            # Apply change
            cur[ei] = new_col
            
            # Update affected K4s
            for k4i in edge_to_k4[ei]:
                update_k4_col_set(k4i, ei, old_col, new_col)
            
            new_viol = get_violation_count()
            new_cols = set(cur)
            new_num = len(new_cols)
            
            # Acceptance criterion: prefer fewer violations, then fewer colours
            delta_viol = new_viol - cur_viol
            delta_num = new_num - cur_num
            
            # Composite cost: violations * 100 + num_colours
            old_cost = cur_viol * 100 + cur_num
            new_cost = new_viol * 100 + new_num
            delta_cost = new_cost - old_cost
            
            if delta_cost <= 0:
                # Always accept improvements
                cur_viol = new_viol
                cur_cols = new_cols
                cur_num = new_num
                if cur_viol == 0 and cur_num < best_num:
                    best = cur.copy()
                    best_num = cur_num
                    best_viol = 0
            else:
                # Metropolis acceptance
                if random.random() < min(1.0, __import__('math').exp(-delta_cost / temp)):
                    cur_viol = new_viol
                    cur_cols = new_cols
                    cur_num = new_num
                else:
                    # Revert
                    cur[ei] = old_col
                    for k4i in edge_to_k4[ei]:
                        update_k4_col_set(k4i, ei, new_col, old_col)
            
            temp *= cooling
        
        return best, best_num
    
    # Run local search
    # First, try a few different starting points
    random.seed(42)
    
    # Starting point 1: base
    best1, num1 = local_search(base_colors, max_time=deadline - 60)
    if num1 < best_num:
        best_colors = best1
        best_num = num1
    
    # Starting point 2: random valid colouring with 15 colours
    # Generate random colouring and try to fix it
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
        
        # Random initial colouring with 15 colours
        rand_colors = [random.randint(0, 14) for _ in range(num_edges)]
        viol = count_violations(rand_colors)
        
        # If already valid, great. Otherwise, run local search to fix
        if viol == 0:
            rand_num = num_colours_used(rand_colors)
            if rand_num < best_num:
                best_colors = rand_colors
                best_num = rand_num
        else:
            # Run local search to try to fix violations
            fixed, fixed_num = local_search(rand_colors, max_time=min(deadline - 30, time.time() + 5))
            if fixed_num < best_num:
                best_colors = fixed
                best_num = fixed_num
    
    # Final verification
    final_viol = count_violations(best_colors)
    if final_viol > 0:
        # Fall back to base
        best_colors = base_colors
        best_num = num_colours_used(base_colors)
    
    # Renormalize colours to 0..best_num-1
    col_map = {}
    new_colors = []
    next_col = 0
    for c in best_colors:
        if c not in col_map:
            col_map[c] = next_col
            next_col += 1
        new_colors.append(col_map[c])
    
    return new_colors

# EVOLVE-BLOCK-END
