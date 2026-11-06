import time
import sys
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute all 4-cycles (3 per 4-vertex subset)
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        cycles.append(((a,c), (c,b), (b,d), (d,a)))

    # Convert to edge indices
    cycle_indices = np.array([tuple(edge_idx[u][v] for u, v in c) for c in cycles], dtype=np.int32)

    # For each edge, which cycles does it belong to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, c_idx in enumerate(cycle_indices):
        for e_idx in c_idx:
            edge_to_cycles[e_idx].append(i)

    def greedy_construct(seed=0):
        """
        Randomized greedy construction:
        - Process edges in a random order.
        - For each edge, try colors from 0 to MAX_COLORS-1.
        - Assign the smallest color that doesn't create a rainbow 4-cycle.
        """
        rng = random.Random(seed)
        colors = [-1] * NUM_EDGES
        freq = [0] * MAX_COLORS
        used_colors = set()
        
        # Random order of edges
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        
        for e in edge_order:
            assigned = False
            # Try colors in increasing order to maximize reuse of low indices
            # but we want to maximize distinct colors, so maybe try unused first?
            # Actually, the "smallest valid color" strategy is standard for interval graphs etc.
            # Here, to maximize distinct colors, we should prefer UNUSED colors if possible,
            # or colors that are already used but allow future flexibility.
            # However, a simple "smallest valid" tends to minimize colors.
            # A "largest valid" or "random valid" might help.
            # Let's try: prefer unused colors, then smallest used.
            
            # Strategy: 
            # 1. If there's an unused color that is valid, use it.
            # 2. Else, use the smallest valid color.
            
            # To make it faster, we can just iterate 0..119 and check validity.
            # But checking validity for every color is expensive.
            # Optimization: Only check colors that are "candidates".
            # Candidates: 
            #   - All unused colors (if any)
            #   - All used colors
            
            # Actually, if we pick a color that is already used, we don't increase distinct count.
            # If we pick an unused color, we increase distinct count by 1.
            # So we should ALWAYS prefer an unused color if one is valid.
            
            # Let's collect valid colors.
            valid_colors = []
            
            # Check unused colors first (randomized order to avoid bias?)
            # Or just check 0..119 and pick the first valid one?
            # If we pick the first valid one, and it's used, we might miss an unused one that is valid.
            # So we must check ALL colors? No, that's 120 checks per edge * 120 edges = 14400 checks.
            # Each check involves ~10-20 cycles. 14400 * 20 = 288,000 operations.
            # This is fast enough for Python if optimized.
            
            # Let's try a different approach:
            # 1. Determine the set of "forbidden" colors for this edge.
            #    A color c is forbidden if assigning it to edge e creates a rainbow cycle.
            #    This happens if there is a cycle containing e where the other 3 edges have colors {c1, c2, c3}
            #    such that {c1, c2, c3} are all distinct and none is c, AND c is distinct from all of them.
            #    Wait, the condition is: all 4 colors distinct.
            #    So if we assign c to e, and the other 3 edges have colors x, y, z.
            #    Rainbow if c != x, c != y, c != z, and x != y, x != z, y != z.
            #    So c is forbidden if there exists a cycle (e, x_edge, y_edge, z_edge) such that
            #    colors[x_edge], colors[y_edge], colors[z_edge] are distinct from each other AND distinct from c.
            
            # This is complex to precompute dynamically.
            # Let's stick to the simple check but optimize it.
            
            # For each edge e, we have a list of cycles `edge_to_cycles[e]`.
            # For each cycle, we look at the other 3 edges.
            # If the other 3 edges have distinct colors, then the color assigned to e MUST be one of those 3 colors
            # to avoid a rainbow.
            # If the other 3 edges do NOT have distinct colors, then ANY color is safe for this cycle.
            
            # So, for edge e:
            # forbidden = set()
            # for each cycle in edge_to_cycles[e]:
            #   other_edges = [x for x in cycle if x != e]
            #   c1, c2, c3 = colors[other_edges[0]], colors[other_edges[1]], colors[other_edges[2]]
            #   if c1 != -1 and c2 != -1 and c3 != -1: # all assigned
            #       if c1 != c2 and c1 != c3 and c2 != c3:
            #           # Then c must be in {c1, c2, c3}
            #           forbidden.update([c1, c2, c3])
            #       # else: any color is fine for this cycle
            #   else:
            #       # some unassigned, so no constraint yet? 
            #       # Wait, if some are unassigned, they will be assigned later.
            #       # This greedy approach works if we process edges such that constraints are resolved?
            #       # No, in a random order, future edges are unassigned.
            #       # If an edge in the cycle is unassigned, we can't determine if it will be a rainbow yet.
            #       # However, if we assign e now, and later assign the others, we might create a rainbow.
            #       # So the greedy "smallest valid" is only safe if we consider the current state.
            #       # But the future assignments might break the current "validity".
            
            # This suggests that simple greedy on a random order might produce INVALID colorings
            # because we don't know the future colors.
            
            # To fix this, we can use a backtracking greedy, or we can process edges in an order
            # that minimizes future conflicts (e.g., high degree first).
            # Or, we can use the greedy approach but then VERIFY and if invalid, RESTART.
            # Given the high density of constraints, restarts might be frequent.
            
            # Alternative: Use the greedy logic but only as a STARTING POINT for a local search.
            # But the local search was stuck at 19.
            
            # Let's try a different structural change:
            # **Simulated Annealing with a better move operator.**
            # The current SA picks a random edge and a random color.
            # Most moves are invalid.
            # New Move: Pick an edge, and ONLY consider colors that are currently VALID for that edge
            # (based on the current coloring).
            # This drastically increases the acceptance rate.
            
            pass # Placeholder for the logic below

    def get_valid_colors_for_edge(e, colors):
        """
        Return a list of colors that can be assigned to edge e without creating a rainbow 4-cycle
        in the current state of `colors`.
        Note: This is a local check. It doesn't guarantee future validity, but it ensures
        that the move doesn't immediately break the constraint.
        """
        forbidden = set()
        for c_idx in edge_to_cycles[e]:
            ci = cycle_indices[c_idx]
            other_edges = [x for x in ci if x != e]
            c1 = colors[other_edges[0]]
            c2 = colors[other_edges[1]]
            c3 = colors[other_edges[2]]
            
            # If all three are assigned and distinct, then e must take one of these colors
            if c1 != c2 and c1 != c3 and c2 != c3:
                forbidden.add(c1)
                forbidden.add(c2)
                forbidden.add(c3)
        
        # Valid colors are those in 0..119 that are NOT in forbidden
        # But we also want to maximize distinct colors.
        # So we return the set of valid colors.
        valid = []
        for c in range(MAX_COLORS):
            if c not in forbidden:
                valid.append(c)
        return valid

    def check_validity_full(colors):
        """Full check for final verification."""
        c = np.array(colors, dtype=np.int32)
        for ci in cycle_indices:
            c0, c1, c2, c3 = c[ci[0]], c[ci[1]], c[ci[2]], c[ci[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # Multi-start Simulated Annealing with Valid-Move-Only
    best_colors = None
    best_score = 0
    
    # Initial coloring: All zeros (valid, 1 color)
    # Or use the hierarchical partition as a better start?
    # The hierarchical partition gave 15 colors.
    # Let's use a random valid start generated by the greedy method (with restarts if invalid).
    
    def generate_random_valid_start(seed):
        rng = random.Random(seed)
        colors = [0] * NUM_EDGES
        # Try to assign colors greedily in random order, restarting if stuck/invalid
        for attempt in range(20):
            trial_colors = [0] * NUM_EDGES
            edge_order = list(range(NUM_EDGES))
            rng.shuffle(edge_order)
            valid_so_far = True
            for e in edge_order:
                valid_cs = get_valid_colors_for_edge(e, trial_colors)
                if not valid_cs:
                    valid_so_far = False
                    break
                # Pick a random valid color, biased towards unused ones to maximize diversity
                # Count current usage
                freq = [0] * MAX_COLORS
                for c in trial_colors:
                    freq[c] += 1
                # Prefer unused
                unused_valid = [c for c in valid_cs if freq[c] == 0]
                if unused_valid:
                    trial_colors[e] = rng.choice(unused_valid)
                else:
                    trial_colors[e] = rng.choice(valid_cs)
            
            if valid_so_far and check_validity_full(trial_colors):
                return trial_colors
        return None

    # Try a few random starts
    starts = []
    for s in range(5):
        if time.time() > deadline - 10.0:
            break
        sc = generate_random_valid_start(s)
        if sc:
            starts.append(sc)
    
    if not starts:
        # Fallback to hierarchical
        def get_seed_colors(seed=0):
            colors = [0] * NUM_EDGES
            color_id = 0
            rng = random.Random(seed)
            def process_partition(vertices):
                nonlocal color_id
                if len(vertices) <= 1:
                    return
                mid = len(vertices) // 2
                left = vertices[:mid]
                right = vertices[mid:]
                for u in left:
                    for v in right:
                        idx = edge_idx[u][v]
                        colors[idx] = color_id
                color_id += 1
                process_partition(left)
                process_partition(right)
            process_partition(list(range(N)))
            return colors
        starts.append(get_seed_colors(0))

    # Run SA on each start
    for start_colors in starts:
        if time.time() > deadline - 5.0:
            break
        
        current_colors = start_colors.copy()
        current_distinct = count_colors(current_colors)
        
        if current_distinct > best_score:
            best_score = current_distinct
            best_colors = current_colors.copy()
        
        freq = [0] * MAX_COLORS
        for c in current_colors:
            freq[c] += 1
            
        rng = random.Random(123)
        T = 10.0
        T_min = 0.5
        cooling = 0.9999
        no_improve = 0
        max_no_improve = 20000
        
        while time.time() < deadline - 2.0:
            # Pick a random edge
            e = rng.randint(0, NUM_EDGES - 1)
            old_color = current_colors[e]
            
            # Get valid colors for this edge
            valid_cs = get_valid_colors_for_edge(e, current_colors)
            
            # Remove old color from valid list if it's there (we want a change)
            if old_color in valid_cs:
                valid_cs.remove(old_color)
            
            if not valid_cs:
                no_improve += 1
                continue
            
            # Choose new color
            # Bias towards unused colors
            unused_valid = [c for c in valid_cs if freq[c] == 0]
            if unused_valid and rng.random() < 0.8:
                new_color = rng.choice(unused_valid)
            else:
                new_color = rng.choice(valid_cs)
            
            # Apply move
            current_colors[e] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            
            new_distinct = current_distinct
            if freq[old_color] == 0:
                new_distinct -= 1
            if freq[new_color] == 1:
                new_distinct += 1
            
            # Acceptance
            if new_distinct >= current_distinct:
                accept = True
            else:
                delta = new_distinct - current_distinct
                accept = rng.random() < np.exp(delta / T)
            
            if accept:
                current_distinct = new_distinct
                if new_distinct > best_score:
                    best_score = new_distinct
                    best_colors = current_colors.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Reject
                current_colors[e] = old_color
                freq[old_color] += 1
                freq[new_color] -= 1
                no_improve += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
                
            if no_improve > max_no_improve:
                no_improve = 0
                T = 10.0
                # Perturb
                for _ in range(3):
                    pe = rng.randint(0, NUM_EDGES - 1)
                    vcs = get_valid_colors_for_edge(pe, current_colors)
                    if vcs:
                        old_c = current_colors[pe]
                        nc = rng.choice(vcs)
                        current_colors[pe] = nc
                        freq[old_c] -= 1
                        freq[nc] += 1
                        # Recalculate distinct roughly
                        # Just let SA handle it

    if best_colors is None:
        # Emergency fallback
        best_colors = [0] * NUM_EDGES
        best_score = 1

    return best_colors
