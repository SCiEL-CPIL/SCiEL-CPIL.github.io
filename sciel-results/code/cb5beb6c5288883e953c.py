import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Precomputation
    # Map edge (i,j) to index
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    
    # For each edge, store the "other 3 edges" for each cycle it belongs to.
    # This allows fast constraint checking.
    # edge_to_cycle_others[e] is a list of tuples (e1, e2, e3)
    edge_to_cycle_others = [[] for _ in range(NUM_EDGES)]
    
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        # Get edge indices
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        # Cycle 1: v1-v2-v3-v4-v1
        # Edges: e12, e23, e34, e14
        # For e12, others are e23, e34, e14
        edge_to_cycle_others[e12].append((e23, e34, e14))
        edge_to_cycle_others[e23].append((e12, e34, e14))
        edge_to_cycle_others[e34].append((e12, e23, e14))
        edge_to_cycle_others[e14].append((e12, e23, e34))
        
        # Cycle 2: v1-v2-v4-v3-v1
        # Edges: e12, e24, e34, e13
        edge_to_cycle_others[e12].append((e24, e34, e13))
        edge_to_cycle_others[e24].append((e12, e34, e13))
        edge_to_cycle_others[e34].append((e12, e24, e13))
        edge_to_cycle_others[e13].append((e12, e24, e34))
        
        # Cycle 3: v1-v3-v2-v4-v1
        # Edges: e13, e23, e24, e14
        edge_to_cycle_others[e13].append((e23, e24, e14))
        edge_to_cycle_others[e23].append((e13, e24, e14))
        edge_to_cycle_others[e24].append((e13, e23, e14))
        edge_to_cycle_others[e14].append((e13, e23, e24))

    # Helper to check validity of a coloring
    def is_valid(coloring):
        for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
            e12 = edge_to_idx[(v1, v2)]
            e13 = edge_to_idx[(v1, v3)]
            e14 = edge_to_idx[(v1, v4)]
            e23 = edge_to_idx[(v2, v3)]
            e24 = edge_to_idx[(v2, v4)]
            e34 = edge_to_idx[(v3, v4)]
            
            # Cycle 1: v1-v2-v3-v4-v1
            c1, c2, c3, c4 = coloring[e12], coloring[e23], coloring[e34], coloring[e14]
            if len({c1, c2, c3, c4}) == 4:
                return False
            # Cycle 2: v1-v2-v4-v3-v1
            c1, c2, c3, c4 = coloring[e12], coloring[e24], coloring[e34], coloring[e13]
            if len({c1, c2, c3, c4}) == 4:
                return False
            # Cycle 3: v1-v3-v2-v4-v1
            c1, c2, c3, c4 = coloring[e13], coloring[e23], coloring[e24], coloring[e14]
            if len({c1, c2, c3, c4}) == 4:
                return False
        return True

    # Helper to get allowed colors for an edge given current colors
    def get_allowed_colors(e_idx, colors):
        allowed = set(range(MAX_COLOR))
        for o1, o2, o3 in edge_to_cycle_others[e_idx]:
            c1 = colors[o1]
            c2 = colors[o2]
            c3 = colors[o3]
            # If c1, c2, c3 are all distinct, the new color MUST be one of them
            if c1 != c2 and c1 != c3 and c2 != c3:
                # Intersect allowed with {c1, c2, c3}
                if len(allowed) <= 3:
                    if c1 not in allowed and c2 not in allowed and c3 not in allowed:
                        return set()
                    new_allowed = set()
                    if c1 in allowed: new_allowed.add(c1)
                    if c2 in allowed: new_allowed.add(c2)
                    if c3 in allowed: new_allowed.add(c3)
                    allowed = new_allowed
                else:
                    allowed.intersection_update({c1, c2, c3})
                if not allowed:
                    return set()
        return allowed

    # Fast check: is color c allowed for edge e_idx given current colors?
    def is_color_allowed(e_idx, c, colors):
        for o1, o2, o3 in edge_to_cycle_others[e_idx]:
            c1 = colors[o1]
            c2 = colors[o2]
            c3 = colors[o3]
            if c1 != c2 and c1 != c3 and c2 != c3:
                if c not in (c1, c2, c3):
                    return False
        return True

    # 2. Hierarchical Clustering Construction
    # We build a hierarchical clustering (dendrogram) of the 16 vertices.
    # Each internal node corresponds to a "cluster" and is assigned a unique color.
    # The edges connecting vertices in different child clusters of a node are assigned that node's color.
    
    def hierarchical_construct():
        # Start with 16 singletons
        clusters = [[i] for i in range(N)]
        color_map = {}  # edge -> color
        next_color = 0
        parent = {}  # cluster_id -> parent_cluster_id
        cluster_id = 0
        cluster_map = {i: i for i in range(N)} # vertex -> current cluster id
        
        # We need a way to identify clusters. Let's use a list of lists.
        # Merging: pick two random clusters, merge them.
        # Assign new color to all edges between the two merged clusters.
        
        active_clusters = list(range(N)) # ids 0..15 initially
        # We need to track which vertices are in which cluster.
        # Let's use a list of sets.
        cluster_vertices = [set([i]) for i in range(N)]
        
        # To ensure we use many colors, we merge in a way that creates many distinct colors.
        # Each merge creates one new color.
        # Total merges = N - 1 = 15.
        # So we will have 15 colors? Plus the initial 0?
        # Wait, the edges inside a singleton have no color.
        # When we merge A and B, we assign a new color to all edges between A and B.
        # Edges inside A and inside B keep their previous colors (or 0 if singleton).
        # This creates a valid coloring?
        # Let's check the 4-cycle constraint.
        # If we have a 4-cycle v1-v2-v3-v4-v1.
        # The edges are e12, e23, e34, e14.
        # In a hierarchical coloring, if two edges share a color, it means they were created in the same merge step.
        # If e12 and e34 have the same color, they were created when the clusters containing {v1,v2} and {v3,v4} were merged?
        # Actually, if e12 and e34 have the same color C, it means there was a merge step where v1,v2 were in one child cluster (A) and v3,v4 in another (B), and the merge assigned C to all edges between A and B.
        # Then e23 and e14 are also edges between A and B, so they also have color C.
        # So the 4-cycle would have colors {C, C, C, C}. This is valid (not 4 distinct).
        # What if e12 and e23 have the same color C?
        # Then v1,v2,v3 were involved in a merge where v1,v2 were in A and v3 in B? No, e12 is inside A or B?
        # If e12 has color C, it was created in a merge. Let's say merge(M1, M2) created C.
        # Then v1, v2 are in different child clusters of M1, M2? No, e12 is between M1 and M2.
        # So v1 in M1, v2 in M2.
        # e23 has color C. So v2, v3 are between M1 and M2.
        # Since v2 is in M2 (from e12), v3 must be in M1.
        # So v1 in M1, v3 in M1, v2 in M2.
        # Then e13 is inside M1. It has some color C13 (created earlier or 0).
        # e24? We need to check the cycle v1-v2-v3-v4-v1.
        # Edges: e12 (C), e23 (C), e34, e14.
        # e14 is between v1 (M1) and v4.
        # e34 is between v3 (M1) and v4.
        # If v4 is in M2, then e14 and e34 are between M1 and M2, so they have color C.
        # Cycle colors: C, C, C, C. Valid.
        # If v4 is in M1, then e14 and e34 are inside M1. They have colors C14, C34.
        # Cycle colors: C, C, C34, C14.
        # Are C34 and C14 distinct from C? Yes, C is the merge color, C34/C14 are internal.
        # Are C34 and C14 distinct from each other? Maybe.
        # If C34 != C14, we have {C, C, C34, C14}. 3 distinct colors. Valid.
        # If C34 == C14, we have {C, C, C34}. 2 distinct colors. Valid.
        # So it seems hierarchical clustering always produces a valid coloring!
        
        # Let's implement this.
        # We need to be careful with the "color 0" for singletons.
        # Singletons have no internal edges.
        
        # Random merge order to maximize variety?
        # Or a specific order?
        # Let's try random merges.
        
        for _ in range(N - 1):
            if time.time() > deadline:
                break
            # Pick two random active clusters
            i = random.randint(0, len(active_clusters) - 1)
            j = random.randint(0, len(active_clusters) - 1)
            while j == i:
                j = random.randint(0, len(active_clusters) - 1)
            
            c1 = active_clusters[i]
            c2 = active_clusters[j]
            
            # Assign new color to edges between c1 and c2
            new_color = next_color
            next_color += 1
            
            for v1 in cluster_vertices[c1]:
                for v2 in cluster_vertices[c2]:
                    if v1 < v2:
                        edge_to_idx[(v1, v2)] # just to ensure it exists
                    else:
                        edge_to_idx[(v2, v1)]
                    e_idx = edge_to_idx[(min(v1, v2), max(v1, v2))]
                    color_map[e_idx] = new_color
            
            # Merge c1 and c2 into c1
            cluster_vertices[c1].update(cluster_vertices[c2])
            active_clusters.remove(c2)
            active_clusters.remove(c1)
            active_clusters.append(c1)
            
        # Initialize color array
        colors = [0] * NUM_EDGES
        for e_idx, c in color_map.items():
            colors[e_idx] = c
            
        # Check validity (should be valid)
        if not is_valid(colors):
            return None
            
        return colors

    # 3. Local Search Improvement
    def local_search(initial_colors, time_limit):
        colors = initial_colors[:]
        freq = [0] * MAX_COLOR
        for c in colors:
            freq[c] += 1
        
        used_colors = set(c for c in colors if freq[c] > 0)
        best_colors = colors[:]
        best_score = len(used_colors)
        
        end_time = time.time() + time_limit
        stall_count = 0
        max_stall = 500
        
        # Track how many times each edge has been "tried" without improvement
        edge_tries = [0] * NUM_EDGES
        
        while time.time() < end_time:
            # Choose edge with bias toward less-tried edges
            # Use weighted random selection
            total_tries = sum(edge_tries)
            if total_tries > 0:
                weights = [1.0 / (1.0 + edge_tries[e]) for e in range(NUM_EDGES)]
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0
                e = 0
                for idx in range(NUM_EDGES):
                    cumulative += weights[idx]
                    if r <= cumulative:
                        e = idx
                        break
            else:
                e = random.randint(0, NUM_EDGES - 1)
            
            allowed = get_allowed_colors(e, colors)
            
            if not allowed:
                edge_tries[e] += 1
                continue
                
            current_color = colors[e]
            unused_in_allowed = [c for c in allowed if freq[c] == 0]
            
            if unused_in_allowed:
                new_color = random.choice(unused_in_allowed)
                # Apply move
                freq[current_color] -= 1
                if freq[current_color] == 0:
                    used_colors.discard(current_color)
                
                colors[e] = new_color
                freq[new_color] += 1
                used_colors.add(new_color)
                
                score = len(used_colors)
                if score > best_score:
                    best_score = score
                    best_colors = colors[:]
                    # Reset edge tries on improvement
                    edge_tries = [0] * NUM_EDGES
                stall_count = 0
            else:
                # Neutral or slightly worse move to escape local optima
                if stall_count < max_stall:
                    # Pick a random allowed color, preferably different
                    candidates = [c for c in allowed if c != current_color]
                    if candidates:
                        new_color = random.choice(candidates)
                    else:
                        new_color = current_color
                        
                    if new_color != current_color:
                        freq[current_color] -= 1
                        if freq[current_color] == 0:
                            used_colors.discard(current_color)
                        
                        colors[e] = new_color
                        freq[new_color] += 1
                        used_colors.add(new_color)
                        
                        score = len(used_colors)
                        if score > best_score:
                            best_score = score
                            best_colors = colors[:]
                            edge_tries = [0] * NUM_EDGES
                    stall_count += 1
                    edge_tries[e] += 1
                else:
                    # Restart from best_colors
                    colors = best_colors[:]
                    freq = [0] * MAX_COLOR
                    for c in colors:
                        freq[c] += 1
                    used_colors = set(c for c in colors if freq[c] > 0)
                    stall_count = 0
                    edge_tries = [0] * NUM_EDGES
        
        return best_colors, best_score

    # 4. Main Loop
    best_overall = [0] * NUM_EDGES
    best_overall_score = 0
    
    # Try hierarchical construction multiple times with different random seeds/orders
    # Then improve with local search
    
    num_restarts = 0
    while time.time() < deadline:
        # Get a base coloring from hierarchical construction
        base_colors = hierarchical_construct()
        if base_colors is None:
            continue
            
        # Check score
        used = set(base_colors)
        base_score = len(used)
        
        if base_score > best_overall_score:
            best_overall = base_colors[:]
            best_overall_score = base_score
            
        # Allocate remaining time to local search
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Use a fraction of remaining time for local search
        # Give more time to early restarts
        search_time = remaining * 0.4
        improved_colors, improved_score = local_search(base_colors, search_time)
        
        if improved_score > best_overall_score:
            best_overall = improved_colors
            best_overall_score = improved_score
            
        num_restarts += 1
        
        # If we have plenty of time, we can restart
        if time.time() > deadline - 2:
            break

    return best_overall
