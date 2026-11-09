"""Strong edge colouring of bounded-degree graphs via greedy + backtracking + local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_MARGIN = 10  # seconds to finish before 150s wall-clock

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    def get_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Generate a random bounded-degree graph
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges), degree

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def precompute_conflicts(edges, adj):
        """
        Precompute the set of edges that conflict with each edge.
        Two edges conflict if they share a vertex or are joined by an edge (distance 2).
        """
        n_edges = len(edges)
        conflicts = [set() for _ in range(n_edges)]
        
        # Build edge lists per vertex for fast lookup
        edges_at_vertex = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            edges_at_vertex[u].append(i)
            edges_at_vertex[v].append(i)
        
        # For each edge, find conflicting edges
        for i, (u1, v1) in enumerate(edges):
            # Edges sharing vertex u1 or v1
            for e2_idx in edges_at_vertex[u1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
                    conflicts[e2_idx].add(i)
            for e2_idx in edges_at_vertex[v1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
                    conflicts[e2_idx].add(i)
            
            # Distance 2 edges (joined by an edge)
            # Check neighbors of u1 and v1
            for neighbor in adj[u1]:
                if neighbor == v1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx != i and (u1, v1) != edges[e2_idx] and (v1, u1) != edges[e2_idx]:
                        # Check if e2_idx doesn't share a vertex with e1
                        u2, v2 = edges[e2_idx]
                        if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                            conflicts[i].add(e2_idx)
                            conflicts[e2_idx].add(i)
            for neighbor in adj[v1]:
                if neighbor == u1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx != i and (u1, v1) != edges[e2_idx] and (v1, u1) != edges[e2_idx]:
                        u2, v2 = edges[e2_idx]
                        if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                            conflicts[i].add(e2_idx)
                            conflicts[e2_idx].add(i)
        
        return conflicts

    def greedy_coloring(edges, conflicts, order):
        """
        Greedy coloring using precomputed conflicts.
        order: list of edge indices in the order to process.
        """
        n_edges = len(edges)
        color = [-1] * n_edges
        color_classes = [set() for _ in range(n_edges)] # Max possible colors
        
        for e_idx in order:
            # Find taken colors
            taken = set()
            for c_idx in conflicts[e_idx]:
                if color[c_idx] != -1:
                    taken.add(color[c_idx])
            
            c = 0
            while c in taken:
                c += 1
            color[e_idx] = c
            color_classes[c].add(e_idx)
            
        return color, color_classes

    def local_search(edges, conflicts, color, color_classes, max_iter=10000):
        """
        Local search: try to move edges to lower colors and reduce total colors.
        Returns improved color and color_classes.
        """
        n_edges = len(edges)
        current_max_color = max(color) + 1 if color else 0
        rng = random.Random(42)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Process edges in random order
            order = list(range(n_edges))
            rng.shuffle(order)
            
            for e_idx in order:
                c_current = color[e_idx]
                
                # Try to move to a lower color
                for new_c in range(c_current):
                    # Check if e_idx conflicts with any edge in color_classes[new_c]
                    can_move = True
                    if color_classes[new_c]:
                        # Quick check: if any conflict of e_idx is in new_c's class
                        for c_idx in conflicts[e_idx]:
                            if c_idx in color_classes[new_c]:
                                can_move = False
                                break
                    
                    if can_move:
                        # Move e_idx to new_c
                        color_classes[c_current].remove(e_idx)
                        color_classes[new_c].add(e_idx)
                        color[e_idx] = new_c
                        improved = True
                        # Update max color if we removed the last edge of the highest color
                        if c_current == current_max_color - 1 and not color_classes[c_current]:
                            current_max_color -= 1
                        break
                
                # If at highest color, try to move to any lower color that frees up space
                if c_current == current_max_color - 1 and color_classes[c_current]:
                    for new_c in range(c_current):
                        can_move = True
                        if color_classes[new_c]:
                            for c_idx in conflicts[e_idx]:
                                if c_idx in color_classes[new_c]:
                                    can_move = False
                                    break
                        if can_move:
                            color_classes[c_current].remove(e_idx)
                            color_classes[new_c].add(e_idx)
                            color[e_idx] = new_c
                            improved = True
                            if not color_classes[c_current]:
                                current_max_color -= 1
                            break
            
            if not improved:
                break
        
        return color, color_classes, current_max_color

    def reduce_color_iterative(edges, conflicts, color, color_classes, target_max_color, time_limit=10):
        """
        Iteratively try to reduce colors to target_max_color.
        Uses local search with color constraint.
        Returns (success, new_color, new_color_classes) if successful.
        """
        n_edges = len(edges)
        start_time = time.time()
        
        # Work with a copy
        cur_color = list(color)
        cur_classes = [set(c) for c in color_classes]
        cur_max = max(cur_color) + 1 if cur_color else 0
        
        # First, try to eliminate colors >= target_max_color
        # by moving edges from those color classes to lower ones
        for _ in range(50):  # max iterations
            if time.time() - start_time > time_limit:
                break
            if cur_max <= target_max_color:
                break
            
            # Find edges in the highest color class
            highest = cur_max - 1
            if not cur_classes[highest]:
                cur_max -= 1
                continue
            
            # Try to move each edge in the highest class to a lower color
            moved_any = False
            for e_idx in list(cur_classes[highest]):
                for new_c in range(highest):
                    can_move = True
                    if cur_classes[new_c]:
                        for c_idx in conflicts[e_idx]:
                            if c_idx in cur_classes[new_c]:
                                can_move = False
                                break
                    
                    if can_move:
                        cur_classes[highest].remove(e_idx)
                        cur_classes[new_c].add(e_idx)
                        cur_color[e_idx] = new_c
                        moved_any = True
                        if not cur_classes[highest]:
                            cur_max -= 1
                        break
                
                if time.time() - start_time > time_limit:
                    break
            
            if not moved_any:
                # Try more aggressive: temporarily move edges to enable moves
                # For now, just break
                break
        
        if cur_max <= target_max_color:
            return True, cur_color, cur_classes
        return False, cur_color, cur_classes

    def optimize_instance(n, d, seed, time_budget):
        """
        Optimize coloring for a single instance within time budget.
        """
        edges, degree = get_instance(n, d, seed)
        adj = build_adjacency(edges)
        conflicts = precompute_conflicts(edges, adj)
        n_edges = len(edges)
        
        best_color_list = None
        best_max_color = float('inf')
        
        elapsed_start = time.time()
        
        # Define multiple orderings
        orderings = []
        
        # 1. Natural order
        orderings.append(list(range(n_edges)))
        
        # 2. Reverse natural
        orderings.append(list(reversed(range(n_edges))))
        
        # 3. Degree-based (high degree vertices first)
        edge_degrees = [degree[u] + degree[v] for u, v in edges]
        orderings.append(sorted(range(n_edges), key=lambda i: edge_degrees[i], reverse=True))
        
        # 4. Random shuffles
        for s in range(25):
            rng = random.Random(seed + 1000 + s)
            shuffled = list(range(n_edges))
            rng.shuffle(shuffled)
            orderings.append(shuffled)
        
        # 5. Most conflicts first
        conflict_counts = [len(conflicts[i]) for i in range(n_edges)]
        orderings.append(sorted(range(n_edges), key=lambda i: conflict_counts[i], reverse=True))
        
        # 6. Least conflicts first
        orderings.append(sorted(range(n_edges), key=lambda i: conflict_counts[i]))
        
        for ordering in orderings:
            if time.time() > deadline:
                break
            if time.time() - elapsed_start > time_budget:
                break
            
            # Greedy
            color, color_classes = greedy_coloring(edges, conflicts, ordering)
            current_max = max(color) + 1 if color else 0
            
            if current_max < best_max_color:
                best_max_color = current_max
                best_color_list = list(color)
            
            # Local Search
            remaining = time_budget - (time.time() - elapsed_start)
            max_iter = max(100, int(remaining * 500))
            
            color, color_classes, current_max = local_search(edges, conflicts, color, color_classes, max_iter=max_iter)
            
            if current_max < best_max_color:
                best_max_color = current_max
                best_color_list = list(color)
        
        # Try iterative color reduction
        if best_color_list is not None and time.time() < deadline:
            remaining = min(time_budget - (time.time() - elapsed_start), 15.0)
            if remaining > 3.0:
                # Build color classes from best_color_list
                best_classes = [set() for _ in range(best_max_color)]
                for i, c in enumerate(best_color_list):
                    best_classes[c].add(i)
                
                # Try reducing by 1, 2, 3 colors
                for reduction in range(1, 4):
                    if time.time() > deadline:
                        break
                    target_colors = best_max_color - reduction
                    if target_colors < 1:
                        break
                    
                    time_per_try = min(remaining * 0.3, 5.0)
                    success, new_color, new_classes = reduce_color_iterative(
                        edges, conflicts, best_color_list, best_classes, 
                        target_colors, time_limit=time_per_try
                    )
                    remaining = remaining - (time.time() - (time.time() - time_per_try))
                    
                    if success:
                        best_max_color = target_colors
                        best_color_list = new_color
                        best_classes = new_classes
                    else:
                        # If we can't reduce by `reduction`, try smaller
                        break
        
        # If no coloring found (should not happen), fallback to simple greedy
        if best_color_list is None:
            color, _ = greedy_coloring(edges, conflicts, list(range(n_edges)))
            best_color_list = list(color)
            best_max_color = max(best_color_list) + 1 if best_color_list else 0
            
        return best_color_list, best_max_color

    results = []
    
    instances = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    remaining_time = deadline - time.time()
    
    for idx, (n, d, seed) in enumerate(instances):
        if time.time() > deadline:
            # Emergency fallback
            edges, _ = get_instance(n, d, seed)
            adj = build_adjacency(edges)
            conflicts = precompute_conflicts(edges, adj)
            color, _ = greedy_coloring(edges, conflicts, list(range(len(edges))))
            results.append(color)
            continue
        
        # Allocate more time to harder instances
        if idx == 0:
            time_budget = min(remaining_time * 0.25, 20.0)
        elif idx == 1:
            time_budget = min(remaining_time * 0.3, 30.0)
        else:
            time_budget = min(remaining_time * 0.35, 40.0)
        
        coloring, _ = optimize_instance(n, d, seed, time_budget)
        results.append(coloring)
        remaining_time = deadline - time.time()
    
    return results

# EVOLVE-BLOCK-END
