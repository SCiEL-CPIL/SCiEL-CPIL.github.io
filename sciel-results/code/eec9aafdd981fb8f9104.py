import time
import sys
from random import Random

def construct():
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin before 120s kill

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        # Generate graph
        # The seed program uses 4*n*d iterations. We stick to this to ensure same graph structure if needed, 
        # but actually the task says "instance from bounded_degree_graph...". 
        # The seed program defines the instance generation. We must replicate it exactly to get the same graphs.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(edges):
        if not edges:
            return []
        
        n_edges = len(edges)
        # Map edges to indices for faster lookup
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Build adjacency list of the graph
        adj = [set() for _ in range(80)] # Max n is 80
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Compute conflict sets: conflict[i] is a set of indices j such that edge i and edge j cannot have the same color
        # Two edges conflict if they share a vertex or are adjacent (distance 2)
        conflict = [set() for _ in range(n_edges)]
        for i in range(n_edges):
            u1, v1 = edges[i]
            for j in range(i + 1, n_edges):
                u2, v2 = edges[j]
                # Check if they share a vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    # Check if they are adjacent (joined by an edge)
                    # This means one endpoint of edge i is connected to one endpoint of edge j
                    if v1 in adj[u2] or v1 in adj[v2] or u1 in adj[u2] or u1 in adj[v2]:
                        conflict[i].add(j)
                        conflict[j].add(i)

        # Heuristic: Order edges by descending size of conflict set (most constrained first)
        # We will permute the edges based on this order to improve search
        order = sorted(range(n_edges), key=lambda i: len(conflict[i]), reverse=True)
        # Create a mapping from new order index to original index
        # Let's just work with the original indices but pick the next variable dynamically.
        
        # Backtracking search
        # color_of_edge[i] = color assigned to edge i, -1 if unassigned
        color_of_edge = [-1] * n_edges
        # For each color, keep a set of edges assigned to it
        # We also need to know which edges are "blocked" for a given color
        # Instead of tracking blocked sets explicitly, we can check conflicts on the fly.
        # However, checking conflicts on the fly for every candidate color is expensive.
        # Let's use a bitset or a list of sets.
        
        # A better state representation:
        # available_colors[i] = set of colors that edge i can still take.
        # Initially, all colors 0..n_edges-1 are available.
        # But we don't know the number of colors in advance.
        
        # Alternative: Greedy with Backtracking on the number of colors.
        # Let's try a recursive solver that assigns colors one by one.
        # To speed up, we can try to find a solution with a fixed number of colors K, and increment K.
        # Or just let the solver assign the smallest available color, and if it fails, backtrack.
        
        # Let's implement a standard backtracking with forward checking.
        # State:
        # assigned: dict edge_idx -> color
        # unassigned: list of edge_idx
        # color_usage: dict color -> set of edge_idx assigned to it
        
        # To make it efficient, we'll pick the edge with the fewest available colors dynamically.
        
        best_solution = None
        current_solution = [-1] * n_edges
        num_colors_used = 0
        
        # Precompute conflict masks as bitsets for fast intersection if needed? 
        # Python sets are fine for n=40-80.
        
        def count_available_colors(edge_idx, current_assignment, color_to_edges):
            # Returns the set of colors that edge_idx can take
            # A color c is invalid if any edge in color_to_edges[c] conflicts with edge_idx
            available = set(range(num_colors_used + 1)) # We allow using existing colors or a new one (num_colors_used)
            # Actually, we can just check against existing colors. If none work, we use a new one.
            # But we want to minimize colors. 
            # Standard approach: try existing colors first. If none, try new color.
            # If we backtrack, we might need to try other existing colors.
            
            valid_colors = []
            # Check existing colors
            for c in range(num_colors_used):
                # Check if any edge in color c conflicts with edge_idx
                can_use = True
                for e_idx in color_to_edges[c]:
                    if e_idx in conflict[edge_idx]:
                        can_use = False
                        break
                if can_use:
                    valid_colors.append(c)
            
            # If no existing color works, we can always use a new color (num_colors_used)
            # So available is valid_colors + [num_colors_used]
            return valid_colors

        # To optimize, we can precompute for each edge, which other edges conflict.
        # We maintain color_to_edges as a list of lists.
        
        # Let's use a more efficient backtracking structure.
        # We will sort edges by a static heuristic first to help the initial greedy part.
        
        # Dynamic variable ordering:
        # At each step, pick the unassigned edge with the fewest valid colors.
        
        def backtrack(assigned_count, current_assignment, color_to_edges):
            nonlocal best_solution, num_colors_used, current_solution
            
            if time.time() - start_time > TIME_LIMIT:
                return False
            
            if assigned_count == n_edges:
                # Found a solution
                if best_solution is None or num_colors_used < best_solution[0]:
                    best_solution = (num_colors_used, current_assignment[:])
                    # We can try to find a better one (fewer colors), but for now we record it.
                    # To find optimal, we would need to continue searching.
                    # Given time limits, finding a good one is often enough.
                    # However, we can prune: if num_colors_used is already >= best known, we can stop this branch?
                    # No, we want to minimize. If we found a solution with K colors, we can prune branches that have already used K colors and still have unassigned edges?
                    # Actually, if we are using a new color, num_colors_used increases.
                    # If num_colors_used >= best_solution[0], we can prune.
                return True
            
            # Find unassigned edge with fewest available colors
            best_edge = -1
            best_valid_colors = None
            min_len = float('inf')
            
            for i in range(n_edges):
                if current_assignment[i] == -1:
                    valid = []
                    # Check existing colors
                    for c in range(num_colors_used):
                        # Check conflicts
                        can_use = True
                        for e_idx in color_to_edges[c]:
                            if e_idx in conflict[i]:
                                can_use = False
                                break
                        if can_use:
                            valid.append(c)
                    
                    # Always can use new color
                    valid.append(num_colors_used)
                    
                    if len(valid) < min_len:
                        min_len = len(valid)
                        best_edge = i
                        best_valid_colors = valid
                        if min_len == 1:
                            break # Can't get better than 1
        
            if best_edge == -1:
                return False # Should not happen
            
            edge_idx = best_edge
            valid_colors = best_valid_colors
            
            # Order colors: try existing ones first? Or random?
            # Trying existing ones first helps keep color count low.
            # We already have them in valid_colors.
            
            for c in valid_colors:
                # Assign c to edge_idx
                current_assignment[edge_idx] = c
                
                if c == num_colors_used:
                    # New color
                    color_to_edges.append([edge_idx])
                    new_num_colors = num_colors_used + 1
                    # Pruning: if new_num_colors >= best known, skip
                    if best_solution is not None and new_num_colors >= best_solution[0]:
                        current_assignment[edge_idx] = -1
                        color_to_edges.pop()
                        continue
                    if backtrack(assigned_count + 1, current_assignment, color_to_edges):
                        # If we found a solution, we might want to continue to find a better one?
                        # For now, let's just return True to get a valid solution quickly.
                        # To find optimal, we would need to not return immediately or use branch and bound properly.
                        # Let's try to find a better solution by not returning immediately if we have time.
                        # But for safety, let's just return the first good one found in this branch.
                        # Actually, we should continue searching to see if we can find a solution with fewer colors.
                        # But if we just return, we get a valid solution.
                        # Let's stick to returning the first solution found in the DFS order, 
                        # but we ordered edges to fail fast.
                        return True
                else:
                    color_to_edges[c].append(edge_idx)
                    if backtrack(assigned_count + 1, current_assignment, color_to_edges):
                        return True
                    color_to_edges[c].remove(edge_idx)
                
                current_assignment[edge_idx] = -1
            
            return False

        # Initialize
        color_to_edges = []
        # Start backtracking
        # We need to handle the case where we might not find a solution within time.
        # If backtrack returns False due to time, we fall back to greedy.
        
        found = backtrack(0, current_solution, color_to_edges)
        
        if found and best_solution is not None:
            # best_solution contains (num_colors, assignment)
            # We need to map assignment back to the original edge order?
            # The assignment is indexed by the original edge index.
            # The output requires a list of colors in the sorted edge order.
            # current_solution[i] is the color for edges[i].
            # So we just return [current_solution[i] for i in range(n_edges)]
            # Wait, best_solution is a copy. We should use the current_solution if it was updated in the successful branch?
            # In the code above, when we find a solution, we update best_solution.
            # But current_solution is modified in place.
            # If we return True, the current_solution holds the valid assignment.
            # However, if we continue searching for better solutions, current_solution changes.
            # Since we return True immediately upon finding a solution, current_solution is valid.
            # But wait, if we want to find a *better* solution (fewer colors), we shouldn't return immediately.
            # Let's modify the logic: 
            # If we find a solution, we store it. We do NOT return True immediately if we have time to search for better.
            # But implementing full branch and bound is complex.
            # Given the small size, let's just do a greedy first, then try to improve with local search or limited backtracking.
            
            pass
        
        # Fallback: Greedy
        # This ensures we always return a valid solution.
        greedy_color = [-1] * n_edges
        color_to_edges_greedy = []
        
        # Sort edges by degree (conflict size) descending for better greedy result
        greedy_order = sorted(range(n_edges), key=lambda i: len(conflict[i]), reverse=True)
        
        for i in greedy_order:
            # Find smallest available color
            assigned_colors = set()
            for j in conflict[i]:
                if greedy_color[j] != -1:
                    assigned_colors.add(greedy_color[j])
            
            c = 0
            while c in assigned_colors:
                c += 1
            
            greedy_color[i] = c
            if c == len(color_to_edges_greedy):
                color_to_edges_greedy.append([i])
            else:
                color_to_edges_greedy[c].append(i)
        
        # Check if backtracking found a better solution
        if found and best_solution is not None:
            # best_solution[1] is the assignment
            # We need to verify it's valid? The backtracking ensures it.
            # We return the backtracking solution if it uses fewer colors than greedy
            bt_colors = best_solution[0]
            greedy_colors = len(color_to_edges_greedy)
            
            if bt_colors <= greedy_colors:
                # Map back to original order
                # best_solution[1] is already indexed by original edge index
                return best_solution[1]
            else:
                return greedy_color
        else:
            return greedy_color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        colors = solve_instance(edges)
        out.append(colors)
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 5:
             # If we are running out of time, we can break and return what we have?
             # But we must return 3 lists.
             # If we haven't finished, we might have partial results.
             # The greedy is fast, so we should be fine.
             pass

    return out

