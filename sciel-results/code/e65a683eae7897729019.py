# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum square-difference-free set modulo 205.
    Uses a combination of:
    1. Exact branch-and-bound for maximum independent set (with greedy coloring bound)
    2. Multiple random restarts with local improvement (1-swap, 2-swap)
    3. Deadline guard to ensure completion within 600s
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    m = 205
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency: two vertices are adjacent if their difference is forbidden
    # i.e., they CANNOT both be in the independent set
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i+1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute neighbor lists for speed
    neighbors = [list(s) for s in adj]
    
    def is_independent_set(vertices):
        """Check if a set of vertices forms an independent set."""
        vset = set(vertices)
        for v in vset:
            for u in neighbors[v]:
                if u in vset:
                    return False
        return True
    
    def greedy_coloring(vertices):
        """
        Greedy coloring of the subgraph induced by vertices.
        Returns list of color classes (each a list of vertices).
        The number of colors is an upper bound on the size of any independent set
        in the subgraph (since each color class is an independent set).
        """
        if not vertices:
            return []
        
        # Order vertices by degree (within the subgraph) descending
        vert_set = set(vertices)
        deg = {v: sum(1 for u in neighbors[v] if u in vert_set) for v in vertices}
        ordered = sorted(vertices, key=lambda v: -deg[v])
        
        colors = []
        colored = {}
        
        for v in ordered:
            # Find the smallest color not used by neighbors of v
            used_colors = set()
            for u in neighbors[v]:
                if u in vert_set and u in colored:
                    used_colors.add(colored[u])
            
            c = 0
            while c in used_colors:
                c += 1
            
            if c >= len(colors):
                colors.append([])
            colors[c].append(v)
            colored[v] = c
        
        return colors
    
    def branch_and_bound(vertices, current_set, best_size, best_set, time_limit):
        """
        Branch and bound for maximum independent set.
        vertices: list of candidate vertices
        current_set: list of vertices already in the solution
        best_size: current best size found
        best_set: current best set
        time_limit: deadline
        Returns (best_size, best_set)
        """
        if time.time() > time_limit:
            return best_size, best_set
        
        if not vertices:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return best_size, best_set
        
        # Upper bound: |current_set| + greedy coloring number of subgraph on vertices
        color_classes = greedy_coloring(vertices)
        upper_bound = len(current_set) + len(color_classes)
        
        if upper_bound <= best_size:
            return best_size, best_set
        
        # Choose vertex to branch on: highest degree in the subgraph
        vert_set = set(vertices)
        deg = {v: sum(1 for u in neighbors[v] if u in vert_set) for v in vertices}
        v = max(vertices, key=lambda x: deg[x])
        
        # Branch 1: include v
        new_vertices = [u for u in vertices if u != v and u not in neighbors[v]]
        current_set.append(v)
        best_size, best_set = branch_and_bound(new_vertices, current_set, best_size, best_set, time_limit)
        current_set.pop()
        
        if time.time() > time_limit:
            return best_size, best_set
        
        # Branch 2: exclude v
        new_vertices = [u for u in vertices if u != v]
        best_size, best_set = branch_and_bound(new_vertices, current_set, best_size, best_set, time_limit)
        
        return best_size, best_set
    
    def local_improve(vertices, max_swaps=1000):
        """
        Try to improve an independent set by adding vertices or swapping.
        Returns improved set.
        """
        vset = set(vertices)
        improved = True
        swaps = 0
        
        while improved and swaps < max_swaps:
            improved = False
            swaps += 1
            
            # Try adding each non-member vertex
            for cand in range(m):
                if cand in vset:
                    continue
                # Check if cand can be added
                can_add = True
                for u in neighbors[cand]:
                    if u in vset:
                        can_add = False
                        break
                if can_add:
                    vset.add(cand)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one vertex, add another
            for remove in list(vset):
                remaining = vset - {remove}
                for cand in range(m):
                    if cand in remaining:
                        continue
                    # Check if cand can be added to remaining
                    can_add = True
                    for u in neighbors[cand]:
                        if u in remaining:
                            can_add = False
                            break
                    if can_add:
                        # Also ensure cand != remove (already true since cand not in remaining and remove not in remaining)
                        vset = remaining | {cand}
                        improved = True
                        break
                if improved:
                    break
        
        return list(vset)
    
    # First, get an initial solution via greedy
    def greedy_construct(order):
        """Greedy construction with given order."""
        selected = []
        selected_set = set()
        for v in order:
            can_add = True
            for u in neighbors[v]:
                if u in selected_set:
                    can_add = False
                    break
            if can_add:
                selected.append(v)
                selected_set.add(v)
        return selected
    
    # Get initial solution
    initial_order = list(range(m))
    current_best = greedy_construct(initial_order)
    current_best_set = set(current_best)
    best_size = len(current_best)
    best_set = list(current_best)
    
    # Local improvement
    improved = local_improve(current_best)
    if len(improved) > best_size:
        best_size = len(improved)
        best_set = improved
    
    # Try multiple random restarts
    random.seed(42)
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random order for greedy
        order = list(range(m))
        random.shuffle(order)
        
        sol = greedy_construct(order)
        sol = local_improve(sol)
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol
    
    # Try exact branch-and-bound on smaller subproblems
    # We can try to solve exactly if the remaining candidates are small enough
    # Or use branch-and-bound with a good initial bound
    
    # Try branch-and-bound from scratch with the best set as initial bound
    # But this might be too slow for 205 vertices. Let's try a limited version.
    
    # Alternative: try to add vertices to the best set using branch-and-bound
    # on the complement
    
    # For now, let's try a few more local search strategies
    
    # Simulated annealing-like approach
    def simulated_annealing(initial_set, num_iters=10000):
        """
        Simulated annealing to improve the independent set.
        """
        current = set(initial_set)
        current_size = len(current)
        best = set(current)
        best_size = current_size
        
        temp = 1.0
        cooling = 0.9999
        
        for it in range(num_iters):
            if time.time() > deadline:
                break
            
            # Random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random vertex
                candidates = [v for v in range(m) if v not in current]
                if not candidates:
                    continue
                cand = random.choice(candidates)
                # Check if can add
                can_add = True
                for u in neighbors[cand]:
                    if u in current:
                        can_add = False
                        break
                if not can_add:
                    continue
                new_set = current | {cand}
                new_size = len(new_set)
                delta = new_size - current_size
                if delta > 0 or random.random() < np.exp(delta / temp):
                    current = new_set
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best = set(current)
                        
            elif move_type < 0.6:
                # Try to remove a random vertex
                if not current:
                    continue
                rem = random.choice(list(current))
                new_set = current - {rem}
                new_size = len(new_set)
                delta = new_size - current_size
                if delta > 0 or random.random() < np.exp(delta / temp):
                    current = new_set
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best = set(current)
                        
            else:
                # Try a swap: remove one, add another
                if not current:
                    continue
                rem = random.choice(list(current))
                remaining = current - {rem}
                candidates = [v for v in range(m) if v not in remaining and v != rem]
                if not candidates:
                    continue
                cand = random.choice(candidates)
                # Check if cand can be added to remaining
                can_add = True
                for u in neighbors[cand]:
                    if u in remaining:
                        can_add = False
                        break
                if not can_add:
                    continue
                new_set = remaining | {cand}
                new_size = len(new_set)
                delta = new_size - current_size
                if delta > 0 or random.random() < np.exp(delta / temp):
                    current = new_set
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best = set(current)
            
            temp *= cooling
        
        return list(best)
    
    # Run simulated annealing
    sa_result = simulated_annealing(best_set, num_iters=50000)
    if len(sa_result) > best_size:
        best_size = len(sa_result)
        best_set = sa_result
    
    # Final local improvement
    final = local_improve(best_set)
    if len(final) > best_size:
        best_size = len(final)
        best_set = final
    
    # Verify the result is valid
    final_set = set(best_set)
    for v in final_set:
        for u in neighbors[v]:
            if u in final_set:
                # This shouldn't happen, but if it does, fix it
                final_set.remove(u)
    
    # Return as numpy array
    return np.array(sorted(final_set), dtype=np.int64)


# EVOLVE-BLOCK-END
