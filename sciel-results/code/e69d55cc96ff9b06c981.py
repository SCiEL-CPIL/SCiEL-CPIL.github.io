"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe under 90s limit
    N = 512
    n = N

    # Precompute adjacency: adj[i] is a set of j such that lcm(i+1, j+1) <= N
    # Using lists of ints for speed
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            g = gcd(a, b)
            if a * b // g <= N:
                adj[i].add(j)
                adj[j].add(i)
    
    # Convert to sorted lists for faster iteration
    adj_lists = [list(s) for s in adj]
    degree = [len(adj_lists[i]) for i in range(n)]

    # Initial lower bound: divisors of 512 (size 9: 1,2,4,8,16,32,64,128,256,512)
    # Actually divisors of 512 = 2^9: 1,2,4,8,16,32,64,128,256,512 -> 10 divisors
    # Better: find a good initial clique
    # Divisors of 360: 1,2,3,4,5,6,8,9,10,12,15,18,20,24,30,36,40,45,60,72,90,120,180,360 -> 24 elements
    # Check: lcm of any two divisors of 360 divides 360, so <= 360 <= 512. Valid!
    div360 = [i for i in range(1, 513) if 360 % i == 0]
    initial_clique = set(d - 1 for d in div360)  # 0-indexed
    initial_size = len(initial_clique)

    # Also try divisors of 480, 432, etc.
    best_lb_size = initial_size
    best_lb_set = initial_clique

    for M in [480, 432, 360, 240, 180, 144, 120, 96, 72, 60, 48, 36, 24, 12, 8, 6, 4, 3, 2, 1]:
        divs = [i for i in range(1, 513) if M % i == 0]
        # All divisors of M form a valid clique since lcm(a,b) | M
        if len(divs) > best_lb_size:
            best_lb_size = len(divs)
            best_lb_set = set(d - 1 for d in divs)

    best_size = best_lb_size
    best_set = set(best_lb_set)

    # Greedy coloring for upper bound in branch-and-bound
    # Colors the subgraph induced by P (candidate vertices)
    # Returns color classes; max color = upper bound on clique size
    def greedy_color(P, adj):
        """Color the graph induced by P. Returns list of color classes and max color."""
        P = list(P)
        colors = {}
        color_classes = []
        
        for v in P:
            # Find the smallest color not used by neighbors of v in P
            used_colors = set()
            for u in adj[v]:
                if u in colors:
                    used_colors.add(colors[u])
            
            c = 0
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(v)
        
        return color_classes

    # Branch and bound for maximum clique
    # P: set of candidate vertices (0-indexed)
    # C: current clique (set of 0-indexed vertices)
    # best_size: current best found
    def bnb(P, C, best_size_ref):
        """Branch and bound. Modifies best_size_ref in place (list of [size, set])."""
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if not P:
            if len(C) > best_size:
                best_size = len(C)
                best_set = set(C)
            return
        
        # Upper bound: color the subgraph induced by P
        color_classes = greedy_color(P, adj)
        max_color = len(color_classes)
        
        if len(C) + max_color <= best_size:
            return
        
        # Order P by color (highest color first for better pruning)
        # Vertices with higher color are more "expensive" to color, so try them first
        ordered_P = []
        for c in range(max_color - 1, -1, -1):
            for v in color_classes[c]:
                ordered_P.append((v, c + 1))
        
        for v, color_of_v in ordered_P:
            if time.time() > deadline:
                return
            
            # Upper bound: len(C) + color_of_v (since v has this color, max clique in this subgraph <= color_of_v)
            if len(C) + color_of_v <= best_size:
                return
            
            # Include v in clique
            new_C = C | {v}
            new_P = P & adj[v]  # Only vertices adjacent to v remain
            new_P.discard(v)
            
            bnb(new_P, new_C, best_size_ref)
            
            # Exclude v
            P.discard(v)
            
            # After removing v, recompute upper bound
            if not P:
                break
            color_classes2 = greedy_color(P, adj)
            if len(C) + len(color_classes2) <= best_size:
                return
    
    # Run branch and bound from different starting points
    # Start with the full graph
    full_P = set(range(n))
    empty_C = set()
    best_size_ref = [best_size, best_set]
    
    # Use a smarter initial ordering: start with high-degree vertices
    # For B&B, we want to find a large clique quickly
    # Let's try multiple B&B runs with different vertex orderings
    
    # First, do a greedy clique to get a good lower bound
    def greedy_clique(order):
        clique = set()
        mask = 0
        for v in order:
            if all(v in adj[u] for u in clique):
                clique.add(v)
        return clique
    
    # Try greedy with different orderings to improve lower bound
    order_desc = sorted(range(n), key=lambda x: -degree[x])
    gclique = greedy_clique(order_desc)
    if len(gclique) > best_size:
        best_size = len(gclique)
        best_set = set(gclique)
    
    order_asc = sorted(range(n), key=lambda x: degree[x])
    gclique2 = greedy_clique(order_asc)
    if len(gclique2) > best_size:
        best_size = len(gclique2)
        best_set = set(gclique2)
    
    # Now run B&B
    # To make B&B efficient, we should start with a good ordering
    # The standard approach: order vertices by degree, then run B&B
    
    # For the full problem, B&B on 512 vertices might be too slow
    # Let's use a reduced approach: start from a good clique and expand
    
    # Alternative: use the fact that the problem has structure
    # Numbers with many small prime factors are more "compatible"
    
    # Let's try B&B but with a time limit and multiple restarts
    # Key: use the greedy coloring bound effectively
    
    # A more practical approach: local search with strong moves
    # Combined with B&B on smaller subproblems
    
    # Let's implement a hybrid: 
    # 1. Get good initial solution via divisors + greedy
    # 2. Use local search (add/remove/swap) to improve
    # 3. Use B&B on the "hard" parts
    
    # Local search with 1-swap and 2-swap
    def local_search(s):
        """Improve clique s via greedy add, 1-swap, 2-swap."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: greedy add - add any vertex compatible with all in s
            while True:
                added = False
                for v in range(n):
                    if v not in s:
                        if all(v in adj[u] for u in s):
                            s.add(v)
                            added = True
                            break
                if not added:
                    break
            if added:
                improved = True
                if len(s) > best_size:
                    best_size = len(s)
                    best_set = set(s)
            
            # Phase 2: 1-swap - remove u, add v if it improves
            s_list = list(s)
            for u in s_list:
                if time.time() > deadline - 5:
                    break
                s_wo_u = s - {u}
                # Find best v to add
                best_v = -1
                best_deg = -1
                for v in range(n):
                    if v in s:
                        continue
                    if all(v in adj[x] for x in s_wo_u):
                        if degree[v] > best_deg:
                            best_deg = degree[v]
                            best_v = v
                if best_v >= 0:
                    s.remove(u)
                    s.add(best_v)
                    improved = True
                    if len(s) > best_size:
                        best_size = len(s)
                        best_set = set(s)
                    break
            
            # Phase 3: 2-swap - remove u1,u2, add v1,v2
            if time.time() < deadline - 10:
                s_list2 = list(s)
                found_2swap = False
                for i in range(min(len(s_list2), 30)):
                    if time.time() > deadline - 10:
                        break
                    u1 = s_list2[i]
                    for j in range(i + 1, min(len(s_list2), 30)):
                        if time.time() > deadline - 10:
                            break
                        u2 = s_list2[j]
                        s_wo = s - {u1, u2}
                        # Find two compatible vertices to add
                        candidates = [v for v in range(n) if v not in s and all(v in adj[x] for x in s_wo)]
                        if len(candidates) >= 2:
                            # Try pairs
                            for a in range(min(len(candidates), 20)):
                                if time.time() > deadline - 10:
                                    break
                                v1 = candidates[a]
                                for b in range(a + 1, min(len(candidates), 20)):
                                    if time.time() > deadline - 10:
                                        break
                                    v2 = candidates[b]
                                    if v2 in adj[v1]:
                                        s.remove(u1)
                                        s.remove(u2)
                                        s.add(v1)
                                        s.add(v2)
                                        found_2swap = True
                                        if len(s) > best_size:
                                            best_size = len(s)
                                            best_set = set(s)
                                        break
                                if found_2swap:
                                    break
                            if found_2swap:
                                break
                        if found_2swap:
                            break
                    if found_2swap:
                        improved = True
                        break
        
        return s
    
    # Run local search from multiple starting points
    starts = [set(best_lb_set)]
    
    # Add greedy cliques
    for _ in range(50):
        import random
        random.seed(_ * 12345)
        order = list(range(n))
        random.shuffle(order)
        gc = greedy_clique(order)
        starts.append(set(gc))
    
    # Also add some structured starts
    for M in [512, 480, 432, 360, 240, 180, 144, 120, 96, 72, 60, 48, 36, 24]:
        divs = [i for i in range(1, 513) if M % i == 0]
        starts.append(set(d - 1 for d in divs))
    
    # Run local search on each start
    for start in starts:
        if time.time() > deadline - 15:
            break
        s = set(start)
        local_search(s)
    
    # Final: try to expand the best solution further
    if best_set and time.time() < deadline - 10:
        s = set(best_set)
        # Multiple rounds of local search
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            local_search(s)
    
    result = sorted([i + 1 for i in best_set])
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
