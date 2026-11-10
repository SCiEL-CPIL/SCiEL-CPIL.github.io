# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Symmetrize: if d is forbidden, -d is also effectively forbidden
    forbidden_sym = set()
    for d in forbidden:
        forbidden_sym.add(d)
        forbidden_sym.add((-d) % m)
    forbidden_sym.discard(0)
    
    # Precompute conflict mask
    # conflict[i] is a set of j such that (i-j) % m in forbidden_sym
    conflict = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            d = (i - j) % m
            if d in forbidden_sym:
                conflict[i].add(j)
    
    # Precompute adjacency lists for faster iteration
    adj = [list(conflict[i]) for i in range(m)]
    
    # Bitset representation for speed in branch and bound
    # Use Python integers as bitsets.
    # bit i is set if vertex i is in the set.
    
    # Precompute neighbor bitmasks
    neigh_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        neigh_mask[i] = mask
        
    all_mask = (1 << m) - 1
    non_adj_mask = [0] * m
    for i in range(m):
        # Non-neighbors in G are those not in neigh_mask[i] and not i.
        non_adj_mask[i] = all_mask ^ (1 << i) ^ neigh_mask[i]

    # Global variables for branch and bound
    best_set = []
    best_size = 0
    deadline = time.time() + 590.0  # Safety margin
    stop_flag = [False]
    
    def greedy_color_bound(candidates_mask):
        """
        Greedy coloring of G_comp restricted to candidates_mask.
        Returns number of colors.
        """
        # Extract vertices
        verts = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            i = lsb.bit_length() - 1
            verts.append(i)
            temp ^= lsb
        
        n = len(verts)
        if n == 0:
            return 0
            
        # Order vertices by degree in G_comp descending
        # deg_comp(u) = n - 1 - popcount(neigh_mask[u] & candidates_mask)
        
        # Calculate degrees
        degrees = []
        for u in verts:
            deg_g = (neigh_mask[u] & candidates_mask).bit_count()
            deg_comp = n - 1 - deg_g
            degrees.append((deg_comp, u))
        
        # Sort by degree descending
        degrees.sort(reverse=True)
        ordered_verts = [u for _, u in degrees]
        
        color_of = {}
        max_color = 0
        
        for u in ordered_verts:
            # Find colors used by neighbors in G_comp
            # Neighbors in G_comp are v in verts such that (non_adj_mask[u] >> v) & 1
            # Which means v is NOT in adj[u].
            
            used_colors = set()
            neigh_comp_mask = non_adj_mask[u] & candidates_mask & ~(1 << u)
            
            temp_mask = neigh_comp_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                if v in color_of:
                    used_colors.add(color_of[v])
                temp_mask ^= lsb
                
            # Find smallest color not in used_colors
            c = 0
            while c in used_colors:
                c += 1
            color_of[u] = c
            if c > max_color:
                max_color = c
            
        return max_color + 1

    def branch_and_bound(candidates_mask, current_size, current_set):
        """
        Recursive branch and bound for Maximum Independent Set.
        """
        nonlocal best_set, best_size, stop_flag
        
        if stop_flag[0]:
            return
            
        if time.time() > deadline:
            stop_flag[0] = True
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        # Upper bound
        bound = greedy_color_bound(candidates_mask)
        if current_size + bound <= best_size:
            return
            
        # Choose a vertex to branch on
        # Heuristic: Choose vertex with highest degree in G restricted to candidates
        temp = candidates_mask
        best_v = -1
        best_deg = -1
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (neigh_mask[v] & candidates_mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        new_mask = candidates_mask & ~neigh_mask[best_v] & ~(1 << best_v)
        current_set.append(best_v)
        branch_and_bound(new_mask, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_mask = candidates_mask & ~(1 << best_v)
        branch_and_bound(new_mask, current_size, current_set)

    # Initial upper bound using greedy coloring on all vertices
    initial_bound = greedy_color_bound((1 << m) - 1)
    
    # We can also get a good initial lower bound using greedy construction
    # Let's run more random greedy constructions to get a good start
    for _ in range(500):
        if time.time() > deadline:
            break
        order = list(range(m))
        random.shuffle(order)
        S = []
        S_set = set()
        for x in order:
            # Check if x conflicts with any in S_set
            ok = True
            for s in S_set:
                if x in conflict[s]:
                    ok = False
                    break
            if ok:
                S.append(x)
                S_set.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Now run branch and bound
    sys.setrecursionlimit(5000)
    
    # Run B&B
    branch_and_bound((1 << m) - 1, 0, [])
    
    # Local search / Simulated Annealing polish
    def polish(S, S_set):
        if not S:
            return S, S_set
        
        improved = True
        # Simple local search first
        while improved and time.time() < deadline:
            improved = False
            # Try to add any possible element
            for x in range(m):
                if x not in S_set:
                    ok = True
                    for s in S_set:
                        if x in conflict[s]:
                            ok = False
                            break
                    if ok:
                        S.append(x)
                        S_set.add(x)
                        improved = True
            # Try remove+add
            if not improved:
                for u in list(S):
                    S_copy = list(S)
                    S_set_copy = set(S_set)
                    S_copy.remove(u)
                    S_set_copy.remove(u)
                    for x in range(m):
                        if x not in S_set_copy:
                            ok = True
                            for s in S_set_copy:
                                if x in conflict[s]:
                                    ok = False
                                    break
                            if ok:
                                S_copy.append(x)
                                S_set_copy.add(x)
                                break # Only add one
                    if len(S_copy) > len(S):
                        S = S_copy
                        S_set = S_set_copy
                        improved = True
        
        # Simulated Annealing
        if time.time() < deadline - 10: # Leave time
            T = 1.0
            T_min = 0.01
            alpha = 0.95
            current_S = set(S)
            current_score = len(S)
            
            # Precompute neighbors for SA
            # We want to minimize H = -|S| + lambda * conflicts
            # Since we maintain valid sets, conflicts are 0.
            # So we just want to maximize |S|.
            # Moves: Add, Remove, Swap.
            
            while T > T_min and time.time() < deadline - 5:
                # Pick a random move
                if len(current_S) < m:
                    # Try Add
                    candidates = [x for x in range(m) if x not in current_S]
                    if candidates:
                        x = random.choice(candidates)
                        # Check if valid
                        if all(x not in conflict[s] for s in current_S):
                            new_S = current_S | {x}
                            new_score = len(new_S)
                            if new_score >= current_score or random.random() < np.exp((new_score - current_score) / T):
                                current_S = new_S
                                current_score = new_score
                                if current_score > best_size:
                                    best_size = current_score
                                    best_set = list(current_S)
                # Try Swap
                if len(current_S) > 0:
                    u = random.choice(list(current_S))
                    candidates = [x for x in range(m) if x not in current_S]
                    if candidates:
                        x = random.choice(candidates)
                        # Check if valid to remove u and add x
                        new_S = (current_S - {u}) | {x}
                        if all(x not in conflict[s] for s in new_S if s != x):
                             # Actually need to check x against all in new_S
                             valid = True
                             for s in new_S:
                                 if s == x: continue
                                 if x in conflict[s]:
                                     valid = False
                                     break
                             if valid:
                                 new_score = len(new_S)
                                 if new_score >= current_score or random.random() < np.exp((new_score - current_score) / T):
                                     current_S = new_S
                                     current_score = new_score
                                     if current_score > best_size:
                                         best_size = current_score
                                         best_set = list(current_S)
                
                T *= alpha
                
        # Final greedy add
        for x in range(m):
            if x not in current_S:
                if all(x not in conflict[s] for s in current_S):
                    current_S.add(x)
                    if len(current_S) > best_size:
                        best_size = len(current_S)
                        best_set = list(current_S)
                        
        return list(current_S), current_S

    if best_set:
        best_set, best_set_set = polish(best_set, set(best_set))
        
    if not best_set:
        best_set = [0]
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
