import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s
    
    N = 256
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build adjacency/conflict sets
    adj = [set() for _ in range(N)]
    
    # Identify vertices with self-loops (2*v is a square)
    forbidden = set()
    
    for i in range(N):
        v = i + 1
        # Check self-loop
        if 2 * v in squares:
            forbidden.add(i)
        
        # Find neighbors
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                adj[i].add(j)
    
    # Candidates are vertices without self-loops
    candidates = [i for i in range(N) if i not in forbidden]
    
    # Precompute degrees for greedy ordering
    degrees = [len(adj[i]) for i in range(N)]
    
    # Modular residue classes: squares mod 8 are {0,1,4}
    # If a+b is a square, then a+b mod 8 ∈ {0,1,4}
    # This gives structure we can exploit for seeding
    def residue_class(v):
        """Return the residue class of v+1 mod 8"""
        return (v + 1) % 8
    
    # Group candidates by residue class
    residue_groups = {}
    for v in candidates:
        rc = residue_class(v)
        if rc not in residue_groups:
            residue_groups[rc] = []
        residue_groups[rc].append(v)
    
    best_size = 0
    best_set = frozenset()
    
    def greedy_solve(order=None):
        """
        Greedy construction. If order is None, use a randomized degree-based order.
        """
        if order is None:
            order = sorted(candidates, key=lambda v: (degrees[v], random.random()))
        
        current_set = set()
        
        for v in order:
            if not (adj[v] & current_set):
                current_set.add(v)
                
        return current_set

    def local_improve(s):
        """Try to add vertices that don't conflict."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            for v in candidates:
                if v in s:
                    continue
                if not (adj[v] & s):
                    s.add(v)
                    improved = True
        return s

    def local_search_12(s, max_iterations=200):
        """
        1-2 exchange local search:
        - Try to add vertices (add-only)
        - If stuck, try removing one vertex and adding up to two others
        """
        s = set(s)
        s = local_improve(s)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations and (time.time() - start_time < DEADLINE - 1):
            improved = False
            iter_count += 1
            
            # Try 1-2 exchanges: remove v from s, then try to add up to 2 new vertices
            vertices_to_try = list(s)
            random.shuffle(vertices_to_try)
            
            for v in vertices_to_try:
                s.remove(v)
                
                # Try to add vertices that were previously blocked by v
                # First, try to add any vertex not in s with no conflicts
                s_new = set(s)
                
                # Collect candidates that could be added
                addable = []
                for u in candidates:
                    if u in s_new:
                        continue
                    if not (adj[u] & s_new):
                        addable.append(u)
                
                if len(addable) >= 2:
                    # Try pairs of addable vertices
                    random.shuffle(addable)
                    for i in range(min(len(addable), 20)):
                        for j in range(i+1, min(len(addable), 20)):
                            u1, u2 = addable[i], addable[j]
                            if u1 == u2:
                                continue
                            # Check if u1 and u2 conflict with each other
                            if u2 in adj[u1]:
                                continue
                            s_test = set(s_new)
                            s_test.add(u1)
                            s_test.add(u2)
                            if len(s_test) > len(s) + 1:
                                s = s_test
                                improved = True
                                break
                        if improved:
                            break
                
                if improved:
                    break
                else:
                    # Restore v
                    s.add(v)
                    # Try to fill up again after restoring
                    s = local_improve(s)
            
            # Also try simple additions in case we missed some
            if not improved:
                s = local_improve(s)
        
        return s

    def milp_polish(s, time_limit=10.0):
        """
        Use MILP to find a maximum independent set starting from s as a warm start.
        Returns the improved set.
        """
        try:
            from scipy.optimize import linprog
            import numpy as np
        except ImportError:
            return s
        
        # Build the conflict matrix for candidates only
        n = len(candidates)
        cand_to_idx = {v: i for i, v in enumerate(candidates)}
        
        # Create adjacency matrix
        A = np.zeros((n, n), dtype=int)
        for i, v in enumerate(candidates):
            for u in adj[v]:
                if u in cand_to_idx:
                    j = cand_to_idx[u]
                    A[i][j] = 1
                    A[j][i] = 1
        
        # MILP formulation for maximum independent set:
        # Maximize sum(x_i)
        # Subject to: x_i + x_j <= 1 for all edges (i,j)
        #             x_i in {0, 1}
        
        # We'll use a heuristic approach: try to add vertices greedily using the current solution as a guide
        # Since scipy's linprog doesn't handle integer constraints directly, we'll use a different approach
        
        # Alternative: Use branch and bound manually or use a different solver
        # For now, let's try a more sophisticated local search
        
        s = set(s)
        s = local_improve(s)
        
        # Try multiple random restarts of the local search
        best = s
        best_size = len(s)
        
        # Time budget for this polish
        polish_start = time.time()
        max_restarts = 20
        
        for restart in range(max_restarts):
            if time.time() - polish_start > time_limit:
                break
            
            # Create a perturbed version of the current best
            perturbed = set(best)
            
            # Remove some random elements
            if perturbed:
                remove_count = max(1, len(perturbed) // 10)
                to_remove = random.sample(list(perturbed), min(remove_count, len(perturbed)))
                for v in to_remove:
                    perturbed.discard(v)
            
            # Improve the perturbed solution
            improved = local_improve(perturbed)
            if time.time() - polish_start < time_limit - 2:
                improved = local_search_12(improved, max_iterations=50)
            
            if len(improved) > best_size:
                best_size = len(improved)
                best = improved
        
        return best

    # Modular-residue-based seeding strategy
    def modular_seed():
        """
        Seed based on modular residue classes (mod 4).
        Squares mod 4 are {0, 1}.
        Residues 1 and 3 are internally safe (1+1=2, 3+3=2 mod 4).
        We prioritize these groups to build a strong initial seed.
        """
        # Group candidates by residue mod 4
        residue_groups_mod4 = {0: [], 1: [], 2: [], 3: []}
        for v in candidates:
            rc = (v + 1) % 4
            residue_groups_mod4[rc].append(v)
        
        orderings = []
        
        # Strategy: Prioritize R1 and R3 (safe internally), then R0, R2
        # Shuffle within groups for diversity
        for perm_seed in range(10):
            random.seed(perm_seed * 98765 + 54321)
            
            # Order of groups: R1, R3 are preferred. R0, R2 are conflict-heavy.
            # We can try different permutations of the group order, but always keep R1/R3 early.
            group_order = [1, 3, 0, 2]
            
            # Shuffle the order of the "less safe" groups or even all groups slightly
            # But keep R1 and R3 at the front to ensure they are considered first
            # Let's shuffle the positions of R0 and R2, and maybe swap R1/R3
            if perm_seed % 2 == 0:
                group_order = [1, 3, 0, 2]
            else:
                group_order = [3, 1, 2, 0]
            
            # Add some randomness to the group order of the tail
            if perm_seed % 3 == 0:
                group_order = [1, 3, 2, 0]
            elif perm_seed % 3 == 1:
                group_order = [3, 1, 0, 2]
            
            order = []
            for rc in group_order:
                group = residue_groups_mod4[rc][:]
                random.shuffle(group)
                order.extend(group)
            
            # Fill in any missing candidates (should be none if groups cover all)
            for v in candidates:
                if v not in order:
                    order.append(v)
            
            orderings.append(order)
        
        random.seed()  # Reset random seed
        return orderings

    # Iterated Local Search / Random Restart Greedy
    modular_orderings = modular_seed()
    mo_idx = 0
    
    while time.time() - start_time < DEADLINE:
        # Choose strategy
        r = random.random()
        if r < 0.3:
            # Strategy 1: Random order
            order = candidates[:]
            random.shuffle(order)
        elif r < 0.6:
            # Strategy 2: Degree-based order with random tie-breaking
            order = sorted(candidates, key=lambda v: (degrees[v], random.random()))
        else:
            # Strategy 3: Modular-residue-based ordering
            if mo_idx < len(modular_orderings):
                order = modular_orderings[mo_idx]
                mo_idx += 1
            else:
                # Regenerate modular orderings
                modular_orderings = modular_seed()
                mo_idx = 0
                order = modular_orderings[mo_idx]
                mo_idx += 1
        
        s = greedy_solve(order)
        s = local_improve(s)
        
        # Do 1-2 exchange local search if we have time
        if time.time() - start_time < DEADLINE - 3:
            s = local_search_12(s)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = frozenset(s)
            
    # Final polish: use MILP-inspired local search on the best solution found
    if time.time() - start_time < DEADLINE - 5 and best_size > 0:
        polished = milp_polish(set(best_set), time_limit=min(10.0, DEADLINE - (time.time() - start_time) - 2))
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = frozenset(polished)
    
    # Convert 0-based indices back to 1-based values
    return sorted([v + 1 for v in best_set])

# EVOLVE-BLOCK-END
