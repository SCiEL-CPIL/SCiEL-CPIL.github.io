"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85
    n = 512
    # Build conflict graph: vertices 1..512 (0-indexed 0..511)
    # Edge between u,v if there exist a,b such that {u,v} is a 2-subset of a 4-AP
    # A 4-AP is {a, a+d, a+2d, a+3d}
    # Conflicts: for each 4-AP, all pairs within it are conflicting
    
    # Precompute conflicts as bitsets
    N = 512
    conflicts = [0] * N  # conflicts[i] is a bitmask of all j that conflict with i
    
    # For each 4-AP (a, a+d, a+2d, a+3d), add edges between all pairs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            ap = [a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d]
            for i in range(4):
                for j in range(i + 1, 4):
                    conflicts[ap[i]] |= (1 << ap[j])
                    conflicts[ap[j]] |= (1 << ap[i])
    
    all_mask = (1 << N) - 1
    
    # Greedy initial solution
    def greedy_forward():
        selected = set()
        for x in range(1, N + 1):
            idx = x - 1
            # Check if adding x creates a 4-AP with existing elements
            can_add = True
            for d in range(1, (x - 1) // 3 + 1):
                if all(x - j * d in selected for j in (1, 2, 3)):
                    can_add = False
                    break
            if can_add:
                selected.add(x)
        return selected
    
    def greedy_reverse():
        selected = set()
        for x in range(N, 0, -1):
            idx = x - 1
            can_add = True
            # Check if x is part of a 4-AP with 3 existing elements
            # x could be any position in the AP
            # If x = a: need a+d, a+2d, a+3d in selected
            for d in range(1, (N - x) // 3 + 1):
                if all(x + j * d in selected for j in (1, 2, 3)):
                    can_add = False
                    break
            # If x = a+d: need a, a+2d, a+3d in selected
            if can_add:
                for d in range(1, x // 1):
                    if x - d >= 1 and x + 2*d <= N and x + 3*d <= N:
                        if all(x + j*d in selected for j in (0, 2, 3)):
                            can_add = False
                            break
            # If x = a+2d: need a, a+d, a+3d in selected
            if can_add:
                for d in range(1, x // 2 + 1):
                    if x - 2*d >= 1 and x + d <= N:
                        if all(x + j*d in selected for j in (-2, -1, 1)):
                            can_add = False
                            break
            # If x = a+3d: need a, a+d, a+2d in selected
            if can_add:
                for d in range(1, x // 3 + 1):
                    if x - 3*d >= 1:
                        if all(x - j*d in selected for j in (0, 1, 2)):
                            can_add = False
                            break
            if can_add:
                selected.add(x)
        return selected
    
    # Base-3 construction: numbers with no digit 2 in base 3 (Cantor-like)
    def base3_construction():
        selected = set()
        for x in range(1, N + 1):
            # Check if x in base 3 has no digit 2
            n3 = x
            valid = True
            while n3 > 0:
                if n3 % 3 == 2:
                    valid = False
                    break
                n3 //= 3
            if valid:
                selected.add(x)
        return selected
    
    # Local search to improve
    def local_search(initial, max_iters=200000):
        selected = set(initial)
        current_size = len(selected)
        
        # Precompute for fast checking
        # For each element, store which other elements it conflicts with in terms of 4-AP completion
        # Actually, let's use a different approach: try adding elements, if conflict, try removing one of the conflicting
        
        def has_4ap_with(sel, x):
            """Check if adding x to sel creates a 4-AP"""
            for d in range(1, (x - 1) // 3 + 1):
                if all(x - j * d in sel for j in (1, 2, 3)):
                    return True
            # x could be other positions
            for d in range(1, x):
                if x - d >= 1 and x + 2 * d <= N and x + 3 * d <= N:
                    if all(x + j * d in sel for j in (0, 2, 3)):
                        return True
            for d in range(1, x // 2 + 1):
                if x - 2 * d >= 1 and x + d <= N:
                    if all(x + j * d in sel for j in (-2, -1, 1)):
                        return True
            for d in range(1, x // 3 + 1):
                if x - 3 * d >= 1:
                    if all(x - j * d in sel for j in (0, 1, 2)):
                        return True
            return False
        
        def try_add(x):
            if x not in selected and 1 <= x <= N:
                if not has_4ap_with(selected, x):
                    selected.add(x)
                    return True
            return False
        
        # Phase 1: Try adding all unselected elements
        for x in range(1, N + 1):
            if x not in selected:
                try_add(x)
        
        # Phase 2: Local search - try swaps
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in range(1, N + 1):
                if x in selected:
                    # Try removing x and adding something else
                    selected.discard(x)
                    # Try adding some other element
                    for y in range(1, N + 1):
                        if y not in selected and y != x:
                            if not has_4ap_with(selected, y):
                                selected.add(y)
                                if len(selected) > current_size:
                                    current_size = len(selected)
                                improved = True
                                break
                    if x not in selected:
                        # If we didn't improve, put x back
                        if not has_4ap_with(selected, x):
                            selected.add(x)
                        else:
                            # x was needed, try to keep it
                            selected.add(x)
        return selected
    
    # Multi-start local search
    best_set = None
    best_size = 0
    
    # Generate diverse initial solutions
    initial_solutions = [
        greedy_forward(),
        greedy_reverse(),
        base3_construction(),
    ]
    
    # Add some random perturbations
    rng = random.Random(42)
    for _ in range(5):
        base = greedy_forward()
        perturbed = set(base)
        # Remove a few random elements
        to_remove = list(perturbed)[:len(perturbed)//10]
        for x in to_remove:
            perturbed.discard(x)
        initial_solutions.append(perturbed)
    
    for init in initial_solutions:
        if time.time() > deadline - 10:
            break
        result = local_search(init)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Bitset-based branch and bound for Maximum Independent Set
    # Vertices are 0..511, edges from conflicts
    # We want max subset with no edges
    
    def branch_and_bound():
        """B&B for MIS using bitsets"""
        best = [best_size]
        best_vertices = [set(x - 1 for x in best_set) if best_set else set()]
        
        # Color-based upper bound: greedy coloring of remaining graph
        def greedy_color_bound(P):
            """Return upper bound on clique size in graph induced by P using greedy coloring"""
            # Greedy coloring: assign colors to vertices in some order
            # Upper bound = max color used
            uncolored = P
            color_count = 0
            while uncolored:
                color_count += 1
                # Find a maximal independent set in the remaining graph (this is a color class)
                # Greedy: pick vertices that don't conflict with each other
                available = uncolored
                color_class = 0
                while available:
                    # Pick vertex with max degree in available
                    lsb = available & -available
                    v = lsb.bit_length() - 1
                    color_class |= lsb
                    available &= ~lsb
                    # Remove neighbors of v from available
                    available &= ~conflicts[v]
                uncolored &= ~color_class
            return color_count
        
        def bnb(P, C, size):
            if not P:
                if size > best[0]:
                    best[0] = size
                    best_vertices[0] = set(C)
                return
            if time.time() > deadline - 3:
                return
            
            # Upper bound
            ub = size + greedy_color_bound(P)
            if ub <= best[0]:
                return
            
            # Pick vertex with max degree
            p_temp = P
            best_v = -1
            best_deg = -1
            while p_temp:
                lsb = p_temp & -p_temp
                v = lsb.bit_length() - 1
                deg = (P & conflicts[v]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                p_temp ^= lsb
            
            if best_v == -1:
                return
            
            # Include best_v
            C.append(best_v)
            new_P = P & ~(1 << best_v)
            new_P &= ~conflicts[best_v]
            bnb(new_P, C, size + 1)
            C.pop()
            
            if time.time() > deadline - 3:
                return
            
            # Exclude best_v
            new_P_excl = P & ~(1 << best_v)
            bnb(new_P_excl, C, size)
        
        # Start with the current best as initial solution
        # B&B on the full graph
        C = []
        bnb(all_mask, C, 0)
        
        if best[0] > best_size:
            best_size = best[0]
            best_set = set(v + 1 for v in best_vertices[0])
        
        return best_set, best_size
    
    # Run B&B if we have time
    if time.time() < deadline - 15:
        best_set, best_size = branch_and_bound()
    
    # Final local search polish
    if best_set and time.time() < deadline - 5:
        polished = local_search(best_set, max_iters=50000)
        if len(polished) > best_size:
            best_set = polished
            best_size = len(polished)
    
    result = sorted(best_set) if best_set else list(range(1, N + 1, 3))
    return result

# EVOLVE-BLOCK-END
