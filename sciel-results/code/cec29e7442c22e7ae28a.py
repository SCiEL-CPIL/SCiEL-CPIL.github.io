import time
import random
from itertools import combinations
import numpy as np

def is_2_colorable_bitset(n, edges):
    """
    Exact 2-colorability check via bitset union.
    For n <= 20, 2^n fits in a Python int.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    if n > 20:
        # Fallback: shouldn't happen for n<=20 in this context, but safe
        return is_2_colorable_backtrack(n, edges)
    
    full_mask = (1 << (1 << n)) - 1
    covered = 0
    
    for e in edges:
        e_mask = 0
        for v in e:
            e_mask |= (1 << v)
        free_mask = ((1 << n) - 1) ^ e_mask
        
        sub = free_mask
        while True:
            m1 = sub
            m2 = sub | e_mask
            covered |= (1 << m1)
            covered |= (1 << m2)
            if sub == 0:
                break
            sub = (sub - 1) & free_mask
        
        if covered == full_mask:
            return False
            
    return True

def is_2_colorable_backtrack(n, edges):
    """
    Backtracking with unit propagation for larger n or as fallback.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    num_edges = len(edges)
    color = [-1] * n
    c0 = [0] * num_edges
    c1 = [0] * num_edges
    
    vertex_edges = [[] for _ in range(n)]
    edge_masks = []
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
            vertex_edges[v].append(i)
        edge_masks.append(mask)
        
    def propagate():
        changed = True
        while changed:
            changed = False
            for i in range(num_edges):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                if c0[i] == 4 and c1[i] == 0:
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            color[v] = 1
                            c1[i] += 1
                            for other_e in vertex_edges[v]:
                                c1[other_e] += 1
                            changed = True
                            if c1[i] == 5:
                                return False
                            break
                elif c1[i] == 4 and c0[i] == 0:
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            color[v] = 0
                            c0[i] += 1
                            for other_e in vertex_edges[v]:
                                c0[other_e] += 1
                            changed = True
                            if c0[i] == 5:
                                return False
                            break
        return True
        
    def backtrack():
        best_v = -1
        best_edge = -1
        best_unassigned = 6
        best_constraint = 0
        
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
            unassigned = 5 - c0[i] - c1[i]
            constraint = max(c0[i], c1[i])
            if unassigned < best_unassigned or (unassigned == best_unassigned and constraint > best_constraint):
                best_unassigned = unassigned
                best_constraint = constraint
                best_edge = i
                if unassigned == 0:
                    return False
            
        if best_edge == -1:
            return True
            
        candidates = []
        for v in range(n):
            if color[v] == -1 and (edge_masks[best_edge] & (1 << v)):
                candidates.append(v)
                
        if not candidates:
            return True
            
        for v in candidates:
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            color[v] = 0
            ok = True
            for e_idx in vertex_edges[v]:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    ok = False
                    break
            if ok and propagate():
                if backtrack():
                    return True
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
            
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            color[v] = 1
            ok = True
            for e_idx in vertex_edges[v]:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    ok = False
                    break
            if ok and propagate():
                if backtrack():
                    return True
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
            
        return False
        
    return backtrack()

def is_2_colorable(n, edges):
    """
    Dispatch to the fastest appropriate checker.
    """
    if n <= 20:
        return is_2_colorable_bitset(n, edges)
    else:
        return is_2_colorable_backtrack(n, edges)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit (or 150s if that's the real limit, but 115 is safe for 120)
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    # We will search for n in a range. The bitset method works up to n=20 efficiently.
    # For n > 20, we might still try, but it will be slower.
    # The known best construction is 51 edges.
    
    n_candidates = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # For each n, try to find a minimal non-2-colorable hypergraph
    for n in n_candidates:
        if time.time() > deadline - 5:
            break
            
        all_possible = list(combinations(range(n), 5))
        
        # Check if the complete hypergraph is non-2-colorable
        # If it is 2-colorable, we can skip this n entirely (since any subgraph of a 2-colorable graph is 2-colorable)
        # Wait, that's wrong. If the complete graph is 2-colorable, a subgraph might be non-2-colorable? No.
        # Property B is monotone decreasing with respect to edge deletion.
        # If H is 2-colorable, any subgraph H' is 2-colorable.
        # So if K_n^5 is 2-colorable, no subgraph is non-2-colorable.
        # If K_n^5 is NOT 2-colorable, some subgraphs might be.
        
        if is_2_colorable(n, [list(e) for e in all_possible]):
            continue
            
        # Start with the complete hypergraph
        current_edges = [list(e) for e in all_possible]
        current_m = len(current_edges)
        
        # Phase 1: Greedy Removal
        # We want to remove edges while keeping the property.
        # Randomized restarts to escape local optima.
        max_restarts = 50
        if n > 15:
            max_restarts = 10  # Slower checks
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Start from the complete graph again for each restart?
            # Or start from the best found so far?
            # Starting from complete is safer but slower to converge.
            # Let's start from the best known for this n if we have one, else complete.
            # Actually, let's just do a few restarts from complete for robustness.
            
            curr = current_edges[:]
            # Shuffle edges to randomize removal order
            random.shuffle(curr)
            
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for i in range(len(curr)):
                    if time.time() > deadline - 10:
                        break
                    # Try removing edge i
                    new_edges = curr[:i] + curr[i+1:]
                    if not is_2_colorable(n, new_edges):
                        curr = new_edges
                        improved = True
                        # Restart the loop with a new shuffle to avoid getting stuck in a specific order
                        random.shuffle(curr)
                        break
                        
            if len(curr) < best_m:
                best_m = len(curr)
                best_edges = curr[:]
                best_n = n
                # Update current_edges for next restarts of this n to start from a better point?
                # Maybe. But let's keep it simple.
                
        # Phase 2: Local Search / Simulated Annealing on the best found for this n
        if best_edges and best_n == n and time.time() < deadline - 5:
            curr_edges = best_edges[:]
            curr_set = set(tuple(e) for e in curr_edges)
            all_set = set(tuple(e) for e in all_possible)
            
            T = 5.0
            cooling = 0.99
            max_steps = 2000 if n <= 15 else 500
            
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                move_type = random.random()
                new_set = set(curr_set)
                
                if move_type < 0.6:
                    # Try to remove a random edge
                    if not curr_set:
                        continue
                    e_to_remove = random.choice(list(curr_set))
                    new_set.remove(e_to_remove)
                else:
                    # Try to add a random edge (to escape local minima where we got stuck)
                    available = list(all_set - curr_set)
                    if not available:
                        continue
                    e_to_add = random.choice(available)
                    new_set.add(e_to_add)
                
                # Check if new_set is non-2-colorable
                new_edges = [list(e) for e in new_set]
                if not is_2_colorable(n, new_edges):
                    # Accept the move
                    curr_set = new_set
                    if len(curr_set) < best_m:
                        best_m = len(curr_set)
                        best_edges = [list(e) for e in curr_set]
                        best_n = n
                else:
                    # Reject (or accept with probability exp(-delta/T) if we were doing SA on cost, 
                    # but here our "cost" is just validity. We only accept valid states.
                    # So this is essentially a random walk on valid states.)
                    pass
                    
                T *= cooling

    # Final check: if we didn't find anything better than a trivial case, return a known valid one.
    # The complete graph K_9^5 has 126 edges and is non-2-colorable.
    if best_edges is None:
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
