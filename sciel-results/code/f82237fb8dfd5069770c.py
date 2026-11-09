# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _is_2_colorable_fast(edges, n, deadline=None):
    """
    Exact 2-colourability check using backtracking with unit propagation.
    Optimized for speed using NumPy for conflict detection where possible, 
    but core logic remains Python for clarity and control.
    Returns True if 2-colorable, False otherwise.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    if not edges:
        return True
        
    # Incidence list
    inc = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            inc[v].append(i)
            
    # Precompute edge vertex arrays for faster access
    edge_verts = [list(e) for e in edges]
    
    assign = [-1] * n
    uncolored = set(range(n))
    
    # Heuristic: Vertex with highest degree first
    # We'll pick dynamically in backtrack
    
    def propagate(assign, changed):
        stack = list(changed)
        visited = set(stack)
        while stack:
            v = stack.pop()
            cv = assign[v]
            if cv == -1:
                continue
            for ei in inc[v]:
                e = edge_verts[ei]
                # Check if e is monochromatic or close
                # We need to know counts of 0s, 1s, -1s
                c0 = 0
                c1 = 0
                c_un = 0
                forced_val = -1
                has_mixed = False
                for u in e:
                    cu = assign[u]
                    if cu == 0:
                        c0 += 1
                    elif cu == 1:
                        c1 += 1
                    else:
                        c_un += 1
                        if c0 > 0 or c1 > 0:
                            # Mixed assigned, so if all assigned are same, force the last one
                            # Wait, if c0>0 and c1>0, it's mixed, no constraint.
                            # If c0>0 and c1==0, all assigned are 0.
                            # If c1>0 and c0==0, all assigned are 1.
                            pass
                            
                # Simplified logic:
                if c0 > 0 and c1 > 0:
                    continue # Mixed, no constraint
                if c_un == 0:
                    if c0 == 5 or c1 == 5:
                        return False # Conflict: monochromatic edge
                    continue
                if c0 == 4 and c1 == 0:
                    # Force the unassigned one to be 1
                    for u in e:
                        if assign[u] == -1:
                            if u not in visited:
                                visited.add(u)
                                assign[u] = 1
                                stack.append(u)
                            break
                elif c1 == 4 and c0 == 0:
                    # Force the unassigned one to be 0
                    for u in e:
                        if assign[u] == -1:
                            if u not in visited:
                                visited.add(u)
                                assign[u] = 0
                                stack.append(u)
                            break
        return True

    def backtrack(assign, uncolored):
        if time.time() > deadline:
            return None
        if not uncolored:
            # Final check
            for e in edge_verts:
                c0 = sum(1 for u in e if assign[u] == 0)
                c1 = sum(1 for u in e if assign[u] == 1)
                if c0 == 0 or c1 == 0:
                    return None
            return assign
        
        # Pick vertex with most unassigned neighbors in edges? 
        # Or just highest degree.
        # To speed up, pick vertex involved in most edges with unassigned vertices.
        # Simplified: highest degree overall.
        v = max(uncolored, key=lambda x: len(inc[x]))
        new_uncolored = uncolored - {v}
        
        for val in (0, 1):
            assign[v] = val
            if propagate(assign, [v]):
                res = backtrack(assign, new_uncolored)
                if res is not None:
                    return res
            assign[v] = -1
        return None

    # Initial propagation? None.
    result = backtrack(assign, uncolored)
    return result is not None

def _paley_5_hypergraph(n):
    """
    Construct a 5-uniform hypergraph on Z_n (n prime) using quadratic residues.
    Edge is {a, a+1, a+2, a+3, a+4} if some condition holds?
    Actually, a common construction for non-2-colorable hypergraphs uses 
    finite fields. 
    Let's use the "Paley graph" analogue.
    For 5-uniform, we can define edges based on sums or differences being in a set.
    
    A simpler robust approach: 
    Use a random regular-ish hypergraph and then prune.
    But we want a deterministic start.
    
    Let's try a construction based on the Fano plane or similar small structures embedded?
    No, n is up to 26.
    
    Let's use a random construction with a fixed seed, then prune.
    The "strategy" change is: 
    1. Generate a dense random 5-uniform hypergraph on n=15 or 20.
    2. Verify it is non-2-colorable (likely yes for dense).
    3. Use Local Search (Simulated Annealing) to remove edges while maintaining non-2-colorability.
    
    This is structurally different from the bitmask set cover.
    """
    return None # Placeholder

def _local_search_minimize(edges, n, deadline):
    """
    Try to remove edges one by one (or in small batches) if the hypergraph remains non-2-colorable.
    Uses a fast checker.
    """
    current_edges = list(edges)
    current_n = n
    
    # Sort edges by some heuristic? 
    # Maybe edges that are "redundant"?
    # Just random removal is okay if we have time.
    
    # We need to be careful: removing an edge might make it 2-colorable.
    # So we check after each removal.
    
    # To speed up, we can try to remove multiple edges if we are confident,
    # but checking is expensive.
    
    # Let's do a greedy removal:
    # 1. Shuffle edges
    # 2. For each edge, try to remove it.
    # 3. If still non-2-colorable, keep it removed.
    # 4. If it becomes 2-colorable, put it back.
    
    # This is O(E * CheckTime).
    # If E=100 and CheckTime=0.01s, total 1s. Good.
    
    random.shuffle(current_edges)
    
    # We can also try to add edges if we are stuck? No, we want to minimize.
    
    # Optimization: If we fail to remove an edge, we can skip it for a while?
    # Or just proceed.
    
    # Let's implement a simple pass.
    # We can do multiple passes.
    
    # First, check if current is non-2-colorable.
    if _is_2_colorable_fast(current_edges, current_n, deadline - 1):
        return current_edges # It's already 2-colorable, can't fix by removing.
        
    for i in range(len(current_edges) - 1, -1, -1):
        if time.time() > deadline - 5:
            break
            
        # Try to remove edge i
        e_to_remove = current_edges[i]
        
        # Temporarily remove
        temp_edges = current_edges[:i] + current_edges[i+1:]
        
        # Check
        if not _is_2_colorable_fast(temp_edges, current_n, deadline - 5):
            # Still non-2-colorable, so keep it removed
            current_edges = temp_edges
        else:
            # Becomes 2-colorable, so we must keep it
            pass # current_edges unchanged
            
    return current_edges

def _generate_random_hypergraph(n, density, seed):
    random.seed(seed)
    all_edges = list(combinations(range(n), 5))
    random.shuffle(all_edges)
    # Pick a subset
    num_edges = int(density * len(all_edges))
    return [list(e) for e in all_edges[:num_edges]]

def construct():
    deadline = time.time() + 110  # Safety margin
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Strategy: 
    # 1. Try several n values (e.g., 14, 16, 18, 20).
    # 2. For each n, generate a dense random hypergraph.
    # 3. Verify it is non-2-colorable.
    # 4. If yes, run local search to minimize edges.
    # 5. Keep the best result.
    
    # We know m(5) <= 51. So we are looking for something <= 51.
    # A dense hypergraph on n=15 has C(15,5) = 3003 edges.
    # We can start with a subset, say 100 edges, and check.
    # If 100 random edges are non-2-colorable, we can prune.
    # If not, we need more edges or different n.
    
    # Let's try n=15 first.
    # C(15,5) = 3003.
    # If we pick 60 edges, is it likely non-2-colorable?
    # Probably not. The threshold is around 50-100 edges for small n?
    # Actually, for n=15, the number of edges needed to force non-2-colorability is likely higher than 51?
    # Wait, m(5) is the MINIMUM number of edges in ANY non-2-colorable 5-uniform hypergraph.
    # It is 51. So there EXISTS a hypergraph with 51 edges.
    # It might have n=10, 11, etc.
    # If we search random hypergraphs on n=15, we might find one with 51 edges?
    # Unlikely, because the structure is specific.
    # However, if we start with a KNOWN non-2-colorable core (if we had one), we could prune.
    # But we don't.
    
    # Alternative: 
    # Use the fact that K_10^5 (252 edges) is non-2-colorable? 
    # Actually, m(5)=51 implies there is a hypergraph with 51 edges.
    # If we can't find it by random search, we should fall back to a known construction.
    # But we don't have a hardcoded 51-edge one.
    
    # Let's try a "greedy construction" that builds up edges to ensure non-2-colorability.
    # Start with 0 edges. Add edges one by one.
    # After each addition, check if it becomes non-2-colorable.
    # Stop when it is.
    # This finds a MINIMAL (by construction order) non-2-colorable hypergraph.
    # It might not be the global minimum, but it's a valid strategy.
    # To minimize edges, we should add edges that "break" colorings efficiently.
    
    # This is essentially the set cover problem again, but in reverse (building up).
    # The current program does set cover on colorings.
    # My new strategy: 
    # 1. Start with a random ordering of all edges on a chosen n.
    # 2. Add edges one by one.
    # 3. Check 2-colorability after each addition.
    # 4. Stop when non-2-colorable.
    # 5. Then try to prune.
    
    # To make it better, we can try multiple random orderings.
    
    # Let's try n=11, 12, 13.
    # C(11,5) = 462.
    # C(12,5) = 792.
    # C(13,5) = 1287.
    
    # We will try to find a small one.
    
    for n in [11, 12, 13, 14]:
        if time.time() > deadline - 20:
            break
            
        # Try a few random seeds
        for seed in range(10):
            if time.time() > deadline - 10:
                break
                
            all_edges = list(combinations(range(n), 5))
            random.seed(seed)
            random.shuffle(all_edges)
            
            current_edges = []
            is_non_colorable = False
            
            # Add edges until non-2-colorable
            for e in all_edges:
                current_edges.append(list(e))
                # Check
                # To speed up, we can check less frequently? 
                # No, we need to stop as soon as it becomes non-2-colorable.
                # But checking every step is slow.
                # We can check every 5 steps? 
                # If we check every 5 steps, we might add extra edges.
                # Let's check every step for n<=12, every 2 for n=13.
                check_freq = 1 if n <= 12 else 2
                if len(current_edges) % check_freq == 0:
                    if not _is_2_colorable_fast(current_edges, n, deadline - 5):
                        is_non_colorable = True
                        break
                        # Once non-2-colorable, we can break the loop
                        # But we might have added too many? 
                        # We added 'e' and maybe others in the batch.
                        # We can try to remove the last few.
            
            if is_non_colorable:
                # Prune
                pruned = _local_search_minimize(current_edges, n, deadline - 5)
                if len(pruned) < best_count:
                    best_count = len(pruned)
                    best_edges = pruned
                    best_n = n
                    # If we get close to 51, we can stop?
                    if best_count <= 51:
                        # We found a candidate. Can we do better?
                        # Maybe try to prune more aggressively?
                        # The local search already did one pass.
                        # Let's do another pass with a different shuffle.
                        random.shuffle(best_edges)
                        best_edges = _local_search_minimize(best_edges, best_n, deadline - 5)
                        if len(best_edges) < best_count:
                            best_count = len(best_edges)
                            
    # Fallback: If we didn't find anything good, return a large known non-2-colorable one.
    # K_11^5 is 462 edges.
    if best_edges is None:
        n = 11
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
        # Try to prune this large one quickly?
        # It might take too long.
        # Just return it.
        # But we want a good score.
        # Let's try one quick prune pass.
        best_edges = _local_search_minimize(best_edges, n, deadline - 5)
        best_count = len(best_edges)
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
