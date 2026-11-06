import time
import random
from itertools import combinations

_START_TIME = None

def _deadline_ok(deadline, margin=3.0):
    return time.time() < deadline - margin

def _is_non_2colorable(n, edges):
    """
    Check if the hypergraph (n, edges) has NO proper 2-coloring.
    Proper 2-coloring: every edge has both colors (not all 0, not all 1).
    This is UNSAT for CSP: for every edge, not(all 0) and not(all 1).
    Use backtracking with unit propagation.
    """
    if not edges:
        return False  # trivially 2-colorable

    # Preprocess edges as tuples of sorted vertices
    edge_list = [tuple(sorted(e)) for e in edges]
    # Remove duplicates
    edge_set = set(edge_list)
    edge_list = list(edge_set)
    m = len(edge_list)
    if m == 0:
        return False

    # For each vertex, list of edges containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_list):
        for v in e:
            vertex_edges[v].append(i)

    # State: colors array, -1 unassigned, 0, 1 assigned
    # Unit propagation: if an edge has 4 vertices of color c, the 5th must be 1-c.
    # If an edge has 5 vertices of color c, contradiction.

    def propagate(colors, queue):
        # queue: list of edge indices to check
        changed = True
        qi = 0
        while qi < len(queue) or changed:
            changed = False
            while qi < len(queue):
                ei = queue[qi]
                qi += 1
                e = edge_list[ei]
                zeros = 0
                ones = 0
                unassigned = -1
                unassigned_idx = -1
                for idx, v in enumerate(e):
                    c = colors[v]
                    if c == 0:
                        zeros += 1
                    elif c == 1:
                        ones += 1
                    else:
                        unassigned = v
                        unassigned_idx = idx
                        if zeros > 3 or ones > 3:
                            # Can't have multiple unassigned if already 4+ of one color? 
                            # Actually if 4 zeros and 1 one, no unassigned. 
                            # If 3 zeros and 1 one, 1 unassigned.
                            # If 4 zeros and 0 ones, 1 unassigned.
                            pass
                # Count properly
                zeros = sum(1 for v in e if colors[v] == 0)
                ones = sum(1 for v in e if colors[v] == 1)
                if zeros == 5 or ones == 5:
                    return False
                if zeros == 4:
                    for v in e:
                        if colors[v] == -1:
                            if colors[v] != 1:
                                colors[v] = 1
                                changed = True
                                for nei_e in vertex_edges[v]:
                                    queue.append(nei_e)
                            break
                elif ones == 4:
                    for v in e:
                        if colors[v] == -1:
                            if colors[v] != 0:
                                colors[v] = 0
                                changed = True
                                for nei_e in vertex_edges[v]:
                                    queue.append(nei_e)
                            break
            # If no queue items left but changed was true, we need to re-scan?
            # The queue approach with appending neighbors should handle it.
            # But we need to ensure we don't miss edges.
            # Actually, the standard approach: 
            # While queue not empty: pop edge. Check. If propagation happened, add affected edges.
            # If queue empty and no changes, done.
            # But if changed is true and queue is empty, it means the last propagation didn't add to queue? 
            # No, if we assign a vertex, we add its edges to queue. So queue won't be empty if changed.
            # Unless the assigned vertex had no edges? Impossible in hypergraph with this edge.
            # So this loop should work.
            
            # Wait, the inner while loop exits when qi >= len(queue).
            # If changed was True, but no new items were added to queue in the last iteration, 
            # then we exit the outer loop. Is that correct?
            # Yes, because if no new items added, no further propagation is triggered by the last change.
            # But what if a change happened in an earlier iteration, and it added items, 
            # but those items were processed and didn't add more? Then queue is empty.
            # So the logic holds.
            
            # However, there is a subtle bug: if an edge triggers propagation, we add its neighbors.
            # But we also need to check if the edge itself becomes monochromatic? 
            # We check zeros==5 or ones==5 at the start of processing the edge.
            # If we assign the last vertex to make it 5 of one color, we would have detected zeros==5 or ones==5?
            # No, we detect 4 and then assign the 5th.
            # After assignment, the edge becomes 5 of one color.
            # But we don't re-check the same edge immediately in this iteration.
            # However, we added the edge's neighbors to the queue. 
            # Does the edge itself need to be re-checked?
            # If we assigned v to 1 because zeros==4, then now ones might be 1 (if there was 1 one) or 0 (if there were 0 ones? No, 4 zeros + 1 unassigned. If we assign 1, we have 4 zeros, 1 one. Not 5 ones.)
            # Wait, if zeros==4, the 5th was unassigned. We assign it to 1.
            # Result: 4 zeros, 1 one. Valid.
            # If ones==4, assign 0. Result: 4 ones, 1 zero. Valid.
            # So no contradiction from this edge directly.
            # But what if the vertex v was already assigned? 
            # We only act if colors[v] == -1.
            # So no contradiction here.
            # Contradiction only if we try to assign a value different from current? 
            # We check `if colors[v] != 1` before assigning 1. But if colors[v] was -1, it's fine.
            # What if colors[v] was 0? Then we wouldn't enter the `if colors[v] == -1` block.
            # So we don't overwrite.
            # But wait: if an edge has 4 zeros, and the 5th is 0? Then zeros==5. 
            # We check zeros==5 first. So if the 5th was 0, we'd return False.
            # So the logic is:
            # 1. If 5 same color -> Contradiction.
            # 2. If 4 same color and 1 unassigned -> Assign the unassigned to the opposite color.
            # 3. If 4 same color and 1 assigned (to same color)? That's case 1.
            # 4. If 4 same color and 1 assigned (to opposite color)? That's valid (4 vs 1).
            
            # So the propagation is sound.
            if not changed and qi >= len(queue):
                break
        return True

    # Backtracking
    # Order vertices by degree
    degrees = [len(vertex_edges[v]) for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -degrees[v])
    
    # Try to find a satisfying assignment.
    # If we find one, it IS 2-colorable -> Return False.
    # If we exhaust all possibilities, it is NOT 2-colorable -> Return True.
    
    # Optimization: If n is small, we can just iterate.
    # But n can be up to 26. 2^26 is too big.
    # We need smart backtracking.
    
    # Simple recursive backtracking with propagation.
    # To avoid deep recursion, use iterative or increase limit.
    import sys
    sys.setrecursionlimit(10000)
    
    best_assignment = None
    found_sat = False
    
    def backtrack(idx):
        nonlocal found_sat, best_assignment
        if found_sat:
            return
        
        if idx == n:
            found_sat = True
            return
            
        v = vertex_order[idx]
        
        # Try 0
        colors_backup = colors.copy()
        queue = []
        # Assign v=0
        if colors[v] == -1:
            colors[v] = 0
            for ei in vertex_edges[v]:
                queue.append(ei)
        if propagate(colors, queue):
            backtrack(idx + 1)
            if found_sat:
                return
        colors[:] = colors_backup
        
        # Try 1
        queue = []
        if colors[v] == -1:
            colors[v] = 1
            for ei in vertex_edges[v]:
                queue.append(ei)
        else:
            # If colors[v] is not -1, it's fixed by propagation?
            # No, in our backtrack, we only assign if -1.
            # But wait, if v was already assigned by a previous step's propagation?
            # Then we shouldn't branch on it. We should skip?
            # Or handle it?
            # In standard DPLL, if a variable is assigned, we don't branch.
            # But here, vertex_order is static.
            # If v is already assigned, we just recurse to next?
            # But we need to ensure consistency.
            # Actually, if v is assigned, we just proceed.
            # But we need to make sure the current state is valid.
            # propagate() already checked for contradictions.
            # So if colors[v] != -1, we just backtrack(idx+1).
            backtrack(idx + 1)
            if found_sat:
                return
            return # Only one branch for assigned var
        
        colors[:] = colors_backup

    # Init
    colors = [-1] * n
    # Heuristic: Start with a random assignment or all -1?
    # Let's try to bias.
    # Actually, just start with -1.
    
    # One optimization: If the hypergraph is sparse, it's likely 2-colorable.
    # If it's dense, likely not.
    
    backtrack(0)
    
    return not found_sat

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 145.0  # Safety margin
    
    # Strategy:
    # 1. Try to find a small non-2-colorable hypergraph.
    # 2. Start with known small ones? 
    #    K_9^5 has 126 edges.
    #    We want to reduce edges.
    # 3. Random search: Generate random hypergraphs, check if non-2-colorable, then minimize edges.
    #    Or start with non-2-colorable and remove edges.
    
    # Since 120s is a hard limit, and checking is expensive, 
    # we should focus on a few promising instances.
    
    # Candidate 1: K_9^5 (126 edges). We can try to remove edges.
    # Candidate 2: Random n=10,11,12... dense hypergraphs.
    
    # Let's try a greedy reduction on K_9^5 first.
    # But 126 edges is a lot. 
    # Maybe try n=10, 11 with specific structures?
    
    # The best known is 51.
    # Let's try to construct a candidate with ~50-60 edges.
    
    # Randomized local search / greedy minimization:
    # 1. Generate a random 5-uniform hypergraph on n vertices.
    # 2. Check if non-2-colorable.
    # 3. If yes, try to remove edges one by one (greedy) while maintaining non-2-colorability.
    # 4. If no, increase density or change n.
    
    # To save time, we will:
    # - Try n=9, 10, 11, 12.
    # - For each n, try a few random instances with high density.
    # - If found non-2-colorable, minimize.
    
    best_n = None
    best_edges = None
    best_count = 1000
    
    def try_minimize(n, edges, time_limit):
        """
        Given a non-2-colorable hypergraph, try to remove edges greedily.
        """
        current_edges = list(set(tuple(sorted(e)) for e in edges))
        current_n = n
        
        # Shuffle edges to break symmetry
        random.shuffle(current_edges)
        
        i = 0
        while i < len(current_edges) and time.time() < time_limit:
            e = current_edges[i]
            # Try removing e
            new_edges = current_edges[:i] + current_edges[i+1:]
            # Check if still non-2-colorable
            # This is expensive. 
            # Optimization: Only check if necessary? 
            # No, must check.
            
            # To speed up, we can use a faster checker for small reductions?
            # Or just call _is_non_2colorable.
            
            if _is_non_2colorable(current_n, new_edges):
                # Keep the removal
                current_edges = new_edges
                # Do not increment i, as the list shifted
            else:
                # Restore, move to next
                i += 1
            
            if len(current_edges) < best_count:
                # Update global best
                pass # We return local best
        
        return current_n, current_edges
    
    # Try n=9 first (K9)
    if time.time() < deadline - 10:
        k9_edges = [list(e) for e in combinations(range(9), 5)]
        # Check if non-2-colorable (should be)
        if _is_non_2colorable(9, k9_edges):
            t_limit = min(deadline - 5, time.time() + 20)
            n_min, edges_min = try_minimize(9, k9_edges, t_limit)
            if len(edges_min) < best_count:
                best_n = n_min
                best_edges = edges_min
                best_count = len(edges_min)
    
    # Try random n=10 to 12
    for n in range(10, 13):
        if time.time() > deadline - 20:
            break
            
        # Try a few random instances
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
                
            # Choose density
            p = 0.6
            # Total possible edges
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            
            # Select random subset
            # Start with dense, then reduce
            # Or start with full?
            # Let's start with random subset of size ~50-80
            num_edges = random.randint(40, 80)
            edges = random.sample(all_edges, num_edges)
            
            if _is_non_2colorable(n, edges):
                t_limit = min(deadline - 5, time.time() + 5)
                n_min, edges_min = try_minimize(n, edges, t_limit)
                if len(edges_min) < best_count:
                    best_n = n_min
                    best_edges = edges_min
                    best_count = len(edges_min)
                    
                if best_count <= 55:
                    break # Good enough
    
    # If we didn't find anything better than 126, return K9
    if best_edges is None:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
    
    # Final check
    # Ensure distinct, valid
    final_edges = []
    seen = set()
    for e in best_edges:
        t = tuple(sorted(e))
        if t not in seen:
            seen.add(t)
            final_edges.append(list(t))
    
    return {'n': best_n, 'edges': final_edges}
