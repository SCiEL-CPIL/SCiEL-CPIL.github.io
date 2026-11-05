import time
import random

N = 40

def _validate(edges, cycles):
    """Validate the graph and cycles."""
    n = N
    edge_set = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))

    if len(cycles) != n - 2:
        return False

    for i, cycle in enumerate(cycles):
        L = i + 3
        if len(cycle) != L:
            return False
        if len(set(cycle)) != L:
            return False
        for j in range(L):
            u = cycle[j]
            v = cycle[(j + 1) % L]
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                return False
    return True

def _build_adj(edges):
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_cycle(adj, L, forbidden=None):
    """Find a simple cycle of length L in graph adj, avoiding forbidden edge if given."""
    if L < 3 or L > N:
        return None
    fe = None
    if forbidden is not None:
        u, v = forbidden
        if u > v:
            u, v = v, u
        fe = (u, v)
    
    # Try a few start vertices
    for start in [0, 1, 2, 3]:
        path = [start]
        visited = {start}
        
        def recurse(current, depth):
            if depth == L:
                if start in adj[current]:
                    if fe is not None and (min(current, start), max(current, start)) == fe:
                        return None
                    return list(path)
                return None
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    result = recurse(neighbor, depth + 1)
                    if result is not None:
                        return result
                    path.pop()
                    visited.discard(neighbor)
            return None
        
        result = recurse(start, 1)
        if result is not None:
            return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # --- Strategy: Ladder Graph with Optimized Chords ---
    # The ladder graph (inspiration 2) provides a robust base with 58 edges.
    # It naturally supports cycles of all lengths.
    # We will use this as the primary candidate and try to remove edges.
    
    # 1. Construct Ladder Graph
    # Vertices 0..19 (top), 20..39 (bottom)
    # Edges:
    # - Top path: (i, i+1) for i in 0..18
    # - Bottom path: (i+20, i+21) for i in 0..18
    # - Rungs: (i, i+20) for i in 0..19
    # Total: 19 + 19 + 20 = 58 edges.
    
    ladder_edges = []
    for i in range(19):
        ladder_edges.append((i, i + 1))
    for i in range(19):
        ladder_edges.append((i + 20, i + 21))
    for i in range(20):
        ladder_edges.append((i, i + 20))
        
    # Helper to generate cycles for the ladder graph
    def get_ladder_cycles():
        cycles = []
        for k in range(3, 41):
            if k % 2 == 1:
                # Odd cycle: uses rung at 0 and rung at m? 
                # Inspiration 2 used a chord (0, 21). But the ladder itself doesn't have that chord.
                # Let's re-verify ladder cycles without extra chords.
                # Ladder is bipartite. It only has even cycles!
                # Wait, a ladder graph L_n is bipartite. It has NO odd cycles.
                # So the standard ladder (58 edges) is NOT pancyclic. It lacks triangles, pentagons, etc.
                # Inspiration 2 added a chord (0, 21) to make it pancyclic?
                # Let's check Inspiration 2 again.
                # "Extra chord to create triangle: (0, 21)"
                # Edges in Insp 2: 19+19+20+1 = 59 edges.
                # So the base ladder is not enough. We need at least one chord to break bipartiteness.
                # 
                # Let's start with the 59-edge solution from Inspiration 2 and try to optimize.
                pass
        return None

    # Since the pure ladder is bipartite, we must add chords.
    # Let's use the 59-edge construction from Inspiration 2 as our seed.
    # Edges: Ladder + Chord (0, 21)
    seed_edges = ladder_edges + [(0, 21)]
    
    # Verify cycles for this seed
    def get_seed_cycles():
        cycles = []
        for k in range(3, 41):
            if k % 2 == 1:
                # Odd cycles use the chord (0, 21)
                # Cycle: 0-1-...-m-(m+20)-(m+19)-...-21-0
                # Length = 2m + 1 => m = (k-1)//2
                m = (k - 1) // 2
                cyc = list(range(0, m + 1))
                cyc.extend(range(m + 20, 20, -1))
                cycles.append(cyc)
            else:
                # Even cycles use the ladder structure
                # Cycle: 0-1-...-m-(m+20)-(m+19)-...-20-0
                # Length = 2m + 2 => m = k//2 - 1
                m = k // 2 - 1
                cyc = list(range(0, m + 1))
                cyc.extend(range(m + 20, 19, -1))
                cycles.append(cyc)
        return cycles

    seed_cycles = get_seed_cycles()
    
    if not _validate(seed_edges, seed_cycles):
        # Fallback to a known valid graph (e.g. from current program's fallback)
        # Hamiltonian cycle + chords
        ham_edges = [(i, (i + 1) % N) for i in range(N)]
        chord_edges = [(0, d) for d in range(2, 21)]
        edges = ham_edges + chord_edges
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                cycles.append(list(range(N)))
            elif L <= 21:
                cycles.append(list(range(L)))
            else:
                end_vertex = N + 1 - L
                cycles.append([0] + list(range(N - 1, end_vertex - 1, -1)))
        if _validate(edges, cycles):
            return {'edges': edges, 'cycles': cycles}
        # If even fallback fails, return empty (should not happen)
        return {'edges': [], 'cycles': []}

    best_edges = seed_edges
    best_cycles = seed_cycles
    best_score = len(best_edges) - 40 # 59 - 40 = 19

    # --- Step 2: Try to remove edges from the 59-edge seed ---
    # We want to get to 58, 57, etc.
    # Removing an edge might break some cycles.
    # We will iterate through edges, try removing one, and check if all cycles can be found.
    # To speed up, we only check cycles that might be affected.
    # However, with 59 edges, removing one leaves 58.
    # If we can remove one, score becomes 18.
    
    # Let's try removing each edge and see if the graph remains pancyclic.
    # We'll use a randomized order to avoid bias.
    edge_indices = list(range(len(seed_edges)))
    random.shuffle(edge_indices)
    
    for idx in edge_indices:
        if time.time() > deadline:
            break
            
        edge_to_remove = seed_edges[idx]
        trial_edges = [e for i, e in enumerate(seed_edges) if i != idx]
        
        # Quick check: does the graph remain connected?
        # And does it still have enough edges?
        # Let's just try to find cycles.
        # To be efficient, we can try to find cycles for the lengths that were using the removed edge.
        # But identifying which cycles used the edge is hard without the explicit mapping.
        # Instead, we can try to find ANY cycle of each length.
        # This is expensive (38 lengths * DFS).
        # Optimization: Only try to find cycles if the edge is likely critical.
        # Or, just run the full cycle search for the first few removals.
        
        # Given the time limit, let's only try a few removals.
        # If we find one valid removal, we update best and continue with the new base.
        
        trial_adj = _build_adj(trial_edges)
        
        valid = True
        trial_cycles = []
        
        # Try to find cycles for all lengths
        # We can prioritize lengths that might be hard.
        # For a ladder + chord, odd cycles depend on the chord. Even cycles depend on rungs/paths.
        # If we remove the chord (0, 21), all odd cycles break.
        # If we remove a rung, some even cycles break.
        
        # Let's just try to find them all.
        for L in range(3, N + 1):
            if time.time() > deadline:
                valid = False
                break
                
            # Try to find a cycle of length L
            # We can try the original cycle first? No, it might use the removed edge.
            # Just use DFS.
            # To make DFS faster, we can restrict search space or use heuristics.
            # But for now, standard DFS.
            
            # Heuristic: If L is odd and we removed the chord, it's likely impossible.
            # If L is even and we removed a rung, it might be possible via other rungs.
            
            # Let's just call _find_cycle.
            # To avoid infinite loops or very deep searches, we rely on the small N.
            # N=40, L up to 40. DFS depth 40. Branching factor ~3.
            # This might be slow.
            
            # Optimization: Check if the specific edge was part of the known seed cycle for this L.
            # If not, the seed cycle is still valid.
            # This is a huge speedup!
            
            # Let's check if the seed cycle for length L uses the edge (u, v).
            seed_cycle_L = seed_cycles[L - 3]
            uses_edge = False
            for j in range(L):
                u = seed_cycle_L[j]
                v = seed_cycle_L[(j + 1) % L]
                if (min(u, v), max(u, v)) == (min(edge_to_remove[0], edge_to_remove[1]), max(edge_to_remove[0], edge_to_remove[1])):
                    uses_edge = True
                    break
            
            if not uses_edge:
                # Seed cycle is still valid
                trial_cycles.append(seed_cycle_L)
            else:
                # Need to find a new cycle
                alt = _find_cycle(trial_adj, L)
                if alt is None:
                    valid = False
                    break
                trial_cycles.append(alt)
                
        if valid and _validate(trial_edges, trial_cycles):
            best_edges = trial_edges
            best_cycles = trial_cycles
            best_score = len(best_edges) - 40
            # Update seed_edges to the new best for subsequent iterations?
            # If we remove one, the next removal is from the new set.
            # This is a greedy local search.
            seed_edges = trial_edges
            seed_cycles = trial_cycles
            edge_indices = list(range(len(seed_edges)))
            random.shuffle(edge_indices)
            # Continue loop to try removing more
            # But we need to be careful with indices.
            # Since we updated seed_edges, the indices in the current loop are stale.
            # Let's break and restart the loop with the new edges?
            # Or just continue, but the `idx` refers to the old list.
            # Let's restart the removal loop.
            break

    # If we broke out, we might have improved.
    # We can run another pass if time permits.
    if time.time() < deadline - 5:
        # Another pass
        edge_indices = list(range(len(best_edges)))
        random.shuffle(edge_indices)
        
        for idx in edge_indices:
            if time.time() > deadline:
                break
                
            edge_to_remove = best_edges[idx]
            trial_edges = [e for i, e in enumerate(best_edges) if i != idx]
            
            trial_adj = _build_adj(trial_edges)
            valid = True
            trial_cycles = []
            
            for L in range(3, N + 1):
                if time.time() > deadline:
                    valid = False
                    break
                
                seed_cycle_L = best_cycles[L - 3]
                uses_edge = False
                for j in range(L):
                    u = seed_cycle_L[j]
                    v = seed_cycle_L[(j + 1) % L]
                    if (min(u, v), max(u, v)) == (min(edge_to_remove[0], edge_to_remove[1]), max(edge_to_remove[0], edge_to_remove[1])):
                        uses_edge = True
                        break
                
                if not uses_edge:
                    trial_cycles.append(seed_cycle_L)
                else:
                    alt = _find_cycle(trial_adj, L)
                    if alt is None:
                        valid = False
                        break
                    trial_cycles.append(alt)
                    
            if valid and _validate(trial_edges, trial_cycles):
                best_edges = trial_edges
                best_cycles = trial_cycles
                best_score = len(best_edges) - 40
                break

    return {'edges': best_edges, 'cycles': best_cycles}
