"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    """Decompose three fixed graphs into few cycles and edges."""
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n, time_budget=40.0):
        """
        Decompose a graph into cycles and edges, minimizing piece count.
        1. Greedy: Repeatedly find the longest cycle, remove it.
        2. Local Search: Try to improve by swapping cycles.
        """
        edge_set = set(tuple(e) for e in edges)
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []

        def find_cycle(adj, edge_set, start=None, max_steps=1000):
            """
            Find a cycle. Returns list of vertices [v0, ..., vk-1] such that (v0,v1)...(vk-1,v0) are edges.
            Uses randomized DFS.
            """
            if not adj:
                return None

            if start is None:
                candidates = [v for v in adj if adj[v]]
                if not candidates:
                    return None
                start = random.choice(candidates)

            best_cycle = None
            
            # Try multiple random DFS traversals from start
            for _ in range(5):
                visited = {start}
                stack = [(start, [start])]
                found = False
                
                while stack and not found:
                    node, path = stack.pop()
                    
                    neighbors = list(adj[node])
                    random.shuffle(neighbors)
                    
                    for nb in neighbors:
                        edge = tuple(sorted((node, nb)))
                        if edge not in edge_set:
                            continue
                        
                        if nb == start and len(path) >= 3:
                            # Found a cycle
                            cycle = path[:]
                            if best_cycle is None or len(cycle) > len(best_cycle):
                                best_cycle = cycle
                            found = True
                            break
                            
                        if nb not in visited:
                            visited.add(nb)
                            stack.append((nb, path + [nb]))
                            if len(path) > 50: # Safety limit for depth
                                break

                if found:
                    break

            return best_cycle

        def remove_cycle_edges(cycle, edge_set, adj):
            m = len(cycle)
            for i in range(m):
                u, v = cycle[i], cycle[(i + 1) % m]
                edge = tuple(sorted((u, v)))
                if edge in edge_set:
                    edge_set.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)

        def add_cycle_edges(cycle, edge_set, adj):
            m = len(cycle)
            for i in range(m):
                u, v = cycle[i], cycle[(i + 1) % m]
                edge = tuple(sorted((u, v)))
                edge_set.add(edge)
                adj[u].add(v)
                adj[v].add(u)

        # Phase 1: Greedy cycle extraction
        # We want to maximize the total length of cycles to minimize pieces.
        # Piece count = (#cycles) + (#remaining_edges).
        # #remaining_edges = E - sum(len(cycle_i)).
        # Total = #cycles + E - sum(len(cycle_i)).
        # To minimize, maximize sum(len(cycle_i) - 1).
        
        max_iterations = 5000
        iter_count = 0
        while edge_set and iter_count < max_iterations:
            if time.time() - start_time > DEADLINE - 10:
                break

            # Find the longest cycle available
            best_cycle = None
            # Try a few times to find a long cycle
            for _ in range(10):
                c = find_cycle(adj, edge_set)
                if c:
                    if best_cycle is None or len(c) > len(best_cycle):
                        best_cycle = c
                    if len(best_cycle) >= 10: # Good enough
                        break

            if best_cycle is None:
                break

            remove_cycle_edges(best_cycle, edge_set, adj)
            pieces.append(best_cycle)
            iter_count += 1

        # Phase 2: Remaining edges as single-edge pieces
        for u, v in sorted(edge_set):
            pieces.append([u, v])

        # Phase 3: Local Search
        # Try to improve by swapping a cycle with a different set of pieces covering the same edges?
        # Or just try to find a cycle in the "single edges" that we missed?
        # If we have single edges, they form a graph. If that graph has a cycle, we should have found it in Phase 1.
        # Why would we miss it? Because Phase 1 removed edges that might have been part of a better cycle.
        
        # Let's try a simple local search:
        # For each cycle in pieces, remove it, put its edges back into the pool of "single edges" (conceptually),
        # and try to find a cycle in the subgraph formed by (Current Single Edges + Edges of this Cycle).
        # If we find a cycle that uses at least one edge from the original cycle, and the new configuration
        # results in fewer pieces, accept it.
        
        # This is complex to manage state. Let's do a simpler improvement:
        # Check if the current "single edges" form any cycles (should be none if Phase 1 was perfect).
        # But Phase 1 might have left a graph with no cycles? No, if it has no cycles, it's a forest.
        # So all remaining edges are bridges or part of trees.
        
        # The main loss is if we picked a short cycle that blocked a long one.
        # We can try to "undo" a cycle and re-greedy.
        
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 20):
            improved = False
            # Pick a random cycle to "challenge"
            if not pieces:
                break
            idx = random.randint(0, len(pieces) - 1)
            if len(pieces[idx]) < 3:
                continue
            
            # Save state
            orig_cycle = pieces[idx]
            # Remove this cycle from pieces and add its edges to a temporary set
            pieces.pop(idx)
            
            # Reconstruct edge set for local search:
            # Current edge_set is empty (all covered by pieces).
            # We need to consider: All other pieces' edges are "fixed".
            # The edges of orig_cycle are "free".
            # The "single edge" pieces are also "free" in the sense that we could merge them into cycles?
            # No, single edge pieces are just edges.
            
            # Let's build a local graph from:
            # 1. Edges of orig_cycle
            # 2. Edges of all single-edge pieces (length 2)
            
            local_edges = set()
            local_adj = defaultdict(set)
            
            # Add orig_cycle edges
            m = len(orig_cycle)
            for i in range(m):
                u, v = orig_cycle[i], orig_cycle[(i + 1) % m]
                e = tuple(sorted((u, v)))
                local_edges.add(e)
                local_adj[u].add(v)
                local_adj[v].add(u)
                
            # Add single edge pieces
            single_indices = []
            for i, p in enumerate(pieces):
                if len(p) == 2:
                    u, v = p
                    e = tuple(sorted((u, v)))
                    local_edges.add(e)
                    local_adj[u].add(v)
                    local_adj[v].add(u)
                    single_indices.append(i)
            
            # Try to find a cycle in local_adj
            # If found, it must use at least one edge from orig_cycle?
            # Actually, if it only uses single edges, that means we missed a cycle in Phase 1.
            # If it uses orig_cycle edges, it might be a better cycle.
            
            new_cycle = find_cycle(local_adj, local_edges)
            
            if new_cycle:
                # Check if this new cycle is "better"
                # Old pieces removed: 1 cycle (len L1) + some single edges (count K) that are in new_cycle?
                # Wait, the new cycle might use some single edges.
                # Let's count pieces before and after.
                
                # Before: 1 cycle (orig) + K_single_in_new_cycle (if any)
                # After: 1 cycle (new) + (K_single_in_new_cycle - count_of_singles_in_new)
                
                # Actually, let's just simulate the replacement.
                # Remove edges of new_cycle from local_edges.
                # The remaining edges in local_edges become single pieces.
                # The new_cycle becomes a cycle piece.
                
                # We need to compare the total piece count for this local set.
                # Old count: 1 (orig cycle) + (number of single edges in local_edges that were NOT in orig_cycle)
                # This is getting messy to track indices.
                
                # Simpler: Just replace orig_cycle with new_cycle IF new_cycle is strictly longer
                # AND uses only edges from orig_cycle? No, it can use single edges.
                
                # Let's just accept if new_cycle length > orig_cycle length.
                # This might not reduce piece count if it "consumes" many single edges.
                # Piece count change = -1 (remove orig) - (singles consumed) + 1 (add new) = - (singles consumed).
                # So if it consumes singles, piece count decreases!
                
                # So we should always accept if new_cycle exists and is different from orig_cycle?
                # If new_cycle == orig_cycle, no change.
                # If new_cycle uses only orig_cycle edges, it's the same or different cycle on same edges.
                # If new_cycle uses single edges, it reduces piece count.
                
                # Let's verify if new_cycle is valid in the global context.
                # It uses edges from local_edges, which are a subset of global edges.
                # Yes.
                
                # Perform the swap:
                # 1. We already popped orig_cycle.
                # 2. We need to remove the single edges that are part of new_cycle from 'pieces'.
                # 3. We need to add new_cycle to 'pieces'.
                
                # Identify which single edges are in new_cycle
                new_cycle_edges = set()
                m_new = len(new_cycle)
                for i in range(m_new):
                    u, v = new_cycle[i], new_cycle[(i + 1) % m_new]
                    new_cycle_edges.add(tuple(sorted((u, v))))
                    
                # Remove those single edges from pieces
                # We need to be careful with indices if we remove while iterating.
                # Create a new list of pieces
                new_pieces = []
                for p in pieces:
                    if len(p) == 2:
                        e = tuple(sorted((p[0], p[1])))
                        if e in new_cycle_edges:
                            # This single edge is consumed by the new cycle
                            continue
                    new_pieces.append(p)
                    
                # Add the new cycle
                new_pieces.append(new_cycle)
                
                # Check if we actually improved (piece count decreased or same but better structure?)
                # Old piece count for this local part: 1 (orig) + len(single_indices)
                # New piece count for this local part: 1 (new) + (len(single_indices) - len(consumed_singles))
                # Improvement = len(consumed_singles)
                
                if len(new_pieces) < len(pieces) + 1: # +1 because we popped orig
                    pieces = new_pieces
                    improved = True
            else:
                # No new cycle found, put orig back
                pieces.insert(idx, orig_cycle)

        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)

        if time.time() - start_time > DEADLINE - 5:
            # Fallback: all edges as single pieces
            results.append([list(e) for e in edges])
            continue

        pieces = decompose_graph(edges, n, time_budget=40.0)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
