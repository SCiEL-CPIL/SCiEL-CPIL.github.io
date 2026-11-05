"""Exact Cover via Max-Flow Cycle Packing."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
from random import Random

DEADLINE_MARGIN = 10  # seconds before 150s hard limit

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_graph(edges):
    """
    Decompose edges into cycles and single edges using a Max-Flow based approach.
    1. Peel off leaves (degree 1) and add their edges as single pieces.
    2. The remaining graph has min degree >= 2.
    3. Use a heuristic to extract edge-disjoint cycles:
       - Try to find a large set of edge-disjoint cycles by solving a flow problem?
       - Actually, finding a maximum set of edge-disjoint cycles is NP-hard.
       - However, for these small graphs (n=20, 25, 30), we can use a simpler heuristic.
       
    Alternative Strategy: 
    - Since the graphs are small, we can try to find a Hamiltonian cycle first.
    - If found, remove it. 
    - Then repeat on the remaining graph.
    - If no Hamiltonian cycle, find a long cycle (e.g., using a greedy DFS or a simple heuristic).
    
    Let's try a "Greedy Long Cycle" approach with a twist:
    - Repeatedly find a cycle using a DFS that tries to maximize length.
    - To maximize length, we can use a simple heuristic: 
      1. Start at a vertex with high degree.
      2. Perform a DFS, but when choosing the next vertex, prefer one that has 
         high degree and is not yet visited.
      3. If we close a cycle, check if it's "long enough". 
         If not, backtrack? That's complex.
    
    A better approach for small graphs:
    - Use a simple iterative removal of cycles found by a DFS.
    - To improve, we can try multiple random starts and keep the best (longest) cycle found.
    
    Let's implement a robust cycle finder:
    1. While edges remain in the 2-core:
       a. Find a cycle using a randomized DFS.
       b. Remove the cycle's edges.
       c. Peel off any new leaves.
       d. Repeat.
    
    To make the cycles longer, we can use a heuristic in the DFS:
    - When at a vertex, sort neighbors by degree (descending) and try them in that order.
    - This encourages the DFS to go deep into high-degree areas, potentially forming longer cycles.
    """
    if not edges:
        return []

    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining_edges = set(edges)

    def peel_leaves():
        """Remove all degree-1 vertices and their edges, adding them as single pieces."""
        changed = True
        while changed:
            changed = False
            to_remove = []
            for v in list(adj.keys()):
                if len(adj[v]) == 1:
                    nb = list(adj[v])[0]
                    to_remove.append((v, nb))
            
            if to_remove:
                changed = True
                for u, v in to_remove:
                    if u in adj and v in adj[u]:
                        e = (u, v) if u < v else (v, u)
                        if e in remaining_edges:
                            pieces.append([u, v])
                            remaining_edges.discard(e)
                            adj[u].discard(v)
                            adj[v].discard(u)
            # Clean up empty nodes
            for v in list(adj.keys()):
                if not adj[v]:
                    del adj[v]

    # Initial peel
    peel_leaves()

    # Phase 2: Extract cycles from the remaining 2-core (or higher)
    # We want to minimize pieces, so we want LONG cycles.
    # A greedy approach: repeatedly find a "good" cycle.
    
    found_cycle = True
    while found_cycle and remaining_edges:
        # Check if any vertex has degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break
        
        found_cycle = False
        # Try a few random starts to find a decently long cycle
        rng = Random(len(remaining_edges)) # Simple seed based on state
        rng.shuffle(candidates)
        
        best_cycle = None
        best_len = 0
        
        # Try up to 10 random starts to find a long cycle
        for start in candidates[:10]:
            # DFS to find a cycle
            visited = {start: 0} # node: index in path
            path = [start]
            stack = [(start, iter(sorted(adj[start], key=lambda x: -len(adj[x]))))] # Sort by degree desc
            
            while stack:
                node, it = stack[-1]
                advanced = False
                for nb in it:
                    if (node, nb) if node < nb else (nb, node) not in remaining_edges:
                        continue
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        cyc_len = len(path)
                        if cyc_len > best_len:
                            best_len = cyc_len
                            best_cycle = path[:]
                        # We could break here to take the first one, but we want longer
                        # However, DFS might find a short one first.
                        # Let's continue searching for a longer one? 
                        # That's expensive. 
                        # Let's just take the first valid cycle found in this DFS?
                        # Or break out of this DFS?
                        # If we break, we might miss a longer cycle in the same DFS tree.
                        # Given time constraints, let's take the first cycle found in this attempt.
                        # But we try multiple starts.
                        break
                    if nb not in visited:
                        visited[nb] = len(path)
                        path.append(nb)
                        stack.append((nb, iter(sorted(adj[nb], key=lambda x: -len(adj[x])))))
                        advanced = True
                        break
                if not advanced:
                    # Backtrack
                    stack.pop()
                    if path:
                        path.pop()
                        # Remove from visited? 
                        # In standard DFS cycle detection, we keep visited to avoid cycles in the tree,
                        # but we need to detect cycles back to start.
                        # This implementation is slightly flawed for finding ALL cycles or the LONGEST.
                        # It finds A cycle.
                    if not path:
                        break
            
            if best_cycle and len(best_cycle) >= 3:
                found_cycle = True
                break # Take the first good start that yields a cycle? 
                # Actually, let's finish the loop to see if another start gives a LONGER cycle?
                # That might be too slow. Let's just take the first one found.
                # To improve, we can prioritize starts with higher degree.
        
        if best_cycle and len(best_cycle) >= 3:
            # Add cycle to pieces
            pieces.append(best_cycle[:])
            # Remove edges
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (u, v) if u < v else (v, u)
                remaining_edges.discard(e)
                if u in adj and v in adj[u]:
                    adj[u].discard(v)
                if v in adj and u in adj[v]:
                    adj[v].discard(u)
            
            # After removing a cycle, degrees change.
            # We might create new leaves.
            peel_leaves()

    # Phase 3: Handle remaining edges
    for e in remaining_edges:
        pieces.append(list(e))

    return pieces

def construct():
    start = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, spec in enumerate(specs):
        if time.time() - start > DEADLINE_MARGIN - 1:
            # Emergency fallback: just return edges as single pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        pieces = decompose_graph(edges)
        results.append(pieces)

    return results[:3]

# EVOLVE-BLOCK-END
