"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def canonical(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def decompose(edges):
        if not edges:
            return []
        
        pieces = []
        remaining_edges = set(canonical(e) for e in edges)
        
        # Build full adjacency once for the original graph to help with cycle finding
        # But we need dynamic adjacency for remaining edges.
        
        while remaining_edges:
            if time.time() - t0 > DEADLINE - 5:
                # Fallback: add all remaining as single edges
                for e in remaining_edges:
                    pieces.append(list(e))
                break
            
            # Build adjacency for current remaining graph
            adj = defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            if not adj:
                break
                
            # Find all cycles in the remaining graph using DFS
            # We want to find edge-disjoint cycles.
            # A simple greedy approach: find a cycle, remove its edges, repeat.
            # To be efficient, we can try to find long cycles or just any cycle.
            
            visited_nodes = set()
            cycles_found = []
            
            # We need to be careful with DFS on undirected graphs to find cycles.
            # Standard DFS finds back edges which indicate cycles.
            
            def find_cycle(start):
                # Returns a list of nodes forming a cycle, or None
                # We need to track parent to avoid going back.
                # Also need to detect if we hit a visited node that is an ancestor (back edge).
                
                # Since we are removing edges, the graph changes.
                # We will do a DFS from start.
                # We need to store the path.
                
                stack = [(start, None, 0)] # (node, parent, state)
                # state 0: entering, 1: exiting
                # This is getting complicated with iterative DFS and path tracking.
                
                # Let's use recursive DFS for simplicity, assuming n is small (<=30).
                # Recursion depth up to 30 is fine.
                
                path = []
                on_stack = set()
                
                def dfs(u, parent):
                    path.append(u)
                    on_stack.add(u)
                    for v in adj.get(u, []):
                        if v == parent:
                            continue
                        if v in on_stack:
                            # Found a cycle
                            # Extract cycle from path
                            idx = path.index(v)
                            cycle = path[idx:]
                            return cycle
                        if v not in visited_nodes:
                            res = dfs(v, u)
                            if res:
                                return res
                    path.pop()
                    on_stack.remove(u)
                    return None
                
                return dfs(start, -1)

            # Try to find cycles from each unvisited node
            # Note: visited_nodes here is for the current decomposition step to avoid redundant searches?
            # Actually, if we find a cycle, we remove its edges. The remaining graph changes.
            # So we should restart the search or continue with the updated graph.
            # Since n is small, restarting is cheap.
            
            # However, we can just find ONE cycle, remove it, and loop back to the top of the while loop.
            # This is simpler and ensures we always work with the current remaining_edges.
            
            found_cycle = None
            # Try starting from each node
            for start in sorted(adj.keys()):
                if start not in remaining_edges: # Optimization? No, start is a node.
                    pass
                # Check if start has any incident edges in remaining graph
                if not adj[start]:
                    continue
                    
                # Reset visited for this specific DFS attempt?
                # The DFS above uses local variables.
                # But we need to ensure we don't use nodes that are "isolated" in some sense?
                # No, DFS handles connectivity.
                
                # We need to make sure we don't traverse edges that are not in remaining_edges.
                # The adj set is built from remaining_edges, so it's correct.
                
                # One issue: the DFS above marks visited_nodes globally? No, it uses local path/on_stack.
                # But it doesn't mark nodes as "permanently visited" for other starts in the same pass.
                # This is fine. We just want to find ANY cycle.
                
                # Optimization: Skip if node degree < 2?
                # A node in a cycle must have degree >= 2.
                if len(adj[start]) < 2:
                    continue
                
                res = find_cycle(start)
                if res:
                    found_cycle = res
                    break
            
            if found_cycle:
                # Validate the cycle
                # Ensure all edges exist in remaining_edges
                valid = True
                seen_nodes = set()
                for i in range(len(found_cycle)):
                    u = found_cycle[i]
                    v = found_cycle[(i+1) % len(found_cycle)]
                    e = canonical((u, v))
                    if e not in remaining_edges:
                        valid = False
                        break
                    if u in seen_nodes:
                        valid = False
                        break
                    seen_nodes.add(u)
                
                if valid and len(found_cycle) >= 3:
                    pieces.append(found_cycle)
                    for i in range(len(found_cycle)):
                        u = found_cycle[i]
                        v = found_cycle[(i+1) % len(found_cycle)]
                        e = canonical((u, v))
                        remaining_edges.discard(e)
                else:
                    # If invalid, treat as no cycle found in this iteration?
                    # This shouldn't happen if adj is correct.
                    # Fallback: take a single edge
                    e = next(iter(remaining_edges))
                    pieces.append(list(e))
                    remaining_edges.discard(e)
            else:
                # No cycles found. Take all remaining as single edges.
                for e in remaining_edges:
                    pieces.append(list(e))
                remaining_edges = set()
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            results.append(decompose(edges))

    return results
# EVOLVE-BLOCK-END
