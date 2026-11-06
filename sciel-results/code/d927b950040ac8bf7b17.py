import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    time_limit = 120.0  # Keep a safety margin from 150s
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle_approx(graph, max_attempts=20, time_budget=0.5):
        """
        Finds a cycle in the graph. Tries multiple randomized DFS paths to find a relatively long one.
        graph: dict mapping node -> set of neighbors
        Returns: list of nodes in the cycle, or None if no cycle.
        """
        if not graph:
            return None
            
        nodes = list(graph.keys())
        if not nodes:
            return None
            
        # Filter to nodes with degree > 0
        active_nodes = [n for n in nodes if graph[n]]
        if not active_nodes:
            return None
            
        best_cycle = None
        local_start = time.time()
        
        for _ in range(max_attempts):
            if time.time() - local_start > time_budget:
                break
                
            # Pick a random start node
            start_node = random.choice(active_nodes)
            
            # DFS to find a cycle
            # State: (current_node, path)
            # We want to find a path that ends at start_node or forms a cycle back to an earlier node in path.
            # Actually, standard cycle finding:
            # We do a DFS. If we hit a visited node that is in the current path, we have a cycle.
            
            visited_in_path = set()
            path = []
            
            # Iterative DFS to avoid recursion limits and allow easier backtracking/heuristics
            # Stack stores (node, iterator over neighbors)
            # But we need to reconstruct the path.
            
            # Let's use a recursive-like approach with a stack for path management.
            # Simpler: Randomized Greedy Path Extension
            current = start_node
            path = [current]
            visited_in_path = {current}
            
            while True:
                neighbors = list(graph[current])
                # Filter neighbors: prefer unvisited, but allow closing to start_node
                unvisited_neighbors = [n for n in neighbors if n not in visited_in_path]
                
                # Check if we can close the cycle
                if start_node in neighbors and len(path) >= 3:
                    # Found a cycle
                    cycle = path[:]
                    if not best_cycle or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    # We found a cycle, but maybe we can find a longer one?
                    # For now, we return this one to keep it simple and fast.
                    # Or we can break and let the outer loop handle removal.
                    # To maximize length, we might want to continue searching?
                    # No, if we continue, we might break the path.
                    # Let's just return the first valid cycle found in this attempt.
                    # The multiple attempts handle the "longest" aspect statistically.
                    return path
                
                if not unvisited_neighbors:
                    # Dead end
                    break
                
                # Pick a random unvisited neighbor
                # Heuristic: pick one with highest degree? Or just random?
                # Random is fine with multiple attempts.
                next_node = random.choice(unvisited_neighbors)
                path.append(next_node)
                visited_in_path.add(next_node)
                current = next_node
            
        return best_cycle

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        if time.time() - start_time > time_limit - 10:
            # Fallback to trivial solution if we are running out of time
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(n, p, seed)
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        pieces = []
        
        # Work on a copy of the adjacency list
        current_adj = defaultdict(set)
        for u, v in adj.items():
            current_adj[u] = set(v)
            
        while True:
            if time.time() - start_time > time_limit - 5:
                # Emergency: add all remaining edges as single pieces
                remaining_edges = set()
                for u in current_adj:
                    for v in current_adj[u]:
                        if u < v:
                            remaining_edges.add((u, v))
                for u, v in remaining_edges:
                    pieces.append([u, v])
                break
                
            # Check if any edges remain
            has_edges = False
            for u in current_adj:
                if current_adj[u]:
                    has_edges = True
                    break
            
            if not has_edges:
                break
            
            # Find a cycle
            cycle = find_longest_cycle_approx(current_adj, max_attempts=50, time_budget=0.1)
            
            if cycle:
                # Add cycle as a piece
                pieces.append(cycle)
                
                # Remove edges of the cycle from current_adj
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    current_adj[u].discard(v)
                    current_adj[v].discard(u)
            else:
                # No cycle found, the remaining graph is a forest
                # Add all remaining edges as single pieces
                remaining_edges = set()
                for u in current_adj:
                    for v in current_adj[u]:
                        if u < v:
                            remaining_edges.add((u, v))
                
                for u, v in remaining_edges:
                    pieces.append([u, v])
                
                break
                
        results.append(pieces)
        
    return results

# EVOLVE-BLOCK-START
# (The construct function is defined above, so we just need to ensure it's the entry point)
# The prompt asks to keep the function names and signatures.
# The seed program had construct() inside EVOLVE-BLOCK.
# I will place the helper functions and construct() inside the block or ensure construct is the main one.
# The task says "Keep the function names and signatures of the seed program."
# Seed: def construct():
# I will define helpers outside or inside? Inside is safer for isolation, but outside is cleaner.
# Given the "EVOLVE-BLOCK" structure in the seed, I should probably put everything inside or just ensure construct is there.
# I'll put the whole logic in construct to be safe, or define helpers at module level if allowed.
# The seed put everything in construct. I will follow that pattern to be safe, or just define construct.
# Actually, defining helpers at module level is fine as long as construct is the entry point.
# But to strictly follow "rewrite the program", I will provide the full file.

# Wait, the seed program is:
# def construct():
#    ...
#
# I will provide the complete program with construct() as the entry point.

# EVOLVE-BLOCK-END
