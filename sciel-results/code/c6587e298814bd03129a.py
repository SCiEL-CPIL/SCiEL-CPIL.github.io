"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
import random

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_local_search(edges, n, time_budget):
        """Start with all edges as pieces, then merge into cycles via local search."""
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        m = len(edge_set)
        
        best_pieces = None
        best_count = m + 1
        best_time = time.time()
        
        # Multiple restarts
        while time.time() < best_time + time_budget:
            restart_start = time.time()
            rng = random.Random()
            
            # Initialize: each edge is a piece
            # pieces is a list of lists of edges (tuples)
            pieces = [[(u, v)] for (u, v) in edge_set]
            
            # Build edge-to-piece index
            edge_to_piece = {}
            for i, piece in enumerate(pieces):
                for e in piece:
                    edge_to_piece[e] = i
            
            # Build adjacency from remaining edges
            adj = defaultdict(set)
            for e in edge_set:
                u, v = e
                adj[u].add(v)
                adj[v].add(u)
            
            # Local search: try to merge pieces into cycles
            improved = True
            max_iter = 5000
            iter_count = 0
            
            while improved and iter_count < max_iter:
                if time.time() > restart_start + time_budget * 0.9:
                    break
                improved = False
                iter_count += 1
                
                # Try all pairs of pieces to see if merging creates a cycle
                # We'll sample a subset for speed
                num_pieces = len(pieces)
                if num_pieces <= 50:
                    pairs_to_try = [(i, j) for i in range(num_pieces) for j in range(i + 1, num_pieces)]
                else:
                    # Sample pairs
                    pairs_to_try = []
                    for _ in range(min(200, num_pieces * 10)):
                        i = rng.randint(0, num_pieces - 1)
                        j = rng.randint(0, num_pieces - 1)
                        if i != j:
                            pairs_to_try.append((min(i, j), max(i, j)))
                    # Remove duplicates
                    pairs_to_try = list(set(pairs_to_try))
                
                for i, j in pairs_to_try:
                    if time.time() > restart_start + time_budget * 0.9:
                        break
                    
                    if i >= len(pieces) or j >= len(pieces):
                        continue
                    
                    piece_i = pieces[i]
                    piece_j = pieces[j]
                    
                    # Try merging: concatenate the two edge sets and check if they form a cycle
                    # A cycle requires: all vertices have degree 2 in the merged subgraph
                    merged_edges = list(piece_i) + list(piece_j)
                    
                    # Check if merged edges form a single cycle
                    # Count degrees in merged subgraph
                    deg = defaultdict(int)
                    for u, v in merged_edges:
                        deg[u] += 1
                        deg[v] += 1
                    
                    # All vertices must have degree exactly 2
                    if all(d == 2 for d in deg.values()) and len(merged_edges) >= 3:
                        # Check connectivity: BFS from one vertex
                        visited = set()
                        start_v = next(iter(deg))
                        stack = [start_v]
                        while stack:
                            v = stack.pop()
                            if v in visited:
                                continue
                            visited.add(v)
                            for u, w in merged_edges:
                                if u == v and w not in visited:
                                    stack.append(w)
                                elif w == v and u not in visited:
                                    stack.append(u)
                        
                        if len(visited) == len(deg):
                            # Valid cycle! Merge the pieces
                            new_piece = merged_edges
                            pieces[i] = new_piece
                            # Remove piece_j
                            pieces.pop(j)
                            
                            # Update edge_to_piece
                            for e in piece_j:
                                edge_to_piece[e] = i
                            
                            improved = True
                            break
                        else:
                            # Not connected, try other merge strategy
                            pass
                    
                    # Alternative: try to extend piece_i with edges from piece_j to form a cycle
                    # This is more complex, skip for now in this iteration
            
            # After local search, convert pieces to output format
            # Cycles are represented by their vertex sequence
            result = []
            for piece in pieces:
                if len(piece) == 1:
                    # Single edge
                    u, v = piece[0]
                    result.append([u, v])
                else:
                    # Try to extract a cycle from the edges
                    # Build adjacency for this piece
                    p_adj = defaultdict(list)
                    for u, v in piece:
                        p_adj[u].append(v)
                        p_adj[v].append(u)
                    
                    # Find the cycle by walking
                    if all(len(nbs) == 2 for nbs in p_adj.values()):
                        start = next(iter(p_adj))
                        cycle = [start]
                        prev = None
                        curr = start
                        while True:
                            nexts = [x for x in p_adj[curr] if x != prev]
                            if not nexts:
                                break
                            next_v = nexts[0]
                            if next_v == start:
                                break
                            cycle.append(next_v)
                            prev = curr
                            curr = next_v
                        
                        if len(cycle) >= 3:
                            result.append(cycle)
                        else:
                            # Fallback: output edges individually
                            for e in piece:
                                result.append([e[0], e[1]])
                    else:
                        # Not a clean cycle, output edges individually
                        for e in piece:
                            result.append([e[0], e[1]])
            
            count = len(result)
            if count < best_count:
                best_count = count
                best_pieces = result
                best_time = time.time()
            
            # Early termination if good enough
            if best_count <= m // 4 + 2:
                break
        
        return best_pieces if best_pieces else [list(e) for e in edge_set]

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    # Allocate time budget per graph
    total_budget = 140  # leave 10s for overhead
    per_graph = total_budget / len(specs)
    
    for idx, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        if time.time() > deadline - 5:
            results.append([list(e) for e in edges])
            continue
        
        # Give more time to larger graphs
        time_budget = per_graph * (0.8 + 0.4 * idx / (len(specs) - 1))
        pieces = decompose_local_search(edges, n, time_budget)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
