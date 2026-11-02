import random
import numpy as np

def construct(k, g):
    if k == 3 and g == 16:
        # Target n around 50000-100000
        # We will try a few sizes
        for n in [50000, 60000, 80000, 100000]:
            # Try to find a valid graph with this n
            for attempt in range(20):
                # Generate random 3-regular graph
                # Configuration model
                while True:
                    stubs = list(range(n)) * 3
                    random.shuffle(stubs)
                    edges = []
                    valid = True
                    seen_edges = set()
                    for i in range(0, 3 * n, 2):
                        u = stubs[i]
                        v = stubs[i + 1]
                        if u == v:
                            valid = False
                            break
                        if u > v:
                            u, v = v, u
                        if (u, v) in seen_edges:
                            valid = False
                            break
                        seen_edges.add((u, v))
                        edges.append((u, v))
                    if valid:
                        break
                
                # Check girth
                # Build adjacency list
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                girth_ok = True
                # BFS from each vertex to depth 7
                # We can optimize by only checking a subset or early exit
                for start in range(n):
                    if not girth_ok:
                        break
                    # BFS
                    visited = {start: 0}
                    queue = [(start, 0)]
                    while queue:
                        node, depth = queue.pop(0)
                        if depth >= 7:
                            continue
                        for neighbor in adj[node]:
                            if neighbor == start:
                                # Cycle of length depth + 1
                                # But we need to check if it's a simple cycle
                                # In a 3-regular graph, backtracking is the only way to return to start at small depths
                                # Actually, BFS tree edges are fine.
                                # If neighbor is start, cycle length is depth+1.
                                # We need depth+1 < 16.
                                # Since depth <= 7, depth+1 <= 8 < 16.
                                # So any return to start is a short cycle.
                                girth_ok = False
                                break
                            if neighbor in visited:
                                # Cycle length = depth + 1 + visited[neighbor]
                                # Wait, visited[neighbor] is the depth from start.
                                # The path is start -> ... -> node (depth) -> neighbor (visited at depth d)
                                # This forms a cycle of length depth + 1 + d?
                                # No, the path from start to neighbor is d.
                                # The path from start to node is depth.
                                # The edge node-neighbor closes the loop?
                                # The cycle is start -> ... -> neighbor -> node -> ... -> start?
                                # No, the cycle is start -> ... -> neighbor (len d) + edge (neighbor-node) + node -> ... -> start (len depth)?
                                # No, node is at depth `depth`. neighbor is at depth `visited[neighbor]`.
                                # The path start->...->node is unique in BFS tree.
                                # The path start->...->neighbor is unique in BFS tree.
                                # The edge (node, neighbor) connects them.
                                # Cycle length = depth + 1 + visited[neighbor].
                                # We need this to be < 16.
                                if depth + 1 + visited[neighbor] < 16:
                                    girth_ok = False
                                    break
                            else:
                                visited[neighbor] = depth + 1
                                queue.append((neighbor, depth + 1))
                
                if girth_ok:
                    return n, edges
                    
        # Fallback if no small graph found
        # Use the original L-U-W or similar, but it's huge
        # Or just return a large random graph
        n = 100000
        while True:
            stubs = list(range(n)) * 3
            random.shuffle(stubs)
            edges = []
            valid = True
            seen_edges = set()
            for i in range(0, 3 * n, 2):
                u = stubs[i]
                v = stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if u > v:
                    u, v = v, u
                if (u, v) in seen_edges:
                    valid = False
                    break
                seen_edges.add((u, v))
                edges.append((u, v))
            if valid:
                break
        return n, edges

    else:
        # General case
        n = 2 * k
        edges = []
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((i, j + k))
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((i + k, j))
        return n, edges
