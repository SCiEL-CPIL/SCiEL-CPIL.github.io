import time
import random
import math
import numpy as np
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    if n <= 6:
        # Standard recursive construction is optimal or near-optimal for small n
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 7:
        # Use de Bruijn graph approach:
        # Nodes: all (n-1)-permutations (6! = 720 nodes)
        # Edges: all n-permutations (7! = 5040 edges)
        # Find Eulerian circuit to cover all edges exactly once
        
        # Generate all (n-1)-permutations and map them to indices
        nodes = list(permutations(range(1, n)))
        node_index = {node: i for i, node in enumerate(nodes)}
        N_nodes = len(nodes)  # 720
        
        # For each n-permutation, determine its source and target nodes
        # Edge from node[:-1] to node[1:]
        all_perms = list(permutations(range(1, n + 1)))
        N_edges = len(all_perms)  # 5040
        
        # Build adjacency list
        # adj[u] = list of (v, edge_id) where edge_id is the permutation
        adj = [[] for _ in range(N_nodes)]
        edge_u = [0] * N_edges  # source node index
        edge_v = [0] * N_edges  # target node index
        
        for eid, perm in enumerate(all_perms):
            u = node_index[perm[:n-1]]
            v = node_index[perm[1:]]
            adj[u].append((v, eid))
            edge_u[eid] = u
            edge_v[eid] = v
        
        # Find Eulerian circuit using Hierholzer's algorithm
        # Since the graph is strongly connected and balanced (in-degree = out-degree for each node),
        # an Eulerian circuit exists.
        
        # Use iterative Hierholzer to avoid recursion depth issues
        visited_edges = [False] * N_edges
        circuit = []
        
        # Start from node 0
        stack = [(0, -1)]  # (node, next_edge_index_in_adj)
        while stack:
            node, next_idx = stack[-1]
            
            # Find next unvisited edge from this node
            found = False
            while next_idx < len(adj[node]):
                v, eid = adj[node][next_idx]
                next_idx += 1
                if not visited_edges[eid]:
                    visited_edges[eid] = True
                    circuit.append(eid)
                    stack[-1] = (node, next_idx)
                    stack.append((v, 0))
                    found = True
                    break
            
            if not found:
                # Backtrack
                stack.pop()
        
        # circuit now contains all edges in reverse order
        circuit.reverse()
        
        # Build the word from the Eulerian circuit
        # The circuit is a sequence of edges (permutations)
        # For the first edge, we include the full permutation
        # For subsequent edges, we append only the last symbol (since the first n-1 symbols match the previous edge's last n-1 symbols)
        
        word = []
        for eid in circuit:
            perm = all_perms[eid]
            if not word:
                word.extend(perm)
            else:
                # Verify overlap (should be n-1)
                # Append only the last symbol
                word.append(perm[-1])
        
        return "".join(map(str, word))

    # Fallback for n > 7
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
