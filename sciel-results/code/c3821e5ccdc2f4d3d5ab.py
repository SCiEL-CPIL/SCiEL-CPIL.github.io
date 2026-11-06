"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find largest connected component
    visited = [False] * n
    largest_cc = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(comp) > len(largest_cc):
                largest_cc = comp
    
    if not largest_cc:
        return [0]
    
    cc_set = set(largest_cc)
    
    best_path = []
    
    def try_greedy(start, order_key):
        """Greedy path extension with a given neighbor ordering."""
        path = [start]
        seen = {start}
        while True:
            candidates = [nb for nb in adj[path[-1]] if nb in cc_set and nb not in seen]
            if not candidates:
                break
            # Choose next vertex based on order_key
            if order_key == 'min_unvisited':
                # Fewest unvisited neighbors first (Warnsdorff)
                next_v = min(candidates, key=lambda w: (len([x for x in adj[w] if x in cc_set and x not in seen]), w))
            elif order_key == 'max_unvisited':
                # Most unvisited neighbors first
                next_v = max(candidates, key=lambda w: (len([x for x in adj[w] if x in cc_set and x not in seen]), w))
            elif order_key == 'min_degree':
                # Lowest total degree first
                next_v = min(candidates, key=lambda w: (len(adj[w]), w))
            elif order_key == 'max_degree':
                # Highest total degree first
                next_v = max(candidates, key=lambda w: (len(adj[w]), w))
            else:
                next_v = candidates[0]
            path.append(next_v)
            seen.add(next_v)
        return path
    
    # Try multiple greedy strategies from each start
    for start in largest_cc:
        for order in ['min_unvisited', 'max_unvisited', 'min_degree', 'max_degree']:
            path = try_greedy(start, order)
            if len(path) > len(best_path):
                best_path = path
    
    # Local search: try to extend the best path from both ends
    improved = True
    while improved:
        improved = False
        path = best_path[:]
        seen = set(path)
        
        # Try extending from the end
        while True:
            candidates = [nb for nb in adj[path[-1]] if nb in cc_set and nb not in seen]
            if not candidates:
                break
            # Try each candidate, pick the one that leads to the longest extension
            best_ext = None
            for nb in candidates:
                # Greedily extend from nb
                sub_path = [nb]
                sub_seen = seen | {nb}
                while True:
                    cands = [x for x in adj[sub_path[-1]] if x in cc_set and x not in sub_seen]
                    if not cands:
                        break
                    next_v = min(cands, key=lambda w: (len([x for x in adj[w] if x in cc_set and x not in sub_seen]), w))
                    sub_path.append(next_v)
                    sub_seen.add(next_v)
                if best_ext is None or len(sub_path) > len(best_ext):
                    best_ext = sub_path
            if best_ext:
                path.extend(best_ext)
                seen.update(best_ext)
                improved = True
        
        # Try extending from the beginning
        while True:
            candidates = [nb for nb in adj[path[0]] if nb in cc_set and nb not in seen]
            if not candidates:
                break
            best_ext = None
            for nb in candidates:
                sub_path = [nb]
                sub_seen = seen | {nb}
                while True:
                    cands = [x for x in adj[sub_path[-1]] if x in cc_set and x not in sub_seen]
                    if not cands:
                        break
                    next_v = min(cands, key=lambda w: (len([x for x in adj[w] if x in cc_set and x not in sub_seen]), w))
                    sub_path.append(next_v)
                    sub_seen.add(next_v)
                if best_ext is None or len(sub_path) > len(best_ext):
                    best_ext = sub_path
            if best_ext:
                path = best_ext + path
                seen.update(best_ext)
                improved = True
        
        if len(path) > len(best_path):
            best_path = path
    
    if not best_path:
        best_path = [largest_cc[0]]
    
    return best_path
# EVOLVE-BLOCK-END
