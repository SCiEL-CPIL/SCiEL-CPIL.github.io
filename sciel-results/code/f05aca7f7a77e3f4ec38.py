import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
            
    components.sort(key=len, reverse=True)
    
    best_path = []
    deadline = time.time() + 55.0
    
    for comp in components:
        if time.time() > deadline:
            break
            
        comp_set = set(comp)
        
        # If component is small, use exact DFS
        if len(comp) <= 20:
             # ... existing DFS logic ...
             pass
        else:
             # Beam Search
             # State: (path, visited_set)
             # We store paths as tuples to be hashable
             
             beam = []
             for node in comp:
                 beam.append(([node], {node}))
                 
             # Sort initial beam by length (all 1)
             
             while beam and time.time() < deadline:
                 new_beam = []
                 for path, vis in beam:
                     last = path[-1]
                     for neighbor in adj[last]:
                         if neighbor in comp_set and neighbor not in vis:
                             new_path = path + [neighbor]
                             new_vis = vis | {neighbor}
                             new_beam.append((new_path, new_vis))
                             
                 # Keep top K
                 K = 5000
                 new_beam.sort(key=lambda x: len(x[0]), reverse=True)
                 beam = new_beam[:K]
                 
                 # Update best path
                 if beam:
                     if len(beam[0][0]) > len(best_path):
                         best_path = beam[0][0]
                         
    return best_path
# EVOLVE-BLOCK-END
