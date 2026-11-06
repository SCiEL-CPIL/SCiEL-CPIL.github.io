import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    for n in [10, 11]:
        all_edges = list(combinations(range(n), 5))
        total_edges = len(all_edges)
        
        # Search for non-2-colorable hypergraph
        # Start from low m
        for m in range(50, min(total_edges, 100) + 1):
            # Try random instances
            found = None
            trials = 0
            while time.time() < deadline and trials < 500:
                trials += 1
                edges = random.sample(all_edges, m)
                if not is_2_colorable_brute_force(n, edges):
                    found = edges
                    break
            
            if found:
                # Try to minimize by removing edges
                edges_list = list(found)
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    for i in range(len(edges_list)):
                        # Try removing edge i
                        trial_edges = edges_list[:i] + edges_list[i+1:]
                        if not is_2_colorable_brute_force(n, trial_edges):
                            edges_list = trial_edges
                            changed = True
                            break # Restart loop as list changed
                
                if time.time() < deadline:
                    return {'n': n, 'edges': [list(e) for e in edges_list]}
    
    # Fallback
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

def is_2_colorable_brute_force(n, edges):
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    for coloring in range(1 << n):
        valid = True
        for mask in edge_masks:
            if (mask & coloring) == 0 or (mask & coloring) == mask:
                valid = False
                break
        if valid:
            return True
    return False
