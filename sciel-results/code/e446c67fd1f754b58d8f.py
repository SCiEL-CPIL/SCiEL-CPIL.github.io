import itertools
import time

def construct():
    start_time = time.time()
    
    # Generate forbidden triples
    triples = []
    # x < y, gcd(x, y) == 1, x >= 2, y >= 3
    # a = k * x * y * (x + y)
    # b = k * x * (x + y)
    # c = k * y * (x + y)
    # a, b, c <= 180
    
    for x in range(2, 181):
        for y in range(x + 1, 181):
            if x * (x + y) > 180: # b must be <= 180 for k=1
                break
            if y * (x + y) > 180: # c must be <= 180 for k=1
                continue
            # Check gcd
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a
            
            if gcd(x, y) != 1:
                continue
                
            # Max k
            a_base = x * y * (x + y)
            if a_base > 180:
                continue
            max_k = 180 // a_base
            
            for k in range(1, max_k + 1):
                a = k * a_base
                b = k * x * (x + y)
                c = k * y * (x + y)
                if a <= 180 and b <= 180 and c <= 180:
                    # Ensure distinct
                    if len({a, b, c}) == 3:
                        triples.append(tuple(sorted((a, b, c))))
    
    # Remove duplicates
    triples = list(set(triples))
    
    # Identify active nodes
    active_nodes = set()
    for t in triples:
        active_nodes.update(t)
    
    # Decompose into connected components
    # Build graph where nodes are connected if they share a triple
    adj = {node: set() for node in active_nodes}
    for t in triples:
        for i in range(3):
            for j in range(i + 1, 3):
                adj[t[i]].add(t[j])
                adj[t[j]].add(t[i])
                
    visited = set()
    components = []
    for node in active_nodes:
        if node not in visited:
            # BFS
            comp = set()
            stack = [node]
            visited.add(node)
            while stack:
                n = stack.pop()
                comp.add(n)
                for neighbor in adj[n]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            components.append(comp)
            
    # Solve min hitting set for each component
    excluded = set()
    for comp in components:
        comp_nodes = list(comp)
        # Find triples within this component
        comp_triples = [t for t in triples if all(x in comp for x in t)]
        
        # Try k = 1, 2, ...
        found = False
        for k in range(1, len(comp_nodes) + 1):
            for subset in itertools.combinations(comp_nodes, k):
                # Check if subset hits all triples
                hit_all = True
                for t in comp_triples:
                    if not any(x in subset for x in t):
                        hit_all = False
                        break
                if hit_all:
                    excluded.update(subset)
                    found = True
                    break
            if found:
                break
            # Time check
            if time.time() - start_time > 40:
                break
        
    # Construct result
    result = [i for i in range(1, 181) if i not in excluded]
    
    return result

