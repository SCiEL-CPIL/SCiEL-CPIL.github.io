import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    def is_2_colorable(n, edges):
        # Fast checker
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        color = [0] * n
        unassigned = (1 << n) - 1
        
        def propagate():
            nonlocal unassigned
            changed = True
            while changed:
                changed = False
                for mask in edge_masks:
                    red = 0
                    blue = 0
                    un_v = []
                    for v in range(n):
                        if mask & (1 << v):
                            if color[v] == 1: red += 1
                            elif color[v] == -1: blue += 1
                            else: un_v.append(v)
                    
                    if red == 5 or blue == 5: return False
                    if red == 4 and un_v:
                        for v in un_v:
                            if color[v] == 0:
                                color[v] = -1
                                unassigned &= ~(1 << v)
                                changed = True
                            elif color[v] != -1: return False
                    elif blue == 4 and un_v:
                        for v in un_v:
                            if color[v] == 0:
                                color[v] = 1
                                unassigned &= ~(1 << v)
                                changed = True
                            elif color[v] != 1: return False
            return True

        def backtrack():
            nonlocal unassigned
            if unassigned == 0: return True
            
            # Pick vertex
            v = -1
            for i in range(n):
                if unassigned & (1 << i):
                    v = i
                    break
            
            # Try 1
            color[v] = 1
            old_un = unassigned
            unassigned &= ~(1 << v)
            if propagate():
                if backtrack(): return True
            
            # Try -1
            color[v] = -1
            unassigned = old_un & ~(1 << v)
            if propagate():
                if backtrack(): return True
            
            color[v] = 0
            unassigned = old_un
            return False

        if not propagate(): return False
        return backtrack()

    best_edges = None
    best_n = 9
    
    random.seed(123)
    
    # Try to find a small hypergraph
    for attempt in range(20):
        if time.time() > deadline: break
        
        n = random.randint(18, 24)
        all_edges = list(itertools.combinations(range(n), 5))
        random.shuffle(all_edges)
        
        edges_set = set()
        found = False
        for e in all_edges:
            if time.time() > deadline: break
            edges_set.add(e)
            # Check periodically
            if len(edges_set) % 10 == 0:
                if not is_2_colorable(n, list(edges_set)):
                    found = True
                    break
        
        if found:
            current_edges = list(edges_set)
            # Minimize
            i = 0
            while i < len(current_edges):
                if time.time() > deadline: break
                test_edges = current_edges[:i] + current_edges[i+1:]
                if not is_2_colorable(n, test_edges):
                    current_edges = test_edges
                else:
                    i += 1
            
            if best_edges is None or len(current_edges) < len(best_edges):
                best_edges = current_edges
                best_n = n

    if best_edges is not None:
        return {'n': best_n, 'edges': [list(e) for e in best_edges]}
    else:
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
