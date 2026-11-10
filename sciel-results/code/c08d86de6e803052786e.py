"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def get_degrees(es, n):
        deg = [0] * n
        for (u, v) in es:
            deg[u] += 1
            deg[v] += 1
        return deg

    def is_valid(es, n):
        """Check if no simple three-edge path has endpoints of equal degree."""
        if not es:
            return True
        deg = get_degrees(es, n)
        adj = [set() for _ in range(n)]
        for (u, v) in es:
            adj[u].add(v)
            adj[v].add(u)
        
        for (u, v) in es:
            nu = adj[u] - {v}
            nv = adj[v] - {u}
            if not nu or not nv:
                continue
            nv_deg = {}
            for t in nv:
                d = deg[t]
                nv_deg[d] = nv_deg.get(d, 0) + 1
            for s in nu:
                ds = deg[s]
                if ds in nv_deg:
                    cnt = nv_deg[ds]
                    if s in nv:
                        if cnt > 1:
                            return False
                    else:
                        return False
        return True

    # Start with K_{7,8}: A={0..6}, B={7..14}
    A = list(range(7))
    B = list(range(7, 15))
    
    edge_set = set()
    for a in A:
        for b in B:
            edge_set.add((a, b))
    
    assert is_valid(edge_set, n), "Base K_{7,8} should be valid"
    
    best_edges = set(edge_set)
    best_count = len(best_edges)
    
    # All possible edges in K_15
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local search: try 1-opt moves (remove one edge, add another)
    # and 2-opt moves (remove two, add two)
    
    def try_improve(current_edges, max_time):
        """Try to improve current_edges via local search."""
        current = set(current_edges)
        count = len(current)
        
        # 1-opt: remove edge e1, add edge e2
        current_list = list(current)
        missing = [e for e in all_edges if e not in current]
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for e1 in list(current):
                if time.time() >= max_time:
                    break
                for e2 in missing:
                    if time.time() >= max_time:
                        break
                    test = set(current)
                    test.discard(e1)
                    test.add(e2)
                    if len(test) > count and is_valid(test, n):
                        current = test
                        count = len(current)
                        missing = [e for e in all_edges if e not in current]
                        improved = True
                        break
                if improved:
                    break
        
        # 2-opt: remove two edges, add two edges
        improved2 = True
        while improved2 and time.time() < max_time:
            improved2 = False
            current_list = list(current)
            missing = [e for e in all_edges if e not in current]
            for i in range(len(current_list)):
                if time.time() >= max_time:
                    break
                e1 = current_list[i]
                for j in range(i+1, len(current_list)):
                    if time.time() >= max_time:
                        break
                    e2 = current_list[j]
                    for k in range(len(missing)):
                        if time.time() >= max_time:
                            break
                        e3 = missing[k]
                        for l in range(k+1, len(missing)):
                            if time.time() >= max_time:
                                break
                            e4 = missing[l]
                            test = set(current)
                            test.discard(e1)
                            test.discard(e2)
                            test.add(e3)
                            test.add(e4)
                            if len(test) > count and is_valid(test, n):
                                current = test
                                count = len(current)
                                improved2 = True
                                break
                        if improved2:
                            break
                    if improved2:
                        break
                if improved2:
                    break
        
        return current, count
    
    # Try local search from K_{7,8}
    if time.time() < deadline - 5:
        improved_edges, improved_count = try_improve(best_edges, deadline - 5)
        if improved_count > best_count:
            best_edges = improved_edges
            best_count = improved_count
    
    # Also try different initial partitions
    # K_{6,9}, K_{5,10}, K_{4,11}, K_{3,12}
    for split in [6, 5, 4, 3]:
        if time.time() > deadline - 10:
            break
        A2 = list(range(split))
        B2 = list(range(split, 15))
        es2 = set()
        for a in A2:
            for b in B2:
                es2.add((a, b))
        
        if is_valid(es2, n):
            if len(es2) > best_count:
                best_edges = es2
                best_count = len(es2)
            
            # Try to improve this
            if time.time() < deadline - 5:
                imp_edges, imp_count = try_improve(es2, deadline - 5)
                if imp_count > best_count:
                    best_edges = imp_edges
                    best_count = imp_count
    
    # Convert to required format
    result = [list(e) for e in best_edges]
    result.sort()
    
    # Ensure types are Python int
    for i, edge in enumerate(result):
        result[i] = [int(edge[0]), int(edge[1])]
    
    return result
# EVOLVE-BLOCK-END
