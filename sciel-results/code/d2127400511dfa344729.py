import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    node_set = set(valid_nodes)
    
    # Build conflict graph
    conflicts = {v: set() for v in valid_nodes}
    for i_idx, v in enumerate(valid_nodes):
        for j_idx in range(i_idx + 1, len(valid_nodes)):
            u = valid_nodes[j_idx]
            if v + u in squares:
                conflicts[v].add(u)
                conflicts[u].add(v)
    
    def greedy_mis(nodes, conf, order=None):
        """Greedy MIS: add nodes in given order if no conflict with already chosen."""
        if order is None:
            order = sorted(nodes, key=lambda x: len(conf[x]))
        chosen = set()
        for v in order:
            if v not in chosen:
                # Check if v conflicts with any already chosen
                conflict = False
                for c in conf[v]:
                    if c in chosen:
                        conflict = True
                        break
                if not conflict:
                    chosen.add(v)
        return chosen
    
    def local_search(mis, conf, deadline):
        """Try 1-swap and 2-swap improvements."""
        mis_set = set(mis)
        mis_list = list(mis_set)
        not_in_mis = [v for v in valid_nodes if v not in mis_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding a single node (shouldn't happen if greedy was maximal, but check)
            for v in not_in_mis:
                if all(c not in mis_set for c in conf[v]):
                    mis_set.add(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one from MIS, add one not in MIS
            for v in list(mis_set):
                for u in not_in_mis:
                    if u == v:
                        continue
                    # Check if u conflicts with any in mis_set except v
                    ok = True
                    for c in conf[u]:
                        if c in mis_set and c != v:
                            ok = False
                            break
                    if ok:
                        mis_set.remove(v)
                        mis_set.add(u)
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                not_in_mis = [v for v in valid_nodes if v not in mis_set]
                continue
            
            # Try 2-swap: remove two from MIS, add two not in MIS
            mis_list = list(mis_set)
            for i in range(len(mis_list)):
                if time.time() >= deadline:
                    break
                v1 = mis_list[i]
                for j in range(i + 1, len(mis_list)):
                    if time.time() >= deadline:
                        break
                    v2 = mis_list[j]
                    for u1 in not_in_mis:
                        if time.time() >= deadline:
                            break
                        if u1 == v1 or u1 == v2:
                            continue
                        for u2 in not_in_mis:
                            if u2 <= u1:
                                continue
                            if u2 == v1 or u2 == v2:
                                continue
                            # Check u1, u2 don't conflict with mis_set minus {v1,v2}
                            ok = True
                            for u in [u1, u2]:
                                for c in conf[u]:
                                    if c in mis_set and c != v1 and c != v2:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                continue
                            # Check u1, u2 don't conflict with each other
                            if u1 in conf[u2]:
                                continue
                            # Valid swap
                            mis_set.remove(v1)
                            mis_set.remove(v2)
                            mis_set.add(u1)
                            mis_set.add(u2)
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    not_in_mis = [v for v in valid_nodes if v not in mis_set]
                    break
        
        return mis_set
    
    # Multiple strategies
    best_mis = set()
    
    # Strategy 1: Greedy by degree
    mis1 = greedy_mis(valid_nodes, conflicts)
    mis1 = local_search(mis1, conflicts, deadline)
    if len(mis1) > len(best_mis):
        best_mis = mis1
    
    # Strategy 2: Greedy by random order (multiple times)
    for attempt in range(50):
        if time.time() >= deadline:
            break
        random.shuffle(valid_nodes)
        mis2 = greedy_mis(valid_nodes, conflicts, order=valid_nodes)
        mis2 = local_search(mis2, conflicts, deadline)
        if len(mis2) > len(best_mis):
            best_mis = mis2
    
    # Strategy 3: Start with all valid, greedily remove highest degree
    if time.time() < deadline:
        current = set(valid_nodes)
        while True:
            if time.time() >= deadline:
                break
            # Find node with highest degree in current
            max_deg = -1
            max_node = None
            for v in current:
                deg = len([c for c in conflicts[v] if c in current])
                if deg > max_deg:
                    max_deg = deg
                    max_node = v
            if max_node is None or max_deg == 0:
                break
            current.remove(max_node)
        mis3 = current
        mis3 = local_search(mis3, conflicts, deadline)
        if len(mis3) > len(best_mis):
            best_mis = mis3
    
    return list(best_mis)
