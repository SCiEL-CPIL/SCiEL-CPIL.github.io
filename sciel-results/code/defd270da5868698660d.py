import time
import random
from itertools import combinations

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    def is_valid(edge_set, n):
        """Check if no simple three-edge path has endpoints of equal degree."""
        if not edge_set:
            return True
            
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        # For any edge (u, v), check paths s-u-v-t
        for (u, v) in edge_set:
            nu = [s for s in adj[u] if s != v]
            nv = [t for t in adj[v] if t != u]
            
            for s in nu:
                ds = deg[s]
                for t in nv:
                    if t == s:
                        continue
                    if deg[t] == ds:
                        return False
        return True

    # Known valid solution: K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    base_edges = [(a, b) for a in A for b in B]
    base_set = set(tuple(sorted(e)) for e in base_edges)
    best_set = set(base_set)
    best_count = len(best_set)
    
    # Generate all possible edges
    all_possible = []
    for i in range(n):
        for j in range(i + 1, n):
            all_possible.append((i, j))
    
    # Internal edges (not in base)
    internal_candidates = [e for e in all_possible if e not in base_set]
    
    # Strategy: Try to find a valid superset of base by adding internal edges
    # Use greedy with backtracking
    
    def try_add_edges(current_set, candidates, depth, max_depth):
        """Recursively try adding edges from candidates."""
        if depth >= max_depth or not candidates:
            return set(current_set)
        
        best = set(current_set)
        for idx, e in enumerate(candidates):
            e_norm = tuple(sorted(e))
            if e_norm in current_set:
                continue
            
            test_set = set(current_set)
            test_set.add(e_norm)
            
            if is_valid(test_set, n):
                # Success: try to add more
                remaining = candidates[:idx] + candidates[idx+1:]
                sub_best = try_add_edges(test_set, remaining, depth + 1, max_depth)
                if len(sub_best) > len(best):
                    best = sub_best
        
        return best
    
    # Greedy approach with multiple restarts
    def greedy_search(candidate_list, current_set):
        """Greedy: add edges that maintain validity."""
        result = set(current_set)
        added = True
        while added:
            added = False
            for e in candidate_list:
                e_norm = tuple(sorted(e))
                if e_norm in result:
                    continue
                test_set = set(result)
                test_set.add(e_norm)
                if is_valid(test_set, n):
                    result = test_set
                    added = True
                    break
        return result
    
    # Try different orderings of internal candidates
    for restart in range(30):
        if time.time() > deadline:
            break
        
        shuffled = list(internal_candidates)
        random.shuffle(shuffled)
        
        # Greedy add
        current = set(base_set)
        for e in shuffled:
            if time.time() > deadline:
                break
            e_norm = tuple(sorted(e))
            test_set = set(current)
            test_set.add(e_norm)
            if is_valid(test_set, n):
                current = test_set
        
        if len(current) > best_count:
            best_set = current
            best_count = len(current)
    
    # Try starting from scratch with different structures
    # Try K_{6,9}
    A2 = list(range(6))
    B2 = list(range(6, 15))
    base_edges2 = [(a, b) for a in A2 for b in B2]
    base_set2 = set(tuple(sorted(e)) for e in base_edges2)
    
    if is_valid(base_set2, n):
        internal2 = [e for e in all_possible if e not in base_set2]
        for restart in range(20):
            if time.time() > deadline:
                break
            shuffled = list(internal2)
            random.shuffle(shuffled)
            current = set(base_set2)
            for e in shuffled:
                if time.time() > deadline:
                    break
                e_norm = tuple(sorted(e))
                test_set = set(current)
                test_set.add(e_norm)
                if is_valid(test_set, n):
                    current = test_set
            if len(current) > best_count:
                best_set = current
                best_count = len(current)
    
    # Try K_{5,10}
    A3 = list(range(5))
    B3 = list(range(5, 15))
    base_edges3 = [(a, b) for a in A3 for b in B3]
    base_set3 = set(tuple(sorted(e)) for e in base_edges3)
    
    if is_valid(base_set3, n):
        internal3 = [e for e in all_possible if e not in base_set3]
        for restart in range(20):
            if time.time() > deadline:
                break
            shuffled = list(internal3)
            random.shuffle(shuffled)
            current = set(base_set3)
            for e in shuffled:
                if time.time() > deadline:
                    break
                e_norm = tuple(sorted(e))
                test_set = set(current)
                test_set.add(e_norm)
                if is_valid(test_set, n):
                    current = test_set
            if len(current) > best_count:
                best_set = current
                best_count = len(current)
    
    # Final validation
    if not is_valid(best_set, n):
        best_set = base_set
        best_count = len(base_set)

    result = [list(e) for e in best_set]
    result.sort()
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
    
    return result
