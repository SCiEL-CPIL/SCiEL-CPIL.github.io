"""Packing trees T1..T20 into K20 using a model-and-optimize approach."""
import sys
import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


def construct():
    """
    Embed trees T1..T20 into K20 with edge-disjoint embeddings.
    Returns a list of 20 entries; entry k-1 is None or a list of k distinct ints in 0..19.
    Position i is the image of vertex i of T_k.
    """
    deadline = time.time() + 145  # safety margin before 150s limit
    trees = instance_trees(20, 611)
    
    def solve_one_restart(rng, order):
        n = 20
        used_edges = set()
        result = [None] * 20
        
        for k in order:
            edges_k = trees[k-1]
            if not edges_k:
                result[k-1] = [0]
                continue
            if k == 2:
                found = False
                for _ in range(5000):
                    a = rng.randint(0, 19)
                    b = rng.randint(0, 19)
                    if a != b:
                        key = (min(a,b), max(a,b))
                        if key not in used_edges:
                            result[k-1] = [a, b]
                            used_edges.add(key)
                            found = True
                            break
                if not found:
                    for a in range(20):
                        for b in range(a+1, 20):
                            key = (a, b)
                            if key not in used_edges:
                                result[k-1] = [a, b]
                                used_edges.add(key)
                                found = True
                                break
                        if found:
                            break
                if not found:
                    return None
                continue
            
            # Build adjacency
            adj = [[] for _ in range(k)]
            for (u, v) in edges_k:
                adj[u].append(v)
                adj[v].append(u)
            
            best_attempt = None
            
            # Try BFS-based embedding with multiple attempts
            for attempt in range(300):
                mapping = [None] * k
                mapping[0] = rng.randint(0, 19)
                
                visited = [False] * k
                visited[0] = True
                queue = [0]
                valid_tree = True
                used_hosts = {mapping[0]}
                
                while queue:
                    node = queue.pop(0)
                    for neighbor in adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            candidates = []
                            for h in range(20):
                                if h in used_hosts:
                                    continue
                                edge_key = (min(mapping[node], h), max(mapping[node], h))
                                if edge_key not in used_edges:
                                    candidates.append(h)
                            if candidates:
                                # Prefer hosts with fewer used edges (more flexibility)
                                rng.shuffle(candidates)
                                mapping[neighbor] = candidates[0]
                                used_hosts.add(mapping[neighbor])
                                queue.append(neighbor)
                            else:
                                valid_tree = False
                                break
                    if not valid_tree:
                        break
                
                if valid_tree and all(mapping[i] is not None for i in range(k)):
                    # Final verification
                    ok = True
                    for (u, v) in edges_k:
                        if (min(mapping[u], mapping[v]), max(mapping[u], mapping[v])) in used_edges:
                            ok = False
                            break
                    if ok:
                        best_attempt = mapping
                        break
            
            if best_attempt is None:
                return None
            
            result[k-1] = list(best_attempt)
            for (u, v) in edges_k:
                used_edges.add((min(best_attempt[u], best_attempt[v]), max(best_attempt[u], best_attempt[v])))
        
        return result
    
    # Generate different orderings to try
    orderings = []
    # Largest first
    orderings.append(list(range(20, 0, -1)))
    # Smallest first
    orderings.append(list(range(1, 21)))
    # Alternating: large, small, second large, second small, etc.
    alt = []
    left, right = 0, 19
    while left <= right:
        if left == right:
            alt.append(left + 1)
            break
        alt.append(left + 1)
        alt.append(right + 1)
        left += 1
        right -= 1
    orderings.append(alt)
    
    # Multiple restarts
    best_solution = None
    best_value = -1
    max_possible = sum(len(trees[k-1]) for k in range(1, 21))
    
    restart = 0
    while time.time() < deadline:
        if time.time() > deadline:
            break
        order_idx = restart % len(orderings)
        order = orderings[order_idx]
        # Add some randomness to the order
        rng = Random(restart * 7919 + 13)
        
        # Slightly perturb the order for diversity
        if rng.random() < 0.3 and len(order) > 2:
            i = rng.randint(1, len(order) - 2)
            order[i], order[i+1] = order[i+1], order[i]
        
        solution = solve_one_restart(rng, order)
        if solution is not None:
            val = sum(len(trees[k-1]) for k in range(1, 21) if solution[k-1] is not None)
            if val > best_value:
                best_value = val
                best_solution = [list(s) if s is not None else None for s in solution]
        
        if best_value >= max_possible:
            break
        restart += 1
    
    if best_solution is None:
        return [None] * 19 + [list(range(20))]
    
    return best_solution


if __name__ == "__main__":
    res = construct()
    trees = instance_trees(20, 611)
    used_edges = set()
    ok = True
    for k in range(1, 21):
        if res[k-1] is not None:
            r = res[k-1]
            if len(r) != k:
                print(f"T{k}: wrong length {len(r)}")
                ok = False
            if len(set(r)) != k:
                print(f"T{k}: not distinct")
                ok = False
            for v in r:
                if not isinstance(v, int) or v < 0 or v > 19:
                    print(f"T{k}: invalid vertex {v}")
                    ok = False
            for (u, v) in trees[k-1]:
                eu, ev = r[u], r[v]
                key = (min(eu, ev), max(eu, ev))
                if key in used_edges:
                    print(f"T{k}: edge conflict {key}")
                    ok = False
                used_edges.add(key)
    total = sum(len(trees[k-1]) for k in range(1, 21) if res[k-1] is not None)
    print(f"Valid: {ok}, Total edges: {total}")
