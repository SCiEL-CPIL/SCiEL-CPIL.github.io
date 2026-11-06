"""MODEL-AND-OPTIMIZE: parameterize the answer as a set of cliques; fit the
parameters (which cliques to select) by optimizing directly against the
objective (minimize number of cliques, keep every clique a valid edge-partition)."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations

def _instances():
    from random import Random
    def instance(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return [instance(n, p, seed) for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602))]

def _adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_maximal_cliques(adj, n):
    # Bron-Kerbosch with pivot
    cliques = []
    def bronk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot u from P U X
        if P:
            u = next(iter(P))
        elif X:
            u = next(iter(X))
        else:
            return
        candidates = [v for v in P if v not in adj[u]]
        for v in candidates:
            bronk(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}
    
    bronk(set(), set(range(n)), set())
    return cliques

def _validate(partition, adj, edgeset):
    seen = set()
    for c in partition:
        if len(c) < 2:
            return False
        # Check clique
        cl = list(c)
        for i in range(len(cl)):
            a = cl[i]
            for b in cl[i+1:]:
                if b not in adj[a]:
                    return False
        # Check edges
        for (a, b) in combinations(c, 2):
            e = (min(a, b), max(a, b))
            if e not in edgeset:
                return False
            if e in seen:
                return False
            seen.add(e)
    if len(seen) != len(edgeset):
        return False
    return True

def _solve_one(edges, n, deadline):
    adj = _adjacency(edges, n)
    edgeset = set(edges)
    
    # Find all maximal cliques
    maximal_cliques = _find_maximal_cliques(adj, n)
    
    # Filter out cliques of size < 2
    maximal_cliques = [c for c in maximal_cliques if len(c) >= 2]
    
    # Calculate weight (savings) for each clique
    # Weight = |E(C)| - 1
    weights = {}
    for c in maximal_cliques:
        sz = len(c)
        weights[c] = sz * (sz - 1) // 2 - 1
        
    # Precompute edges for each clique
    clique_edges = {}
    for c in maximal_cliques:
        cl = list(c)
        es = set()
        for i in range(len(cl)):
            for j in range(i+1, len(cl)):
                es.add((min(cl[i], cl[j]), max(cl[i], cl[j])))
        clique_edges[c] = es

    # Greedy selection: sort by weight descending
    sorted_cliques = sorted(maximal_cliques, key=lambda c: weights[c], reverse=True)
    
    selected = []
    used_edges = set()
    
    for c in sorted_cliques:
        es = clique_edges[c]
        if es & used_edges:
            continue
        selected.append(c)
        used_edges |= es
        
    # Local Search: try to improve selection
    # If we replace a set of selected cliques S_conf with a new clique C, 
    # and weight(C) > sum(weight(C') for C' in S_conf), we improve.
    
    # To make this efficient, we can iterate:
    # For each non-selected clique C, find selected cliques that conflict with it.
    # If sum(weights of conflicting) < weight(C), replace them.
    
    # We need a way to quickly find conflicting selected cliques.
    # Map edge -> selected clique index?
    # Since selected cliques are edge-disjoint, each edge belongs to at most one selected clique.
    edge_to_clique_idx = {}
    for idx, c in enumerate(selected):
        for e in clique_edges[c]:
            edge_to_clique_idx[e] = idx
            
    improved = True
    while improved and time.time() < deadline - 0.1:
        improved = False
        # Sort non-selected cliques by weight descending to try best improvements first
        non_selected = [c for c in maximal_cliques if c not in selected]
        non_selected.sort(key=lambda c: weights[c], reverse=True)
        
        for C in non_selected:
            if time.time() > deadline - 0.1:
                break
            es_C = clique_edges[C]
            
            # Find conflicting selected cliques
            conflicting_indices = set()
            for e in es_C:
                if e in edge_to_clique_idx:
                    conflicting_indices.add(edge_to_clique_idx[e])
            
            if not conflicting_indices:
                # C is edge-disjoint from all selected, add it
                selected.append(C)
                for e in es_C:
                    edge_to_clique_idx[e] = len(selected) - 1
                improved = True
            else:
                # Calculate sum of weights of conflicting cliques
                sum_conf_weights = sum(weights[selected[i]] for i in conflicting_indices)
                if weights[C] > sum_conf_weights:
                    # Replace conflicting cliques with C
                    # Remove conflicting cliques
                    # We need to remove them from 'selected' and update edge_to_clique_idx
                    # This is tricky with indices. Let's reconstruct 'selected' or use a set.
                    
                    # Let's use a set of frozensets for selected to make removal easy, 
                    # but we need to maintain order? No, order doesn't matter for validity.
                    # But we need to update edge_to_clique_idx.
                    
                    # Better approach: 
                    # 1. Identify conflicting cliques (objects).
                    # 2. Remove them from selected.
                    # 3. Add C.
                    # 4. Rebuild edge_to_clique_idx.
                    
                    conf_cliques = [selected[i] for i in conflicting_indices]
                    
                    # Check if replacement is strictly better
                    # We already checked weights.
                    
                    # Remove conf_cliques from selected
                    new_selected = []
                    for c in selected:
                        if c not in conf_cliques:
                            new_selected.append(c)
                    new_selected.append(C)
                    
                    # Rebuild edge_to_clique_idx
                    edge_to_clique_idx = {}
                    for idx, c in enumerate(new_selected):
                        for e in clique_edges[c]:
                            edge_to_clique_idx[e] = idx
                            
                    selected = new_selected
                    improved = True
                    
    # Construct final partition
    partition = []
    for c in selected:
        partition.append(list(c))
        
    # Add remaining edges as 2-cliques
    for e in edgeset:
        if e not in edge_to_clique_idx:
            partition.append(list(e))
            
    # Validate
    if not _validate(partition, adj, edgeset):
        return [list(e) for e in edgeset]
        
    return partition

def construct():
    start = time.time()
    DEADLINE = 145.0
    results = []
    try:
        insts = _instances()
        for idx, edges in enumerate(insts):
            n = 64 if idx == 1 else 32
            remaining = DEADLINE - (time.time() - start)
            if remaining <= 1.0:
                results.append([list(e) for e in edges])
                continue
            part = _solve_one(edges, n, DEADLINE - 0.1)
            results.append(part)
        # Final validation
        ok = True
        for idx, part in enumerate(results):
            edges = insts[idx]
            n = 64 if idx == 1 else 32
            adj = _adjacency(edges, n)
            edgeset = set(edges)
            if not _validate(part, adj, edgeset):
                ok = False
                break
        if not ok:
            results = [[list(e) for e in edges] for edges in insts]
        return results
    except Exception:
        try:
            insts = _instances()
            return [[list(e) for e in edges] for edges in insts]
        except Exception:
            return [[], []]
# EVOLVE-BLOCK-END
