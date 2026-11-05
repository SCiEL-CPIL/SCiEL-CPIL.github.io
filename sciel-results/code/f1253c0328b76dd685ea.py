"""Solve minimum transversal for maximal cliques of three fixed random graphs."""
import time
import random
import sys
from collections import Counter

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def bron_kerbosch(edges, n):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot: vertex in P∪X with max |P ∩ adj[u]|
            candidates = P | X
            pivot = max(candidates, key=lambda u: len(P & adj[u]))
            # Iterate over P \ adj[pivot]
            for v in list(P - adj[pivot]):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(frozenset(), set(range(n)), set())
        return cliques
    
    def greedy_hitting_set(cliques, n):
        """Greedy: repeatedly pick vertex covering most uncovered cliques."""
        # Map each clique to its vertex set
        uncovered = set(range(len(cliques)))
        # Count how many cliques each vertex covers
        vert_clique_count = [0] * n
        vert_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vert_clique_count[v] += 1
                vert_cliques[v].add(i)
        
        selected = []
        remaining_cliques = set(range(len(cliques)))
        
        while remaining_cliques:
            # Find vertex covering most remaining cliques
            best_v = -1
            best_count = 0
            for v in range(n):
                count = len(vert_cliques[v] & remaining_cliques)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1:
                break
            selected.append(best_v)
            remaining_cliques -= vert_cliques[best_v]
        
        return selected
    
    def local_search_improve(selected, cliques, n, max_iters=5000):
        """Try to improve by removing redundant vertices and k-exchange swaps."""
        selected = list(selected)
        
        # Precompute: for each vertex, which cliques it belongs to
        vert_to_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vert_to_cliques[v].add(i)
        
        def remove_redundant(vert_set):
            """Remove redundant vertices from vert_set."""
            vert_set = list(vert_set)
            changed = True
            while changed:
                changed = False
                for i, v in enumerate(vert_set):
                    remaining = set(vert_set)
                    remaining.discard(v)
                    # Check all cliques are still covered
                    all_covered = True
                    for clique in cliques:
                        if not (clique & remaining):
                            all_covered = False
                            break
                    if all_covered:
                        vert_set = [x for x in vert_set if x != v]
                        changed = True
                        break
            return vert_set
        
        # Remove redundant vertices
        selected = remove_redundant(selected)
        
        # k-exchange: try replacing k vertices with k-1 vertices
        improved = True
        while improved:
            improved = False
            
            # 1-for-0: try removing a single vertex (redundancy check)
            for v in selected:
                remaining = set(selected)
                remaining.discard(v)
                all_covered = True
                for clique in cliques:
                    if not (clique & remaining):
                        all_covered = False
                        break
                if all_covered:
                    selected = [x for x in selected if x != v]
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2-for-1: try replacing two vertices with one
            n_sel = len(selected)
            for i in range(n_sel):
                for j in range(i + 1, n_sel):
                    v1, v2 = selected[i], selected[j]
                    c1 = vert_to_cliques[v1]
                    c2 = vert_to_cliques[v2]
                    other_selected = set(selected) - {v1, v2}
                    
                    # Find cliques that need coverage from v1 or v2 (not covered by others)
                    needed = set()
                    for c_idx in c1:
                        if not (cliques[c_idx] & other_selected):
                            needed.add(c_idx)
                    for c_idx in c2:
                        if not (cliques[c_idx] & other_selected):
                            needed.add(c_idx)
                    
                    if not needed:
                        # Both are redundant
                        selected = [x for x in selected if x != v1 and x != v2]
                        improved = True
                        break
                    
                    # Find a vertex w that covers all needed cliques
                    best_w = -1
                    for w in range(n):
                        if w == v1 or w == v2:
                            continue
                        if vert_to_cliques[w] >= needed:
                            best_w = w
                            break
                    
                    if best_w != -1:
                        new_selected = [x for x in selected if x != v1 and x != v2]
                        new_selected.append(best_w)
                        selected = new_selected
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # 3-for-2: try replacing three vertices with two
            n_sel = len(selected)
            if n_sel >= 3:
                for i in range(n_sel):
                    for j in range(i + 1, n_sel):
                        for k in range(j + 1, n_sel):
                            v1, v2, v3 = selected[i], selected[j], selected[k]
                            c1 = vert_to_cliques[v1]
                            c2 = vert_to_cliques[v2]
                            c3 = vert_to_cliques[v3]
                            other_selected = set(selected) - {v1, v2, v3}
                            
                            # Find cliques that need coverage from v1, v2, or v3
                            needed = set()
                            for c_idx in c1 | c2 | c3:
                                if not (cliques[c_idx] & other_selected):
                                    needed.add(c_idx)
                            
                            if not needed:
                                selected = [x for x in selected if x != v1 and x != v2 and x != v3]
                                improved = True
                                break
                            
                            # Find two vertices w1, w2 that together cover all needed cliques
                            found = False
                            for w1 in range(n):
                                if w1 in (v1, v2, v3):
                                    continue
                                c_w1 = vert_to_cliques[w1]
                                remaining_needed = needed - c_w1
                                if not remaining_needed:
                                    # w1 alone covers everything, so we can use w1 + any other
                                    # But we need 2-for-3, so we add w1 and keep one of v1,v2,v3? No, we replace 3 with 2.
                                    # Actually if w1 covers all, we just need one more vertex (any non-conflicting)
                                    # But that would be 3-for-1. Let's handle 3-for-1 separately or just accept 3-for-2 where one is dummy
                                    # Better: if w1 covers all needed, we can pick w1 and any w2 != v1,v2,v3,w1
                                    # But we want to minimize, so let's try 3-for-1 first
                                    for w2 in range(n):
                                        if w2 in (v1, v2, v3, w1):
                                            continue
                                        new_selected = [x for x in selected if x != v1 and x != v2 and x != v3]
                                        new_selected.append(w1)
                                        new_selected.append(w2)
                                        # But this is 3-for-2, and w2 might be redundant
                                        # Let's just do 3-for-1 if possible
                                        selected = [x for x in selected if x != v1 and x != v2 and x != v3]
                                        selected.append(w1)
                                        improved = True
                                        found = True
                                        break
                                    if found:
                                        break
                                    continue
                                # Find w2 that covers remaining_needed
                                for w2 in range(n):
                                    if w2 in (v1, v2, v3, w1):
                                        continue
                                    if vert_to_cliques[w2] >= remaining_needed:
                                        new_selected = [x for x in selected if x != v1 and x != v2 and x != v3]
                                        new_selected.append(w1)
                                        new_selected.append(w2)
                                        selected = new_selected
                                        improved = True
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        # Final redundancy removal
        selected = remove_redundant(selected)
        
        return selected
    
    def random_restart_improve(selected, cliques, n, budget=5.0):
        """Try random perturbations to escape local optima."""
        start_time = time.time()
        best = list(selected)
        best_size = len(best)
        
        vert_to_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vert_to_cliques[v].add(i)
        
        def is_valid(vert_set):
            for clique in cliques:
                if not (clique & vert_set):
                    return False
            return True
        
        def remove_redundant(vert_set):
            vert_set = list(vert_set)
            changed = True
            while changed:
                changed = False
                for i, v in enumerate(vert_set):
                    remaining = set(vert_set)
                    remaining.discard(v)
                    if is_valid(remaining):
                        vert_set = [x for x in vert_set if x != v]
                        changed = True
                        break
            return vert_set
        
        def greedy_repair(candidate_set):
            """Add vertices to cover all uncovered cliques using greedy."""
            candidate_set = set(candidate_set)
            remaining = set(range(len(cliques)))
            for i, clique in enumerate(cliques):
                if clique & candidate_set:
                    remaining.discard(i)
            
            while remaining:
                # Find vertex covering most remaining cliques
                best_v = -1
                best_count = 0
                for v in range(n):
                    if v in candidate_set:
                        continue
                    count = len(vert_to_cliques[v] & remaining)
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_v == -1:
                    break
                candidate_set.add(best_v)
                remaining -= vert_to_cliques[best_v]
            
            return candidate_set
        
        while time.time() - start_time < budget:
            # Random perturbation: remove 1-3 random vertices, add 1-3 random vertices
            current = list(best)
            num_remove = random.randint(1, min(3, len(current)))
            num_add = random.randint(1, 3)
            
            to_remove = random.sample(current, num_remove)
            candidate = [v for v in current if v not in to_remove]
            
            # Add random vertices
            for _ in range(num_add):
                v = random.randint(0, n - 1)
                if v not in candidate:
                    candidate.append(v)
            
            # Repair: add vertices to cover uncovered cliques using greedy
            candidate_set = set(candidate)
            candidate_set = greedy_repair(candidate_set)
            
            # Remove redundant
            candidate = remove_redundant(list(candidate_set))
            
            if len(candidate) < best_size:
                best = candidate
                best_size = len(candidate)
        
        return best
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Emergency: return greedy solutions for remaining
            for remaining_n, remaining_p, remaining_seed in instances[instances.index((n, p, seed)):]:
                edges = generate_graph(remaining_n, remaining_p, remaining_seed)
                cliques = bron_kerbosch(edges, remaining_n)
                sol = greedy_hitting_set(cliques, remaining_n)
                results.append(sorted(sol))
            break
        
        edges = generate_graph(n, p, seed)
        cliques = bron_kerbosch(edges, n)
        
        # Initial greedy solution
        sol = greedy_hitting_set(cliques, n)
        
        # Local search improvement
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            sol = local_search_improve(sol, cliques, n)
        
        # Random restart improvement if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 15:
            budget = min(remaining_time - 5, 12.0)
            sol = random_restart_improve(sol, cliques, n, budget)
        
        # Second local search pass after random restarts
        remaining_time = deadline - time.time()
        if remaining_time > 8:
            sol = local_search_improve(sol, cliques, n)
        
        results.append(sorted(sol))
    
    return results
