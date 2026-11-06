import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe, limit is 150s

    def instance(n, p, seed):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex from P ∪ X with max |N(u) ∩ P|
            union_px = P | X
            if union_px:
                pivot = max(union_px, key=lambda u: len(adj[u] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P = P - {v}
                X = X | {v}
        
        bk(frozenset(), set(range(n)), frozenset())
        return cliques

    def min_hitting_set(cliques, n, time_limit):
        """Find minimum set of vertices that hits every clique in cliques."""
        if not cliques:
            return []
        
        # Convert cliques to frozensets for fast intersection
        cliques = [frozenset(c) for c in cliques]
        m = len(cliques)
        
        # Precompute: for each vertex, which cliques it hits
        vertex_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].append(i)
        
        # Greedy upper bound
        temp_remaining = set(range(m))
        greedy_cover = []
        
        while temp_remaining:
            # Find vertex that hits most remaining cliques
            best_v = None
            best_count = -1
            for v in range(n):
                count = sum(1 for ci in vertex_cliques[v] if ci in temp_remaining)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count == 0:
                break
                
            greedy_cover.append(best_v)
            for ci in vertex_cliques[best_v]:
                temp_remaining.discard(ci)
        
        best_cover = list(greedy_cover)
        best_size = len(greedy_cover)
        
        if best_size == 0:
            return []
        
        # Local search: try 1-swap improvements to reduce upper bound
        def try_improve(cover):
            """Try to improve cover by removing vertices and re-adding optimally."""
            improved = True
            while improved:
                improved = False
                for i, v in enumerate(list(cover)):
                    # Try removing v
                    new_cover = cover[:i] + cover[i+1:]
                    # Find uncovered cliques
                    covered_set = set()
                    for cv in new_cover:
                        covered_set.update(vertex_cliques[cv])
                    uncovered = set(range(m)) - covered_set
                    if not uncovered:
                        # v was redundant
                        return new_cover
                    # Try to add back one vertex that covers most uncovered
                    best_add = None
                    best_add_count = -1
                    for av in range(n):
                        count = sum(1 for ci in vertex_cliques[av] if ci in uncovered)
                        if count > best_add_count:
                            best_add_count = count
                            best_add = av
                    if best_add is not None and best_add_count > 0:
                        new_cover2 = new_cover + [best_add]
                        if len(new_cover2) < len(cover):
                            return new_cover2
            return cover
        
        # Apply local search
        for _ in range(10):
            improved_cover = try_improve(best_cover)
            if len(improved_cover) < len(best_cover):
                best_cover = improved_cover
                best_size = len(best_cover)
            else:
                break
        
        # Fast lower bound: packing bound using disjoint cliques
        def fast_lower_bound(remaining_clique_set):
            """Compute a lower bound on the number of additional vertices needed."""
            if not remaining_clique_set:
                return 0
            
            remaining = set(remaining_clique_set)
            lb = 0
            # Greedy packing: repeatedly pick a clique, add 1 to lb, remove all cliques sharing vertices
            while remaining:
                # Pick the smallest clique for better packing
                min_ci = min(remaining, key=lambda i: len(cliques[i]))
                lb += 1
                # Remove all cliques that share any vertex with min_ci
                min_clique = cliques[min_ci]
                to_remove = set()
                for ci in remaining:
                    if cliques[ci] & min_clique:
                        to_remove.add(ci)
                remaining -= to_remove
            
            return lb
        
        # Branch and bound
        def bbs(remaining_clique_set, current_size, current_cover):
            nonlocal best_size, best_cover
            
            if time.time() > deadline:
                return
            
            if not remaining_clique_set:
                if current_size < best_size:
                    best_size = current_size
                    best_cover = list(current_cover)
                return
            
            # Lower bound
            lb = current_size + fast_lower_bound(remaining_clique_set)
            if lb >= best_size:
                return
            
            # Choose a clique to branch on: smallest remaining clique
            min_clique_idx = min(remaining_clique_set, key=lambda i: len(cliques[i]))
            branch_clique = cliques[min_clique_idx]
            
            # Branch: for each vertex in branch_clique, try adding it
            # Order vertices by how many remaining cliques they cover (most first)
            vertex_scores = []
            for v in branch_clique:
                score = sum(1 for ci in vertex_cliques[v] if ci in remaining_clique_set)
                vertex_scores.append((score, v))
            vertex_scores.sort(reverse=True)
            
            for score, v in vertex_scores:
                new_remaining = set(remaining_clique_set)
                for ci in vertex_cliques[v]:
                    new_remaining.discard(ci)
                bbs(new_remaining, current_size + 1, current_cover + [v])
                
                if time.time() > deadline:
                    return
        
        # Start B&B
        initial_remaining = set(range(m))
        bbs(initial_remaining, 0, [])
        
        return best_cover

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = instance(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        cover = min_hitting_set(cliques, n, deadline)
        out.append(sorted(cover))
        
    return out
