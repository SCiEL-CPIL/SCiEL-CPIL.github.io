"""Path decomposition of fixed connected graphs via greedy path extension."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque
from itertools import combinations

def construct():
    _start_time = time.time()
    _DEADLINE = 130.0  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def odd_vertices(adj):
        odd = []
        for v in sorted(adj.keys()):
            if len(adj[v]) % 2 == 1:
                odd.append(v)
        return odd

    def greedy_path_decomposition(adj, odd_list, max_paths_hint=None):
        """
        Greedy: pair up odd vertices, extend each pair into a maximal path.
        Returns list of paths (each path is a list of vertices).
        """
        if not odd_list:
            # All even degrees: Eulerian circuit exists, but we need paths.
            # For all-even, we can have 0 paths only if no edges. Otherwise need at least 1.
            # Actually if all degrees are even and there are edges, we need at least 1 path? 
            # No: a cycle can be one path (closed). But our paths are open (vertex lists).
            # A cycle of length k is a path of k vertices where first and last are connected.
            # But "simple path" in graph theory usually means no repeated vertices.
            # A cycle is not a simple path unless we break it.
            # Let's handle: if no odd vertices, try to find one long path covering many edges.
            # Actually, if all degrees are even, minimum path cover can be 1 (if connected and has edges).
            # We'll just do a greedy walk.
            pass

        n_odd = len(odd_list)
        # Pair up odd vertices: (0,1), (2,3), ...
        pairs = []
        for i in range(0, n_odd - 1, 2):
            pairs.append((odd_list[i], odd_list[i + 1]))
        if n_odd % 2 == 1:
            # Shouldn't happen for connected graph (handshaking lemma), but handle
            pairs.append((odd_list[-1], odd_list[-1]))

        # Start with each pair as a 2-vertex path
        paths = [[a, b] for a, b in pairs]
        # Track which edges are used
        used_edges = set()
        for a, b in pairs:
            e = tuple(sorted((a, b)))
            if e in adj.get(a, set()) or (a, b) in adj.get(a, set()):
                used_edges.add(e)

        # Build edge set for quick lookup
        all_edges = set()
        for u in adj:
            for v in adj[u]:
                all_edges.add(tuple(sorted((u, v))))

        # Now extend paths greedily
        # For each path, try to extend from either end using unused edges
        changed = True
        max_iter = 10000
        iter_count = 0
        while changed and iter_count < max_iter:
            changed = False
            iter_count += 1
            for pi, path in enumerate(paths):
                if len(path) < 2:
                    continue
                # Try extending from the end
                end_v = path[-1]
                neighbors = [v for v in adj.get(end_v, set()) 
                             if tuple(sorted((end_v, v))) not in used_edges]
                if neighbors:
                    # Pick neighbor that leads to longest possible extension
                    best_next = None
                    best_len = 0
                    for nb in neighbors:
                        # Try a quick DFS to see how far we can go
                        trial_path = path + [nb]
                        trial_used = set(used_edges)
                        trial_used.add(tuple(sorted((end_v, nb))))
                        # Greedy extend from nb
                        cur = nb
                        while True:
                            cands = [w for w in adj.get(cur, set())
                                     if tuple(sorted((cur, w))) not in trial_used
                                     and w not in set(trial_path)]
                            if not cands:
                                break
                            # Pick one (arbitrary for now, could be smarter)
                            nxt = cands[0]
                            trial_path.append(nxt)
                            trial_used.add(tuple(sorted((cur, nxt))))
                            cur = nxt
                        if len(trial_path) > best_len:
                            best_len = len(trial_path)
                            best_next = nb
                            best_trial_path = trial_path
                            best_trial_used = trial_used
                    if best_next is not None and best_len > len(path):
                        # Commit
                        for i in range(len(path), len(best_trial_path) - 1):
                            used_edges.add(tuple(sorted((best_trial_path[i], best_trial_path[i+1]))))
                        # Also add the edge from old end to best_next
                        used_edges.add(tuple(sorted((end_v, best_next))))
                        paths[pi] = best_trial_path
                        changed = True
                        continue

                # Try extending from the start
                start_v = path[0]
                neighbors = [v for v in adj.get(start_v, set())
                             if tuple(sorted((start_v, v))) not in used_edges]
                if neighbors:
                    best_next = None
                    best_len = 0
                    for nb in neighbors:
                        trial_path = [nb] + path
                        trial_used = set(used_edges)
                        trial_used.add(tuple(sorted((start_v, nb))))
                        cur = nb
                        while True:
                            cands = [w for w in adj.get(cur, set())
                                     if tuple(sorted((cur, w))) not in trial_used
                                     and w not in set(trial_path)]
                            if not cands:
                                break
                            nxt = cands[0]
                            trial_path.insert(0, nxt)
                            trial_used.add(tuple(sorted((cur, nxt))))
                            cur = nxt
                        if len(trial_path) > best_len:
                            best_len = len(trial_path)
                            best_next = nb
                            best_trial_path = trial_path
                            best_trial_used = trial_used
                    if best_next is not None and best_len > len(path):
                        for i in range(1, len(best_trial_path)):
                            used_edges.add(tuple(sorted((best_trial_path[i-1], best_trial_path[i]))))
                        paths[pi] = best_trial_path
                        changed = True

        # Now handle remaining unused edges: add them as new paths
        remaining = [e for e in all_edges if e not in used_edges]
        for e in remaining:
            paths.append(list(e))
            used_edges.add(e)

        return paths

    def better_decomposition(adj, edges_list):
        """
        Multiple strategies, pick best.
        """
        odd = odd_vertices(adj)
        n_odd = len(odd)
        
        # Strategy 1: simple pairing
        paths1 = greedy_path_decomposition(adj, odd)
        score1 = len(paths1)
        
        # Strategy 2: try different pairings if n_odd >= 4
        best_paths = paths1
        best_score = score1
        
        if n_odd >= 4 and n_odd % 2 == 0:
            # Try a few alternative pairings
            import random as _rnd
            _rng = _rnd.Random(42)
            for attempt in range(5):
                shuffled = odd[:]
                _rng.shuffle(shuffled)
                paths_try = greedy_path_decomposition(adj, shuffled)
                if len(paths_try) < best_score:
                    best_score = len(paths_try)
                    best_paths = paths_try
        
        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > _DEADLINE - 5:
            # Fallback: single edges
            for s in spec:
                pass
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        adj = build_adjacency(edges)
        paths = better_decomposition(adj, edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
