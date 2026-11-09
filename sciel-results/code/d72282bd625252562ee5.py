"""Decompose graphs into few cycles and edges."""
import time
from random import Random

def construct():
    _start = time.time()
    _DEADLINE = 135.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj

    def has_ham_cycle(n, adj, rng, max_tries=20000):
        """Try random permutations to find a Hamiltonian cycle."""
        for _ in range(max_tries):
            perm = list(range(n))
            rng.shuffle(perm)
            ok = True
            for i in range(n):
                if perm[(i+1) % n] not in adj.get(perm[i], set()):
                    ok = False
                    break
            if ok:
                return perm
        return None

    def decompose_remaining(remaining_edges, adj, n, rng, max_iter=50000):
        """Decompose remaining edges into cycles using randomized DFS."""
        # Work with mutable edge set
        edge_set = set(remaining_edges)
        pieces = []
        
        for _ in range(max_iter):
            if not edge_set:
                break
            # Pick a random edge to start from
            start_edge = rng.choice(list(edge_set))
            u, v = start_edge
            # Try to find a cycle containing this edge
            # DFS from v, trying to get back to u
            found = _find_cycle_containing(u, v, edge_set, adj, rng, max_depth=n)
            if found:
                # Remove edges in the cycle
                for i in range(len(found)):
                    a, b = found[i], found[(i+1) % len(found)]
                    e = tuple(sorted((a, b)))
                    edge_set.discard(e)
                pieces.append(found)
            else:
                # Can't find a cycle through this edge; try another
                # If we keep failing, just emit single edges
                break
        
        # Remaining edges as single pieces
        for e in edge_set:
            pieces.append(list(e))
        return pieces

    def _find_cycle_containing(start, end, edge_set, adj, rng, max_depth):
        """Find a cycle that includes edge (start, end) via DFS."""
        # We need a path from end back to start using edges in edge_set
        # that doesn't reuse edges
        path = [end]
        visited_edges = {tuple(sorted((start, end)))}
        visited_nodes = {start, end}
        
        def dfs(node, depth):
            if depth >= max_depth:
                return None
            if node == start and len(path) >= 3:
                return list(path)
            neighbors = [nb for nb in adj.get(node, set()) 
                        if tuple(sorted((node, nb))) in edge_set 
                        and (node, nb) not in visited_edges]
            rng.shuffle(neighbors)
            for nb in neighbors:
                if nb == start:
                    if len(path) >= 3:
                        path.append(start)
                        return list(path)
                    continue
                if nb in visited_nodes:
                    continue
                visited_nodes.add(nb)
                visited_edges.add(tuple(sorted((node, nb))))
                path.append(nb)
                result = dfs(nb, depth + 1)
                if result:
                    return result
                path.pop()
                visited_nodes.discard(nb)
                visited_edges.discard(tuple(sorted((node, nb))))
            return None
        
        return dfs(end, 0)

    def solve_one(n, p, seed):
        edges = instance(n, p, seed)
        adj = build_adj(edges)
        rng = Random(seed * 1000 + 42)
        
        best_pieces = None
        best_count = len(edges)  # fallback: all single edges
        
        # Try to find Hamiltonian cycle
        ham = has_ham_cycle(n, adj, rng, max_tries=50000)
        
        if ham:
            # Hamiltonian cycle covers n edges
            ham_edges = set()
            for i in range(n):
                a, b = ham[i], ham[(i+1) % n]
                ham_edges.add(tuple(sorted((a, b))))
            
            remaining = [e for e in edges if e not in ham_edges]
            
            # Try multiple decompositions of remaining
            for attempt in range(20):
                if time.time() > _DEADLINE - 15:
                    break
                rng2 = Random(seed * 7 + attempt * 13 + 1)
                rem_pieces = decompose_remaining(remaining, adj, n, rng2, max_iter=30000)
                total = 1 + len(rem_pieces)  # 1 for ham cycle + remaining pieces
                if total < best_count:
                    best_count = total
                    best_pieces = [list(ham)] + rem_pieces
        
        # Also try without forcing Hamiltonian cycle - pure decomposition
        for attempt in range(10):
            if time.time() > _DEADLINE - 15:
                break
            rng2 = Random(seed * 13 + attempt * 31 + 7)
            pieces = decompose_remaining(edges, adj, n, rng2, max_iter=50000)
            if len(pieces) < best_count:
                best_count = len(pieces)
                best_pieces = pieces
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
        
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > _DEADLINE - 10:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        results.append(solve_one(*spec))
    
    return results
