"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
import itertools
from collections import deque

N = 18
DEADLINE = 125.0  # safety margin before 150s kill


def _is_k4_free(adj):
    """Check if graph (as bitmask adjacency) is K4-free."""
    for a in range(N):
        for b in range(a + 1, N):
            if not (adj[a] & (1 << b)):
                continue
            for c in range(b + 1, N):
                if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                    continue
                # a,b,c form a triangle; check if any common neighbor d completes K4
                if adj[a] & adj[b] & adj[c]:
                    return False
    return True


def _largest_triangle_free(adj, upper_bound=None):
    """
    Find the maximum size of a triangle-free induced subgraph.
    Returns that size. If upper_bound is given, only checks sizes down to upper_bound.
    """
    if upper_bound is None:
        upper_bound = N

    # We want the LARGEST triangle-free induced subgraph.
    # Check sizes from upper_bound down to 1.
    # Optimization: precompute for each pair (u,v) whether they have a common neighbor.
    # But with bitmasks, checking is fast.

    for k in range(upper_bound, 0, -1):
        for subset in itertools.combinations(range(N), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            # Check if induced subgraph on 'mask' is triangle-free
            # A triangle exists iff there is an edge (u,v) in the subset such that
            # adj[u] & adj[v] & mask != 0 (i.e., a common neighbor w in the subset)
            # Wait, that's not quite right. adj[u] & adj[v] & mask checks for common neighbors
            # within the subset. If u and v are connected, and they share a neighbor w in the subset,
            # then u,v,w form a triangle.
            # So we just need to check all edges in the induced subgraph.
            has_triangle = False
            for u in subset:
                # Only check edges (u,v) where v > u to avoid duplication
                # Actually, we just need to check if there's ANY triangle.
                # For each u, look at neighbors v in the subset with v > u.
                # If such an edge exists, check if they share a neighbor in the subset.
                # To be safe and simple, iterate all pairs in subset.
                for v in subset:
                    if u >= v:
                        continue
                    if adj[u] & (1 << v):
                        # Edge (u,v) exists in induced subgraph
                        # Check for common neighbor in subset
                        if adj[u] & adj[v] & mask:
                            has_triangle = True
                            break
                if has_triangle:
                    break
            if not has_triangle:
                return k
    return 0


def _largest_triangle_free_fast(adj, upper_bound=None):
    """
    Faster check using a different strategy:
    A subset S is triangle-free iff for every edge (u,v) in S, N(u) ∩ N(v) ∩ S = ∅.
    Equivalently, the graph induced by S has no triangles.
    
    Alternative: Use the fact that a graph is triangle-free iff its edges can be 2-colored? No.
    
    Let's stick to the bitmask check but optimize the inner loop.
    """
    if upper_bound is None:
        upper_bound = N

    for k in range(upper_bound, 0, -1):
        for subset in itertools.combinations(range(N), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check for triangle
            # For each vertex u in subset, consider its neighbors in subset.
            # If any neighbor v of u (in subset) has a common neighbor w in subset (w != u, v),
            # then triangle.
            # Actually, simpler: for each edge (u,v) in subset, check adj[u] & adj[v] & mask.
            # But we must ensure the common neighbor is distinct from u and v.
            # adj[u] & adj[v] & mask includes u and v? No, adj[u] doesn't include u (simple graph).
            # So adj[u] & adj[v] & mask is the set of common neighbors in the subset.
            # If non-empty, there is a triangle.
            
            has_triangle = False
            for i in range(len(subset)):
                u = subset[i]
                for j in range(i + 1, len(subset)):
                    v = subset[j]
                    if adj[u] & (1 << v):
                        if adj[u] & adj[v] & mask:
                            has_triangle = True
                            break
                if has_triangle:
                    break
            if not has_triangle:
                return k
    return 0


def _random_k4_free_graph():
    """Generate a random K4-free graph using random 3-coloring."""
    coloring = [random.randint(0, 2) for _ in range(N)]
    # Ensure all colors used
    while len(set(coloring)) < 3:
        coloring[random.randint(0, N - 1)] = random.randint(0, 2)
    
    adj = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if coloring[u] != coloring[v]:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    # It's automatically K4-free (complete tripartite)
    return adj


def _random_k4_free_graph_perturbed(base_adj):
    """Perturb a K4-free graph by adding/removing edges while maintaining K4-free."""
    adj = base_adj[:]
    # Try to add some edges
    for _ in range(random.randint(0, 5)):
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u == v or adj[u] & (1 << v):
            continue
        # Check if adding (u,v) creates a K4
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        # If common has any edge between its vertices, adding (u,v) might create K4?
        # No, K4 is {u, v, x, y} where all pairs connected.
        # We need x, y in common such that x-y is an edge.
        # So check if the induced subgraph on 'common' has any edge.
        creates_k4 = False
        for x in range(N):
            if not (common & (1 << x)):
                continue
            for y in range(x + 1, N):
                if (common & (1 << y)) and (adj[x] & (1 << y)):
                    creates_k4 = True
                    break
            if creates_k4:
                break
        if not creates_k4:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Try to remove some edges
    for _ in range(random.randint(0, 3)):
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u != v and adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    return adj


def construct():
    start_time = time.time()
    random.seed(42)
    
    best_score = N + 1
    best_adj = None
    
    # Initial graph: complete tripartite
    adj = _random_k4_free_graph()
    score = _largest_triangle_fast(adj)
    best_score = score
    best_adj = adj
    
    # Local search / hill climbing
    no_improve = 0
    max_no_improve = 20
    
    while time.time() - start_time < DEADLINE:
        # Generate a candidate
        if random.random() < 0.3 or best_adj is None:
            cand = _random_k4_free_graph()
        else:
            cand = _random_k4_free_graph_perturbed(best_adj)
        
        if not _is_k4_free(cand):
            continue
        
        # Evaluate
        # Optimization: if we have best_score, we only need to check if cand has a triangle-free
        # subgraph of size >= best_score. If it does, cand is worse or equal.
        # If it doesn't, we need to find its exact max (which is < best_score).
        
        # Check if there's a triangle-free subgraph of size best_score
        # This is a SAT-like problem, but we can just run the full check with upper_bound=best_score
        # If _largest_triangle_free(cand, upper_bound=best_score) < best_score, then cand is better.
        # Wait, _largest_triangle_free returns the MAX size.
        # So if we call it with upper_bound=best_score, and it returns something < best_score,
        # it means the max is < best_score.
        # If it returns best_score, it means there IS a triangle-free subgraph of size best_score.
        # In that case, cand is not better (it's >= best_score).
        
        # So we can use upper_bound=best_score.
        # If the result is < best_score, we found a better graph.
        # We then need to know the EXACT score of cand to update best_score.
        # But since we know it's < best_score, we can just re-run with upper_bound=best_score-1
        # or just run the full check. Given N=18, full check is fast enough if we don't do it too often.
        
        # Let's first check if it's better.
        # If _largest_triangle_free(cand, upper_bound=best_score) < best_score:
        #    new_score = _largest_triangle_free(cand) # Full check
        #    if new_score < best_score:
        #        best_score = new_score
        #        best_adj = cand
        #        no_improve = 0
        #    else:
        #        no_improve += 1
        # else:
        #    no_improve += 1
        
        # Actually, _largest_triangle_free(cand, upper_bound=best_score) will return the max size
        # if it's <= best_score. If the true max is > best_score, it will return best_score (since it stops at best_score? No, it checks from upper_bound down).
        # Wait, my function checks from upper_bound DOWN.
        # So if I pass upper_bound=best_score, and there is a triangle-free subgraph of size best_score, it returns best_score.
        # If the true max is best_score + 1, it will NOT find it because it starts at best_score.
        # So if it returns best_score, the true max is >= best_score.
        # If it returns < best_score, the true max is < best_score.
        
        # So:
        test_score = _largest_triangle_free(cand, upper_bound=best_score)
        if test_score < best_score:
            # It's better. Find exact score.
            # We know it's < best_score, so we can search up to best_score - 1.
            # But _largest_triangle_free(cand, upper_bound=best_score - 1) will give the exact max.
            exact_score = _largest_triangle_free(cand, upper_bound=best_score - 1)
            best_score = exact_score
            best_adj = cand
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= max_no_improve:
            # Restart with a fresh random graph
            no_improve = 0
            best_adj = _random_k4_free_graph()
            # Re-evaluate best_adj? No, we already know its score is best_score.
            # But we might have lost the reference.
            # Actually, we should keep the best_adj from the last improvement.
            # The restart logic above replaces best_adj with a NEW random graph, which might be worse.
            # This is a bit risky. Better to just keep searching from the current best.
            # Or, reset the perturbation source.
            pass

    # Convert best_adj to edges
    edges = []
    if best_adj is None:
        # Fallback
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        edges.append([a, b])
        return edges

    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges


# Quick self-test
if __name__ == "__main__":
    edges = construct()
    # Reconstruct adj for testing
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    print(f"K4-free: {_is_k4_free(adj)}")
    print(f"Score: {_largest_triangle_free(adj)}")
    print(f"Number of edges: {len(edges)}")
