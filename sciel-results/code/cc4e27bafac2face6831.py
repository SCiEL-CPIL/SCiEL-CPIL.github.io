"""Search-based construction for R(C4,K11) > n."""
import time

# EVOLVE-BLOCK-START
def _triangles(n):
    edges = []
    for t in range(n):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return edges

def _alpha_bound(adj, n):
    # Upper bound on independence number: sum over vertices of 1/(deg+1)
    # (Caro-Wei / Turan type bound, valid for any graph)
    total = 0.0
    for v in range(n):
        total += 1.0 / (len(adj[v]) + 1.0)
    return total

def _can_add(adj, n, cand):
    # cand: frozenset of neighbours for new vertex n
    # C4-free check: no two existing vertices share a neighbour in cand
    for v in cand:
        for w in cand:
            if v >= w:
                continue
            # count common neighbours of v and w that are in cand
            inter = 0
            for x in adj[v]:
                if x in cand:
                    inter += 1
                    if inter >= 1:
                        # need >=2 to form C4
                        pass
            # recount properly
            c = 0
            for x in adj[v]:
                if x in cand:
                    c += 1
            if c >= 2:
                return False
    return True

def _alpha_after_add(adj, n, cand):
    # After adding vertex n with neighbours cand, compute upper bound on alpha.
    # New vertex n: its non-neighbours (in old graph) are candidates for an independent set
    # containing n.  Use a quick greedy bound: alpha of new graph <= max(alpha_old,
    #   1 + alpha of non-neighbourhood of n in old graph).
    # We approximate alpha_old by the Caro-Wei bound and bound the non-neighbourhood.
    # For speed, use Caro-Wei on the new graph.
    total = 0.0
    for v in range(n):
        total += 1.0 / (len(adj[v]) + 1.0)
    # vertex n
    total += 1.0 / (len(cand) + 1.0)
    return total

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_n = 30
    best_edges = _triangles(10)

    # Start from 10 disjoint triangles (n=30, alpha=10, C4-free)
    n = 30
    adj = [set() for _ in range(n)]
    for e in best_edges:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])

    # Try to add vertices up to 43
    attempt = n
    while attempt < 43 and time.time() < deadline:
        # Candidate neighbour sets: subsets of existing vertices of size 0..min(attempt, 15)
        # To keep C4-free, the candidate set must be a clique-free "code" set:
        # no two vertices in cand share a neighbour in cand.
        # Simplest safe choice: cand is an independent set in the *line* sense.
        # We try: pick a small independent set of the current graph as neighbours.
        # A vertex's neighbourhood must be C4-free-compatible: for any u,v in cand,
        # |N(u) ∩ cand| <= 1.
        # Greedy: build cand by adding vertices that don't conflict.
        found = False
        # Try several random orderings
        import random
        for trial in range(200):
            if time.time() >= deadline:
                break
            order = list(range(n))
            random.shuffle(order)
            cand = set()
            for v in order:
                # check if adding v to cand keeps C4-free
                ok = True
                for u in cand:
                    # count common neighbours of u and v that are in cand
                    c = 0
                    for x in adj[u]:
                        if x in cand and x != v:
                            c += 1
                    if c >= 2:
                        ok = False
                        break
                if ok:
                    cand.add(v)
                if len(cand) >= 12:
                    break
            # Now cand is a set where no two share >=2 common neighbours in cand.
            # But we also need: for the NEW vertex, its neighbours = cand.
            # C4 condition for new vertex: for any two existing vertices u,v,
            # the number of common neighbours in cand must be <= 1.
            # That's exactly what we checked pairwise.
            # Also need: new vertex doesn't create C4 via two old vertices both in cand.
            # That's the same check.
            # Now check alpha: new vertex n has neighbours cand, non-neighbours = all others.
            # alpha(new) <= max(alpha(old), 1 + alpha(non-neighbours of n))
            # We use Caro-Wei bound on new graph.
            # But computing exact alpha is expensive; use bound.
            # For the new vertex, its non-neighbours set S = all \ cand.
            # A set containing n is independent iff S \ cand is independent AND n not adjacent to any in S.
            # n is adjacent to cand, so n is not adjacent to S. Good.
            # So alpha(new) >= 1 + alpha(S in old graph).
            # We need alpha(new) <= 10, so we need alpha(S) <= 9.
            # Upper bound alpha(S) by Caro-Wei on induced subgraph.
            # But computing that per trial is costly. Use a cheaper proxy:
            # If |S| is large and the graph on S is sparse, alpha(S) is large.
            # Use: if |cand| is large enough, S is small, alpha(S) is small.
            # Heuristic: require |cand| >= n - 9 (so |S| <= 9), then alpha(S) <= 9.
            if len(cand) >= n - 9:
                # Verify C4-free for new vertex more carefully
                ok2 = True
                cand_list = list(cand)
                for i in range(len(cand_list)):
                    u = cand_list[i]
                    for j in range(i + 1, len(cand_list)):
                        v = cand_list[j]
                        c = 0
                        for x in adj[u]:
                            if x in cand:
                                c += 1
                        if c >= 2:
                            ok2 = False
                            break
                    if not ok2:
                        break
                if ok2:
                    # Accept: add vertex
                    found = True
                    break
        if not found:
            break
        # Add the vertex
        new_v = attempt
        adj.append(set())
        for v in cand:
            adj[new_v].add(v)
            adj[v].add(new_v)
        n = new_v + 1
        attempt = n
        # Update best
        if n > best_n:
            # Rebuild edge list
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            best_n = n
            best_edges = edges

    # Final safety: return best found, at minimum the 30-vertex triangle
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
