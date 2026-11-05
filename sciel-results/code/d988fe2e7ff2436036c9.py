"""Book-size minimizer for 40 vertices, 401 edges."""
import random
import time

N = 40
M = 401

def _bit_count(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

def _init_graph():
    """K_{20,20} plus 1 edge inside part A. Very low book sizes."""
    edges = []
    for a in range(20):
        for b in range(20, 40):
            edges.append((a, b))
    edges.append((0, 1))
    return edges

def _build_adj(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def _compute_books(adj, edges):
    """Return list of book sizes (common-neighbor counts) for each edge."""
    books = []
    for u, v in edges:
        books.append(_bit_count(adj[u] & adj[v]))
    return books

def _score(books):
    return max(books)

def _score_sq(books):
    """Sum of squares of books; smoother objective for local search."""
    s = 0
    for b in books:
        s += b * b
    return s

def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    random.seed(12345)

    # Start from K_{20,20} + 1 internal edge
    edges = _init_graph()
    edge_set = set(edges)
    adj = _build_adj(edges)
    books = _compute_books(adj, edges)
    best_books = list(books)
    best_score = _score(best_books)
    best_sq = _score_sq(best_books)
    best_edges = list(edges)

    # Local search: try swapping one edge for another
    # Objective: minimize max book, then minimize sum of squares
    max_iters = 500000
    it = 0
    stuck = 0

    while it < max_iters:
        if time.time() > DEADLINE:
            break
        it += 1

        # Pick a random edge to remove
        i = random.randrange(M)
        u, v = edges[i]

        # Remove edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_set.discard((u, v))
        edge_set.discard((v, u))
        books[i] = 0  # temporarily

        # Find a candidate edge to add
        # Try edges that connect vertices with low common-neighbor counts
        # Sample a few random non-edges
        best_add = None
        best_add_sq = None
        best_add_max = None

        # Try up to 20 random non-edge candidates
        for _ in range(20):
            a = random.randrange(N)
            b = random.randrange(N)
            if a >= b:
                continue
            if (a, b) in edge_set:
                continue
            if a == u and b == v:
                continue
            if a == v and b == u:
                continue

            # Compute book size of (a,b) in current adj
            common = _bit_count(adj[a] & adj[b])
            # After adding (a,b), books of edges incident to a or b that go through common vertices change
            # For simplicity, estimate: new max book approx max(current max excluding removed, common)
            # and new sum-of-squares change:
            # Edges (a,c) for c in adj[a] & adj[b]: their book increases by 1
            # Edges (b,c) for c in adj[a] & adj[b]: their book increases by 1
            # Edge (a,b): book = common
            # This is an approximation; we'll verify if accepted

            # Quick estimate of new sum of squares
            # Current sum of squares (with edge i removed, book=0)
            # We don't recompute fully; use delta
            # delta_sq = common^2 - 0 (for new edge)
            #           + sum over affected edges of (book+1)^2 - book^2
            #           = common^2 + sum_{affected} (2*book + 1)
            # Affected edges: (a,c) and (b,c) for c in common_set
            common_set = adj[a] & adj[b]
            delta_sq = common * common
            # For each c in common_set, edges (a,c) and (b,c) exist
            # (a,c) book was books of edge (a,c); we don't have indexed books easily
            # Use approximation: assume average book ~ best_sq / M
            # This is too rough; instead just use common as proxy
            # Simpler: minimize common primarily
            if best_add is None or common < best_add_max or (common == best_add_max and delta_sq < best_add_sq):
                best_add = (a, b)
                best_add_sq = delta_sq
                best_add_max = common

        if best_add is None:
            # Revert removal
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            edge_set.add((u, v))
            books[i] = _bit_count(adj[u] & adj[v])
            continue

        a, b = best_add

        # Add the new edge
        adj[a] |= 1 << b
        adj[b] |= 1 << a
        edge_set.add((a, b))
        edges[i] = (a, b)

        # Recompute books for affected edges: all edges incident to a or b, plus the new edge
        # For efficiency, recompute books for edges incident to a, b, u, v
        # Actually, changing adj[a] and adj[b] affects books of all edges (a,c) and (b,c)
        # and also the new edge (a,b)
        # And removing (u,v) affected books of edges (u,c) and (v,c) -- but we already removed it
        # Let's just recompute all books; M=401, N=40, bit_count is fast
        # 401 * bit_count(40-bit) is very fast in Python? Let's check: _bit_count loops over set bits
        # Average degree ~ 20, so average common neighbors ~ 10, so ~10 iterations per edge
        # 401 * 10 = 4010 operations per recompute. 500k iters * 4010 = 2e9 -- too slow in Python
        # Need to be smarter.

        # Better: only recompute books for edges that could have changed.
        # When we remove (u,v) and add (a,b):
        # - adj[u], adj[v] lost each other
        # - adj[a], adj[b] gained each other
        # Books that change:
        #   (u, c) for c != v: book may decrease by 1 if v was in common(u,c)
        #   (v, c) for c != u: book may decrease by 1 if u was in common(v,c)
        #   (a, c) for c != b: book may increase by 1 if b was in common(a,c)
        #   (b, c) for c != a: book may increase by 1 if a was in common(b,c)
        #   (a, b): new, book = common(a,b)
        # This is complex. Given time constraints, let's just recompute all books
        # but limit iterations.

        books = _compute_books(adj, edges)
        cur_score = _score(books)
        cur_sq = _score_sq(books)

        # Accept if better
        if (cur_score < best_score) or (cur_score == best_score and cur_sq < best_sq):
            best_score = cur_score
            best_sq = cur_sq
            best_books = list(books)
            best_edges = list(edges)
            stuck = 0
        else:
            # Revert
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            edge_set.discard((a, b))
            edges[i] = (u, v)
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            edge_set.add((u, v))
            books = _compute_books(adj, edges)
            stuck += 1

        # If stuck too long, do a small random perturbation
        if stuck > 500:
            stuck = 0
            for _ in range(5):
                i2 = random.randrange(M)
                u2, v2 = edges[i2]
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                edge_set.discard((u2, v2))
                edge_set.discard((v2, u2))
                # Add random non-edge
                for _ in range(50):
                    a2 = random.randrange(N)
                    b2 = random.randrange(N)
                    if a2 >= b2:
                        continue
                    if (a2, b2) in edge_set:
                        continue
                    adj[a2] |= 1 << b2
                    adj[b2] |= 1 << a2
                    edge_set.add((a2, b2))
                    edges[i2] = (a2, b2)
                    break
            books = _compute_books(adj, edges)
            cur_score = _score(books)
            cur_sq = _score_sq(books)
            if (cur_score < best_score) or (cur_score == best_score and cur_sq < best_sq):
                best_score = cur_score
                best_sq = cur_sq
                best_books = list(books)
                best_edges = list(edges)

    # Return best edges found
    return [list(e) for e in best_edges]
