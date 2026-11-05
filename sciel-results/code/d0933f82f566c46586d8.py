"""Combined program: structural approach from inspiration 1 (complete-graph repair + structured starts)
with refined local search and efficient violation checking from the current program.
Added: bipartite double cover construction with spectral clustering refinement."""
import time
import random
from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    full_mask = (1 << n) - 1

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def is_valid(adj):
        """Check if no 3 vertices have 3 common neighbors (K3,3-free)."""
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def find_violation(adj):
        """Find a K3,3 violation. Returns (i,j,k) or None."""
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return (i, j, k)
        return None

    def make_valid(adj):
        """Remove edges to make the graph valid."""
        adj = adj[:]
        iterations = 0
        while iterations < 5000:
            viol = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            cij = adj[i] & adj[j]
            cijk = cij & adj[k]
            if cijk == 0:
                break
            # Pick a common neighbor and remove its edge to one of i,j,k
            bits = []
            v = cijk
            while v:
                low = v & (-v)
                bits.append(low.bit_length() - 1)
                v ^= low
            x = bits[0]
            # Remove edge (i, x)
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
            iterations += 1
        return adj

    def greedy_add(adj):
        """Try to add edges greedily until no more can be added."""
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    improved = True
                    break
                else:
                    adj[u], adj[v] = old_u, old_v
        return adj

    def local_search(adj, max_iter=5000):
        """Local search: try adding edges, and if blocked, try swaps."""
        adj = adj[:]
        iters = 0
        while iters < max_iter and time.time() < deadline - 15:
            # Try to add an edge
            added = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    added = True
                    break
                else:
                    adj[u], adj[v] = old_u, old_v
            if added:
                iters += 1
                continue

            # Try a swap: remove one edge, add another
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]

            if not edges_out:
                break

            random.shuffle(edges_in)
            random.shuffle(edges_out)

            swapped = False
            for e_rem in edges_in[:50]:
                if swapped:
                    break
                for e_add in edges_out[:50]:
                    (u1, v1) = e_rem
                    (u2, v2) = e_add
                    if (u1, v1) == (u2, v2):
                        continue
                    old = (adj[u1], adj[v1], adj[u2], adj[v2])
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    if is_valid(adj):
                        swapped = True
                        break
                    else:
                        adj[u1], adj[v1], adj[u2], adj[v2] = old
                if not swapped:
                    continue

            if not swapped:
                break
            iters += 1

        return adj

    # Best solution tracking
    best_adj = [0] * n
    best_count = 0
    best_edges = []

    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    # Strategy 1: Start from complete graph, make valid, then improve
    adj = [(full_mask ^ (1 << i)) for i in range(n)]
    adj = make_valid(adj)
    adj = greedy_add(adj)
    adj = local_search(adj)
    update_best(adj)

    # Strategy 2: Try various structured initializations
    # Circulant graphs
    for s in range(5, 12):
        if time.time() > deadline - 40:
            break
        for trial in range(3):
            if time.time() > deadline - 40:
                break
            random.seed(42 + s * 10 + trial)
            offsets = set()
            while len(offsets) < s:
                d = random.randint(1, n - 1)
                offsets.add(d)
                offsets.add(n - d)
            offsets.discard(0)

            adj = [0] * n
            for u in range(n):
                for d in offsets:
                    v = (u + d) % n
                    if v > u:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

            if not is_valid(adj):
                adj = make_valid(adj)
            adj = greedy_add(adj)
            adj = local_search(adj)
            update_best(adj)

    # Strategy 3: Random dense graphs with local search
    for trial in range(10):
        if time.time() > deadline - 25:
            break
        random.seed(1000 + trial * 7)
        adj = [0] * n
        p = 0.5 + trial * 0.05
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj = make_valid(adj)
        adj = greedy_add(adj)
        adj = local_search(adj)
        update_best(adj)

    # Strategy 4: Start from best and do more local search
    if best_edges:
        adj = best_adj[:]
        adj = local_search(adj, max_iter=8000)
        update_best(adj)

        # Try more aggressive swaps
        adj = best_adj[:]
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            adj = local_search(adj, max_iter=3000)
            update_best(adj)

    # Strategy 5: Bipartite double cover of structured graphs with spectral refinement
    def spectral_refine(adj, max_iter=2000):
        """Refine bipartition using spectral clustering to minimize K3,3 violations."""
        adj = adj[:]
        for _ in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Build adjacency matrix
            A = np.zeros((n, n), dtype=np.int64)
            for u in range(n):
                v = adj[u]
                while v:
                    low = v & (-v)
                    vv = low.bit_length() - 1
                    A[u, vv] = 1
                    A[vv, u] = 1
                    v ^= low
            
            # Compute Laplacian
            deg = np.sum(A, axis=1)
            L = np.diag(deg) - A
            
            # Get eigenvectors for smallest non-zero eigenvalues
            try:
                # Use scipy for sparse eigenvalue computation
                L_sparse = csr_matrix(L)
                # Find eigenvalues near zero (Fiedler vector and related)
                vals, vecs = eigsh(L_sparse, k=4, sigma=0.01, which='LM')
            except:
                # Fallback to dense
                vals, vecs = np.linalg.eigh(L)
                vecs = vecs[:, :4]
            
            # Use the eigenvectors to suggest a new bipartition
            # Project onto first few eigenvectors
            X = vecs.T
            
            # K-means style clustering into 2 groups
            # Simple approach: sign of first non-trivial eigenvector
            fiedler = vecs[:, 0] if vecs.shape[1] > 0 else np.zeros(n)
            partition = (fiedler >= 0).astype(int)
            
            # Try to improve: move vertices to reduce violations
            improved = False
            for v in range(n):
                if time.time() > deadline - 20:
                    break
                current_part = partition[v]
                other_part = 1 - current_part
                
                # Count violations if v moves to other_part
                viol_before = 0
                viol_after = 0
                
                # For each pair of vertices in the same part as v, check common neighbors
                same_part = [u for u in range(n) if u != v and partition[u] == current_part]
                for i, j in combinations(same_part, 2):
                    cij = adj[i] & adj[j]
                    if cij.bit_count() >= 3:
                        viol_before += 1
                
                # If v moves to other_part, check new violations
                other_same = [u for u in range(n) if u != v and partition[u] == other_part]
                for i, j in combinations(other_same, 2):
                    cij = adj[i] & adj[j]
                    if cij.bit_count() >= 3:
                        viol_after += 1
                
                if viol_after < viol_before:
                    partition[v] = other_part
                    improved = True
            
            if not improved:
                break
        
        return adj

    def bipartite_double_cover(base_adj, base_n):
        """Create bipartite double cover of a graph."""
        # Vertices 0..base_n-1 are copy 1, base_n..2*base_n-1 are copy 2
        n_cover = 2 * base_n
        adj = [0] * n_cover
        
        for u in range(base_n):
            v = base_adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    # Edge (u, vv) in base -> edges (u, base_n+vv) and (base_n+u, vv) in cover
                    adj[u] |= (1 << (base_n + vv))
                    adj[base_n + vv] |= (1 << u)
                    adj[base_n + u] |= (1 << vv)
                    adj[vv] |= (1 << (base_n + u))
                v ^= low
        
        return adj

    # Try Petersen graph (10 vertices, 15 edges)
    if time.time() < deadline - 60:
        # Petersen graph: vertices 0-9, edges form outer pentagon and inner star
        petersen_edges = [(0,1), (1,2), (2,3), (3,4), (4,0),
                          (5,6), (6,7), (7,8), (8,9), (9,5),
                          (0,5), (1,6), (2,7), (3,8), (4,9)]
        
        base_n = 10
        petersen_adj = [0] * base_n
        for u, v in petersen_edges:
            petersen_adj[u] |= (1 << v)
            petersen_adj[v] |= (1 << u)
        
        cover_adj = bipartite_double_cover(petersen_adj, base_n)
        
        # The cover has 20 vertices, should be K3,3-free by construction
        # Verify and refine
        if is_valid(cover_adj):
            cover_adj = greedy_add(cover_adj)
            cover_adj = local_search(cover_adj, max_iter=3000)
            update_best(cover_adj)
        
        # Try spectral refinement
        cover_adj2 = bipartite_double_cover(petersen_adj, base_n)
        cover_adj2 = spectral_refine(cover_adj2)
        if is_valid(cover_adj2):
            cover_adj2 = greedy_add(cover_adj2)
            cover_adj2 = local_search(cover_adj2, max_iter=3000)
            update_best(cover_adj2)

    # Try complete graph K5 (10 edges) as base
    if time.time() < deadline - 40:
        base_n = 10
        k5_adj = [0] * base_n
        for i in range(base_n):
            for j in range(i+1, base_n):
                k5_adj[i] |= (1 << j)
                k5_adj[j] |= (1 << i)
        
        cover_adj = bipartite_double_cover(k5_adj, base_n)
        # K5 double cover will have many violations, need repair
        cover_adj = make_valid(cover_adj)
        cover_adj = greedy_add(cover_adj)
        cover_adj = local_search(cover_adj, max_iter=3000)
        update_best(cover_adj)

    # Try cycle C10
    if time.time() < deadline - 20:
        base_n = 10
        c10_adj = [0] * base_n
        for i in range(base_n):
            j = (i + 1) % base_n
            c10_adj[i] |= (1 << j)
            c10_adj[j] |= (1 << i)
        
        cover_adj = bipartite_double_cover(c10_adj, base_n)
        if is_valid(cover_adj):
            cover_adj = greedy_add(cover_adj)
            cover_adj = local_search(cover_adj, max_iter=3000)
            update_best(cover_adj)

    # Fallback
    if not best_edges:
        return [[i, i+1] for i in range(19)]

    return best_edges
