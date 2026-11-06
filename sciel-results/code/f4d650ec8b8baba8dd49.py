"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Degrees
    deg = A.sum(axis=1)
    
    # Local edge density among neighbors: for vertex v, count edges within N(v)
    # Edges in N(v) = sum over u in N(v) of (A[v,:] * A[u,:]) / 2
    # Efficient: (A @ A @ 1)[v] gives sum_{u in N(v)} deg(u) but we want edges among neighbors
    # Use: for each v, edges_among_neighbors[v] = 0.5 * (sum_{u in N(v)} sum_{w in N(v)} A[u,w])
    # = 0.5 * (A @ A @ deg)[v] ... no, that's not right either.
    # Actually: (A^2)[v,u] = number of common neighbors of v and u
    # Edges among N(v) = 0.5 * sum_{u in N(v)} sum_{w in N(v), w != u} A[u,w]
    # = 0.5 * (sum_{u in N(v)} (A @ deg)[u] - 0) ... let me think more carefully.
    #
    # For vertex v: the subgraph induced by N(v) has edge count = 0.5 * sum_{u in N(v)} deg_within_Nv(u)
    # where deg_within_Nv(u) = number of neighbors of u that are also in N(v) = (A^2)[v,u] for u in N(v)
    # So edges_in_N(v) = 0.5 * sum_{u in N(v)} (A^2)[v,u] = 0.5 * (A^2 @ A)[v] ... no
    # (A^2)[v,u] = sum_w A[v,w]*A[w,u] = number of common neighbors of v and u
    # For u in N(v), (A^2)[v,u] = |N(v) ∩ N(u)|
    # Sum over u in N(v) of |N(v) ∩ N(u)| = sum over u in N(v) of (A^2)[v,u] = (A^2 @ A)[v]? No.
    # (A^2 @ A)[v] = sum_u (A^2)[v,u] * A[u,:]... that's a vector.
    # Let me just compute: edges_in_N(v) = 0.5 * sum_{u} A[v,u] * (A^2 @ A_col_sum)[u]... 
    #
    # Simpler: edges_in_N(v) = 0.5 * (A @ A @ deg)[v] - 0.5 * deg[v]
    # Wait: (A @ A)[v,u] = |N(v) ∩ N(u)|. Then (A @ A @ deg)[v] = sum_u |N(v)∩N(u)| * deg(u).
    # That's not quite what we want.
    #
    # Let's use: for each v, the number of edges in the subgraph induced by N(v)
    # = 0.5 * sum_{u in N(v)} sum_{w in N(v)} A[u,w]
    # = 0.5 * sum_{u} A[v,u] * sum_{w} A[v,w] * A[u,w]
    # = 0.5 * sum_{u} A[v,u] * (A @ A)[v,u]  ... no, (A@A)[v,u] = sum_w A[v,w]A[w,u]
    #
    # Actually: sum_{w in N(v)} A[u,w] for fixed u in N(v) = (A @ A)[v, u] only if we consider
    # the row v of A^2. (A^2)[v,u] = sum_w A[v,w]*A[w,u]. Since A[v,u]=1 (u in N(v)),
    # this includes w=u: A[v,u]*A[u,u]=0 (no self-loops), so (A^2)[v,u] = sum_{w in N(v), w≠u} A[u,w].
    # So sum_{w in N(v)} A[u,w] = (A^2)[v,u] for u in N(v).
    # Therefore: edges_in_N(v) = 0.5 * sum_{u in N(v)} (A^2)[v,u] = 0.5 * (A^2 @ A^T)[v, :] dot... 
    # = 0.5 * sum_u A[v,u] * (A^2)[v,u] = 0.5 * (A * A^2).sum(axis=1)
    
    A2 = A @ A
    # edges_in_N(v) = 0.5 * sum_u A[v,u] * A2[v,u]
    edges_in_N = 0.5 * (A * A2).sum(axis=1)
    
    # Local density: edges_in_N(v) / (deg(v)*(deg(v)-1)/2) for deg >= 2
    # This measures how "clique-like" the neighborhood is
    safe_denom = np.maximum(deg * (deg - 1), 1.0)
    local_density = edges_in_N / safe_denom
    
    # Score: prefer low local density (sparse neighborhoods) and moderate degree
    # Low density means picking v doesn't block a dense cluster that could contribute many vertices
    # Also add a mild degree penalty since high-degree vertices block more
    score = -local_density * 3.0 - 0.3 * deg + 0.1 * (deg ** 0.5)
    
    # Check deadline
    if time.time() > deadline:
        return score.astype(float)
    
    return score.astype(float)
# EVOLVE-BLOCK-END
