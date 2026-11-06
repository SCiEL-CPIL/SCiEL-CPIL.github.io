"""Improved search for small triangle degree sums on 16 vertices, 65 edges."""
import time
import random
import itertools

N = 16
TARGET_EDGES = 65
DEADLINE_MARGIN = 10  # seconds before hard kill


def _edges_to_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _degrees(adj):
    return [len(s) for s in adj]


def _triangles(adj):
    tris = []
    for u in range(N):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        tris.append((u, v, w))
    return tris


def _score_graph(adj, tris, deg):
    if not tris:
        return 999
    return max(deg[u] + deg[v] + deg[w] for u, v, w in tris)


def _score_edges(edges):
    adj = _edges_to_adj(edges)
    deg = _degrees(adj)
    tris = _triangles(adj)
    return _score_graph(adj, tris, deg)


def _initial_edges():
    # K_{8,8} plus one intra-part edge
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return edges


def _all_pairs():
    return [(u, v) for u in range(N) for v in range(u + 1, N)]


def _edge_index(edges):
    """Map (u,v) with u<v to index in edges list."""
    m = {}
    for i, (u, v) in enumerate(edges):
        a, b = (u, v) if u < v else (v, u)
        m[(a, b)] = i
    return m


def _try_swap(edges, ei, add_pair, remove_idx, adj, deg, tris, current_score):
    """Attempt to add add_pair and remove edges[remove_idx]. Return new state or None."""
    u, v = add_pair
    if u == v:
        return None
    a, b = (u, v) if u < v else (v, u)
    if (a, b) in ei:
        return None
    # Check that removing remove_idx doesn't break triangle count constraints? 
    # We just need exactly 65 edges. Adding 1 and removing 1 keeps count.
    # But we need to ensure the graph remains simple.
    
    # Check if add_pair creates a triangle with higher score? We'll compute fully.
    # To speed up, we can check if the new edge (u,v) is in a triangle.
    # Triangles involving new edge: for w in adj[u] & adj[v]
    common = adj[u] & adj[v]
    if common:
        # New triangles formed
        max_new = 0
        for w in common:
            # deg[u] and deg[v] will increase by 1
            # deg[w] unchanged
            s = (deg[u] + 1) + (deg[v] + 1) + deg[w]
            if s > max_new:
                max_new = s
        if max_new > current_score:
            return None  # Skip if it immediately worsens max
    
    # Perform swap
    rem_u, rem_v = edges[remove_idx]
    ra, rb = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
    
    # Update adj
    adj[u].add(v)
    adj[v].add(u)
    adj[rem_u].remove(rem_v)
    adj[rem_v].remove(rem_u)
    
    # Update degrees
    deg[u] += 1
    deg[v] += 1
    deg[rem_u] -= 1
    deg[rem_v] -= 1
    
    # Update edge index
    ei[(a, b)] = len(edges)  # temporary, we'll rebuild edges list
    # Actually, let's just modify the list in place for speed
    old_edge = edges[remove_idx]
    edges[remove_idx] = [u, v]
    # Remove from ei
    del ei[(ra, rb)]
    ei[(a, b)] = remove_idx
    
    # Recompute triangles and score
    # This is expensive. Let's only do it if the quick check passed.
    tris_new = _triangles(adj)
    score_new = _score_graph(adj, tris_new, deg)
    
    # Revert
    adj[u].remove(v)
    adj[v].remove(u)
    adj[rem_u].add(rem_v)
    adj[rem_v].add(rem_u)
    deg[u] -= 1
    deg[v] -= 1
    deg[rem_u] += 1
    deg[rem_v] += 1
    edges[remove_idx] = old_edge
    del ei[(a, b)]
    ei[(ra, rb)] = remove_idx
    
    if score_new < current_score:
        # Accept
        adj[u].add(v)
        adj[v].add(u)
        adj[rem_u].remove(rem_v)
        adj[rem_v].remove(rem_u)
        deg[u] += 1
        deg[v] += 1
        deg[rem_u] -= 1
        deg[rem_v] -= 1
        edges[remove_idx] = [u, v]
        del ei[(ra, rb)]
        ei[(a, b)] = remove_idx
        return (edges, ei, adj, deg, tris_new, score_new)
    else:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Start with the seed
    edges = _initial_edges()
    ei = _edge_index(edges)
    adj = _edges_to_adj(edges)
    deg = _degrees(adj)
    tris = _triangles(adj)
    current_score = _score_graph(adj, tris, deg)
    
    best_edges = edges[:]
    best_score = current_score
    
    all_pairs = _all_pairs()
    
    # Hill climbing with random restarts
    while time.time() < deadline:
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle pairs for randomness
            random.shuffle(all_pairs)
            for pair in all_pairs:
                if time.time() > deadline:
                    break
                u, v = pair
                if (min(u,v), max(u,v)) in ei:
                    continue
                # Try removing each existing edge
                # To speed up, only consider removing edges not incident to u or v? 
                # No, removing any edge is valid. But we can prioritize.
                # Let's just iterate through all edges
                for idx in range(len(edges)):
                    if time.time() > deadline:
                        break
                    result = _try_swap(edges, ei, pair, idx, adj, deg, tris, current_score)
                    if result is not None:
                        edges, ei, adj, deg, tris, current_score = result
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = edges[:]
                        improved = True
                        break  # Restart pair loop after improvement
        
        # Random restart: perturb the current best
        if time.time() < deadline:
            edges = best_edges[:]
            ei = _edge_index(edges)
            adj = _edges_to_adj(edges)
            deg = _degrees(adj)
            tris = _triangles(adj)
            current_score = _score_graph(adj, tris, deg)
            
            # Perform a few random swaps to perturb
            for _ in range(3):
                if time.time() > deadline:
                    break
                # Pick a random edge to remove
                idx = random.randint(0, len(edges) - 1)
                rem_u, rem_v = edges[idx]
                # Pick a random pair to add
                pair = random.choice(all_pairs)
                u, v = pair
                if (min(u,v), max(u,v)) in ei:
                    continue
                # Do the swap unconditionally
                a, b = (u, v) if u < v else (v, u)
                ra, rb = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
                
                adj[u].add(v)
                adj[v].add(u)
                adj[rem_u].remove(rem_v)
                adj[rem_v].remove(rem_u)
                
                deg[u] += 1
                deg[v] += 1
                deg[rem_u] -= 1
                deg[rem_v] -= 1
                
                edges[idx] = [u, v]
                del ei[(ra, rb)]
                ei[(a, b)] = idx
                
                tris = _triangles(adj)
                current_score = _score_graph(adj, tris, deg)
    
    return {'n': N, 'edges': best_edges}


if __name__ == "__main__":
    res = construct()
    adj = _edges_to_adj(res['edges'])
    deg = _degrees(adj)
    tris = _triangles(adj)
    print("Score:", _score_graph(adj, tris, deg))
    print("Edges:", len(res['edges']))
