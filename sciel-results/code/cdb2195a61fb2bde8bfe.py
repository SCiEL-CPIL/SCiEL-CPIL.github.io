import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)

    # Build edge lookup and adjacency
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i

    edge_u = np.array([u for u, v in edges], dtype=np.int32)
    edge_v = np.array([v for v, u in edges], dtype=np.int32)

    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    # Identify spanning tree and chords using BFS
    visited = [False] * n
    tree_edges = set()
    visited[0] = True
    queue = [0]

    while queue:
        u = queue.pop(0)
        for v, ei in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(ei)
                queue.append(v)

    chord_indices = [i for i in range(num_edges) if i not in tree_edges]
    num_chords = len(chord_indices)

    # Helper: check if lift is connected (has a negative cycle)
    def is_lift_connected(signs):
        # A signed graph's lift is connected iff the signed graph is balanced
        # OR has a negative cycle. Actually, for a 2-lift to be connected,
        # the signed graph must NOT be balanced (i.e., must have at least one negative cycle).
        # Wait, let me re-read the problem statement.
        # "The lift must be connected"
        # In a signed graph, the 2-lift is connected if and only if the signed graph is NOT balanced.
        # A signed graph is balanced if every cycle has positive product of signs.
        # So we need at least one negative cycle.
        # Actually, the condition for the 2-lift to be connected is that the signed graph is not balanced.
        # Let's check: if all cycles are positive (balanced), the lift splits into two components.
        # If there's a negative cycle, the lift is connected.
        
        # We can check balance by doing a BFS/DFS and assigning parities.
        # If we can assign parities consistently, it's balanced (disconnected lift).
        # If we find an inconsistency, it's not balanced (connected lift).
        
        parity = [-1] * n
        parity[0] = 0
        q = [0]
        
        while q:
            u = q.pop(0)
            pu = parity[u]
            for v, ei in adj[u]:
                s = signs[ei]
                # If sign is +1, parity doesn't change. If -1, parity flips.
                # Let's use 0 for +, 1 for -.
                flip = 0 if s == 1 else 1
                expected_parity = pu ^ flip
                
                if parity[v] == -1:
                    parity[v] = expected_parity
                    q.append(v)
                else:
                    if parity[v] != expected_parity:
                        # Inconsistency found: negative cycle exists
                        # Lift is connected
                        return True
        
        # No inconsistency: balanced graph, lift is disconnected
        return False

    # Helper: compute spectral radius
    def compute_radius(signs):
        A_signed = np.zeros((n, n), dtype=np.float64)
        # signs are 1 or -1
        A_signed[edge_u, edge_v] = signs.astype(np.float64)
        A_signed[edge_v, edge_u] = signs.astype(np.float64)
        eigvals = np.linalg.eigvalsh(A_signed)
        return np.max(np.abs(eigvals))

    best_radius = float('inf')
    best_signs = None

    # Strategy:
    # 1. If num_chords is small, enumerate all 2^num_chords assignments.
    # 2. If num_chords is larger, use random sampling + local search.
    #    But for m=4,5,7,9:
    #    m=4: n=8, edges=12. Tree edges=7, chords=5. 2^5=32. Enumerate.
    #    m=5: n=10, edges=15. Tree edges=9, chords=6. 2^6=64. Enumerate.
    #    m=7: n=14, edges=21. Tree edges=13, chords=8. 2^8=256. Enumerate.
    #    m=9: n=18, edges=27. Tree edges=17, chords=10. 2^10=1024. Enumerate.
    #    
    #    Wait, let's verify the number of edges in a prism graph.
    #    Prism graph P_m has 2m vertices and 3m edges.
    #    m=4: 12 edges. 2m-1 = 7 tree edges. 12-7=5 chords.
    #    m=5: 15 edges. 2m-1 = 9 tree edges. 15-9=6 chords.
    #    m=7: 21 edges. 2m-1 = 13 tree edges. 21-13=8 chords.
    #    m=9: 27 edges. 2m-1 = 17 tree edges. 27-17=10 chords.
    #    
    #    So for all four cases, num_chords <= 10.
    #    We can enumerate ALL possible assignments!
    #    This guarantees the global optimum.

    if num_chords <= 20:
        # Enumerate all chord assignments
        # Tree edges are fixed to +1 (gauge invariance)
        for mask in range(1 << num_chords):
            if time.time() > deadline:
                break
            
            signs = np.ones(num_edges, dtype=np.int8)
            for j, idx in enumerate(chord_indices):
                if mask & (1 << j):
                    signs[idx] = -1
            
            if not is_lift_connected(signs):
                continue
                
            radius = compute_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    else:
        # Fallback for larger graphs (not expected for this task)
        rng = np.random.default_rng(42)
        for _ in range(100000):
            if time.time() > deadline:
                break
                
            signs = np.ones(num_edges, dtype=np.int8)
            for idx in chord_indices:
                signs[idx] = rng.choice([-1, 1])
                
            if not is_lift_connected(signs):
                continue
                
            radius = compute_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        
        # Local search
        if best_signs is not None:
            current_signs = best_signs.copy()
            current_radius = best_radius
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for ei in range(num_edges):
                    trial = current_signs.copy()
                    trial[ei] = -trial[ei]
                    
                    if not is_lift_connected(trial):
                        continue
                        
                    radius = compute_radius(trial)
                    if radius < current_radius - 1e-10:
                        current_signs = trial
                        current_radius = radius
                        best_radius = radius
                        best_signs = trial.copy()
                        improved = True

    # Fallback if no valid sign found (should not happen)
    if best_signs is None:
        signs = np.ones(num_edges, dtype=np.int8)
        if num_chords > 0:
            signs[chord_indices[0]] = -1
        else:
            signs[0] = -1
        return signs
        
    return best_signs
