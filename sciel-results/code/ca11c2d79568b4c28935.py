import numpy as np
import time

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n = graph['n']
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    def build_matrix(signs):
        A = np.zeros((n, n))
        for i in range(n_edges):
            s = signs[i]
            A[u_arr[i], v_arr[i]] = s
            A[v_arr[i], u_arr[i]] = s
        return A
    
    def spectral_radius(signs):
        A = build_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid(signs):
        try:
            check_lift(graph, signs)
            return True
        except Exception:
            return False
            
    # Identify edge indices
    top_edges = []
    bot_edges = []
    match_edges = []
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(i)
        elif u >= m and v >= m:
            bot_edges.append(i)
        else:
            match_edges.append(i)
            
    top_edges = np.array(top_edges)
    bot_edges = np.array(bot_edges)
    match_edges = np.array(match_edges)

    best_signs = None
    best_radius = float('inf')
    
    # Generate deterministic candidates based on symmetry
    candidates = []
    
    # 1. All +1
    candidates.append(np.ones(n_edges, dtype=int))
    
    # 2. All -1
    candidates.append(-np.ones(n_edges, dtype=int))
    
    # 3. Top +1, Bot -1, Match +1
    c = np.ones(n_edges, dtype=int)
    c[bot_edges] = -1
    candidates.append(c)
    
    # 4. Top -1, Bot +1, Match -1
    c = np.ones(n_edges, dtype=int)
    c[top_edges] = -1
    c[match_edges] = -1
    candidates.append(c)
    
    # 5. Alternating Top/Bot, Match +1
    c = np.ones(n_edges, dtype=int)
    # We need to map edge indices to positions in the cycle to apply alternating signs
    # The edges are given in an ordered list. We assume standard ordering or just use index parity
    # However, "alternating" depends on the vertex indices.
    # Let's construct signs based on vertex indices for top/bot cycles.
    # For edge (u,v) in top cycle, sign = 1 if u % 2 == 0 else -1?
    # Let's try sign = (-1)^u for top, (-1)^u for bot, 1 for match.
    c = np.ones(n_edges, dtype=int)
    for i in top_edges:
        u = edges[i][0]
        c[i] = 1 if u % 2 == 0 else -1
    for i in bot_edges:
        u = edges[i][0] - m
        c[i] = 1 if u % 2 == 0 else -1
    # Match edges are +1
    candidates.append(c)
    
    # 6. Alternating Top/Bot, Match -1
    c = candidates[-1].copy()
    c[match_edges] = -1
    candidates.append(c)
    
    # 7. Top +1, Bot +1, Match -1
    c = np.ones(n_edges, dtype=int)
    c[match_edges] = -1
    candidates.append(c)
    
    # 8. Top -1, Bot -1, Match +1
    c = np.ones(n_edges, dtype=int)
    c[top_edges] = -1
    c[bot_edges] = -1
    candidates.append(c)
    
    # 9. Top +1, Bot -1, Match -1
    c = np.ones(n_edges, dtype=int)
    c[bot_edges] = -1
    c[match_edges] = -1
    candidates.append(c)
    
    # 10. Top -1, Bot +1, Match +1
    c = np.ones(n_edges, dtype=int)
    c[top_edges] = -1
    candidates.append(c)

    for signs in candidates:
        if not is_valid(signs):
            continue
        radius = spectral_radius(signs)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        best_signs = np.ones(n_edges, dtype=int)
        
    return best_signs
