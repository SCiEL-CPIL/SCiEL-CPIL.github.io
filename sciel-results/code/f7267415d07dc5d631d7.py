import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s margin for safety
    
    # Use Paley graph on 41 vertices as seed
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    # Initialize adjacency matrix or set of edges
    # Using set of tuples (i, j) with i < j
    edges = set()
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.add((i, j))
    
    n = q
    vertices = list(range(n))
    
    def find_violation():
        """Find a K5 or independent set of size 5. Returns (is_clique, vertices) or None."""
        # Check for K5
        for combo in combinations(vertices, 5):
            is_clique = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = combo[i], combo[j]
                    if u > v:
                        u, v = v, u
                    if (u, v) not in edges:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return (True, combo)
        
        # Check for independent set of size 5
        for combo in combinations(vertices, 5):
            is_independent = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = combo[i], combo[j]
                    if u > v:
                        u, v = v, u
                    if (u, v) in edges:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return (False, combo)
                
        return None
    
    # Repair phase
    max_iterations = 5000
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        violation = find_violation()
        if violation is None:
            return n, list(edges)
            
        is_clique, vset = violation
        
        if is_clique:
            # Remove a random edge from the clique
            pairs = list(combinations(vset, 2))
            random.shuffle(pairs)
            for u, v in pairs:
                if u > v:
                    u, v = v, u
                if (u, v) in edges:
                    edges.remove((u, v))
                    break
        else:
            # Add a random non-edge from the independent set
            pairs = list(combinations(vset, 2))
            random.shuffle(pairs)
            for u, v in pairs:
                if u > v:
                    u, v = v, u
                if (u, v) not in edges:
                    edges.add((u, v))
                    break
                    
    # Fallback: Vertex deletion
    # If repair didn't work or timed out, delete vertices until valid
    # We need to re-evaluate because the graph might still be invalid
    # Also, we might have modified the graph significantly, so we just delete vertices from the current state
    
    # To be safe, let's restart the deletion from the current edges state
    # But we need to update the vertex list if we delete vertices
    # Actually, the vertex set is fixed at 0..40. If we delete a vertex, we just ignore it?
    # No, the output must be a graph on n vertices with indices 0..n-1.
    # If we delete vertex 10, the indices shift. This is messy.
    
    # Better fallback: Start fresh from P_41 and delete vertices
    # This ensures indices are 0..n-1
    
    # If we didn't return in the repair loop, we do vertex deletion
    # We need to reconstruct the graph from scratch for the deletion phase to keep indices clean
    # Or, we can just return the best valid graph found during repair if we track it?
    # No, repair modifies the graph. If it doesn't become valid, we discard it.
    
    # Let's do a simple vertex deletion from P_41
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    edges = set()
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.add((i, j))
    
    vertices = list(range(q))
    n = q
    
    # We need a faster violation check for the deletion loop
    # But deletion reduces n, so the check becomes faster.
    
    while time.time() < deadline:
        violation = find_violation()
        if violation is None:
            return n, list(edges)
            
        is_clique, vset = violation
        # Remove a vertex from the violation
        # Random choice
        v_to_remove = random.choice(vset)
        
        # Remove vertex v_to_remove
        vertices.remove(v_to_remove)
        n -= 1
        
        # Remove edges connected to v_to_remove
        edges_to_remove = []
        for e in edges:
            if v_to_remove in e:
                edges_to_remove.append(e)
        
        for e in edges_to_remove:
            edges.remove(e)
            
        # Renumber vertices to 0..n-1
        # This is O(n^2) but n is small
        mapping = {v: i for i, v in enumerate(vertices)}
        new_edges = set()
        for u, v in edges:
            nu = mapping[u]
            nv = mapping[v]
            if nu > nv:
                nu, nv = nv, nu
            new_edges.add((nu, nv))
        edges = new_edges

    # If time runs out, return the current best (which might be invalid if we didn't check at the very end)
    # But the loop checks at the start of each iteration.
    # If we exit due to time, we should return the last known valid state?
    # We don't track the last valid state explicitly in the deletion loop.
    # However, the probability of timing out with an invalid graph is low if we check often.
    # To be safe, if we exit the loop, we can do one final check.
    # If invalid, we can't return it.
    # But we must return something.
    # The seed P_37 is valid. We can hardcode P_37 as the absolute fallback.
    
    # Let's add a hard fallback to P_37 if everything fails
    if time.time() > deadline - 5:
         q = 37
         residues = {(x * x) % q for x in range(1, q)}
         return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

    # Final check
    violation = find_violation()
    if violation is None:
        return n, list(edges)
    
    # If still invalid, return P_37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
