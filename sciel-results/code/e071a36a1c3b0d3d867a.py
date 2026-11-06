import time
import random
import sys

def construct():
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Target n values to try, starting from the best known 39
    # If we find a valid graph for 39, we return it.
    # We can try higher n if we have time, but 39 is the known bound.
    # Let's focus on 39 first, then try 40, 41...
    
    target_ns = [39, 40, 41, 42, 43]
    
    # If we don't find 39 quickly, we might fall back to a known construction or smaller n
    # But random search for 39 should work.
    
    # Try n=39
    for n in target_ns:
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a valid graph for this n
        result = _search_for_graph(n, deadline)
        if result is not None:
            return result
            
    # Fallback: Return a known valid graph if possible, or a smaller one
    # If we couldn't find 39, try to find 38, 37...
    # Or just return a safe small graph
    # A graph of 30 vertices (10 disjoint triangles) is C4-free and alpha=10.
    # This guarantees a score, though lower.
    return _fallback_graph()

def _search_for_graph(n, deadline):
    """
    Search for a C4-free graph on n vertices with alpha <= 10.
    """
    # Heuristic for p:
    # We want few C4s. E[C4] = C(n,4) * p^4.
    # For n=39, C(39,4) ~ 82k.
    # If p=0.1, p^4=1e-4, E[C4] ~ 8.
    # If p=0.05, p^4=6e-6, E[C4] ~ 0.5.
    # But we need alpha <= 10.
    # Random graph alpha is roughly 2 log_b n where b = 1/(1-p).
    # For p=0.1, b=1.11, log_{1.11}(39) ~ 35. Too high.
    # We need higher p to reduce alpha, but higher p increases C4s.
    # This is the tension.
    # However, C4-free graphs have specific structures.
    # The Erdos-Renyi argument for R(C4, K_k) uses p = n^{-1/2} * sqrt(ln n / k)?
    # Actually, the lower bound construction often uses random graphs with p ~ n^{-1/2}.
    # For n=39, n^{-1/2} ~ 0.16.
    # Let's try p in range [0.1, 0.25].
    
    p_values = [0.15, 0.18, 0.2, 0.22, 0.25, 0.12, 0.1]
    
    for p in p_values:
        if time.time() > deadline - 5.0:
            break
            
        # Try multiple trials for this p
        max_trials = 200
        for _ in range(max_trials):
            if time.time() > deadline - 5.0:
                break
                
            # Generate random graph
            edges = []
            adj = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        edges.append([i, j])
                        adj[i][j] = True
                        adj[j][i] = True
                        
            # Check C4-free
            if not _is_c4_free(adj, n):
                continue
                
            # Check independence number <= 10
            # This is equivalent to Max Clique in complement <= 10
            # Build complement adjacency
            comp_adj = [[not adj[i][j] for j in range(n)] for i in range(n)]
            for i in range(n):
                comp_adj[i][i] = False # No self loops in clique search usually, or handle carefully
                
            # Check if max clique size <= 10
            # We can use a branch and bound to see if clique size > 10 exists
            if _has_clique_size_gt(comp_adj, n, 10):
                continue
            else:
                # Valid graph found
                return {'n': n, 'edges': edges}
                
    return None

def _is_c4_free(adj, n):
    """
    Check if graph is C4-free.
    C4 exists if there are two vertices with >= 2 common neighbors.
    """
    # Count common neighbors for all pairs
    # common[u][v] = number of w such that adj[u][w] and adj[v][w]
    # We can do this efficiently by iterating over each vertex w, 
    # and for each pair of neighbors (u,v) of w, increment count.
    
    # Initialize counts
    # Using a dict or 2D list. 2D list is fine for n=43.
    common_counts = [[0] * n for _ in range(n)]
    
    for w in range(n):
        # Get neighbors of w
        neighbors = [i for i in range(n) if adj[i][w]]
        # For each pair of neighbors, they share w as a common neighbor
        for i in range(len(neighbors)):
            u = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                v = neighbors[j]
                common_counts[u][v] += 1
                common_counts[v][u] += 1
                if common_counts[u][v] > 1:
                    return False
                    
    return True

def _has_clique_size_gt(adj, n, threshold):
    """
    Check if there is a clique of size > threshold in the graph defined by adj.
    Uses a simple branch and bound / backtracking.
    adj[i][j] is True if i and j are connected.
    """
    # We want to find if there is a clique of size threshold + 1.
    # If yes, return True.
    # If no, return False.
    
    # To speed up, we can use a simple recursive search.
    # Order vertices by degree descending to prune faster?
    # For small n, simple recursion is okay.
    
    def expand(clique, candidates):
        if len(clique) > threshold:
            return True
        if not candidates:
            return False
            
        # Pruning: If len(clique) + len(candidates) <= threshold, cannot exceed threshold
        if len(clique) + len(candidates) <= threshold:
            return False
            
        # Pick a vertex from candidates
        # To improve pruning, pick the vertex with highest degree in the candidate subgraph?
        # Or just pick the last one?
        # Let's pick the one with highest degree in the induced subgraph of candidates.
        
        # For simplicity and speed in Python, let's just iterate.
        # We can use a heuristic: remove candidates that don't connect to the new vertex.
        
        # Actually, a standard MCQ (Maximum Clique) algorithm is better.
        # But for just checking > 10, we can do:
        
        # Try adding each candidate to the clique
        # We process candidates in reverse order to allow removal
        for i in range(len(candidates) - 1, -1, -1):
            v = candidates[i]
            
            # New candidates must be neighbors of v and already in candidates (and index < i? No, we need to maintain order)
            # Standard branch and bound:
            # New candidates are candidates[j] for j < i such that adj[v][candidates[j]] is True
            
            new_candidates = []
            for j in range(i):
                if adj[v][candidates[j]]:
                    new_candidates.append(candidates[j])
                    
            if expand(clique + [v], new_candidates):
                return True
                
            # Remove v from candidates for subsequent iterations (handled by loop index)
            
        return False
        
    # Initial candidates: all vertices
    # Sort by degree descending?
    degrees = [sum(adj[i]) for i in range(n)]
    # indices sorted by degree
    sorted_indices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    return expand([], sorted_indices)

def _fallback_graph():
    """
    Returns a known valid C4-free graph with alpha <= 10.
    10 disjoint triangles: n=30, alpha=10.
    """
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# To ensure we don't hit recursion limits or slow down too much, 
# we rely on the fact that n is small (<=43).
# The _has_clique_size_gt function might be slow if the graph is dense in the complement (sparse in G).
# But we filter C4-free first. C4-free graphs are sparse.
# Complement of a sparse graph is dense.
# Max clique in a dense graph is easier to find (or disprove large cliques?)
# Actually, if G is sparse, \bar{G} is dense.
# A dense graph likely has large cliques.
# We are looking for alpha(G) <= 10, i.e., omega(\bar{G}) <= 10.
# If \bar{G} is dense, it's LIKELY to have a clique > 10.
# So most random graphs will FAIL the independence test quickly?
# No, if \bar{G} has a clique of size 11, we return True (fail).
# The branch and bound will find a clique of size 11 quickly if it exists.
# If no clique of size 11 exists, it has to prove it, which is harder.
# But for n=39, proving omega <= 10 is manageable.

# Let's double check the logic.
# We want alpha(G) <= 10.
# alpha(G) = omega(\bar{G}).
# So we check if omega(\bar{G}) <= 10.
# _has_clique_size_gt(\bar{G}, 10) returns True if omega(\bar{G}) > 10.
# If it returns False, then omega(\bar{G}) <= 10, so alpha(G) <= 10. Correct.

# One optimization: If the graph G has a vertex of degree < n - 10, 
# does that help? 
# If v has degree d, it is non-adjacent to n-1-d vertices.
# If n-1-d >= 10, those 10 vertices form an independent set in G?
# No, they are non-adjacent to v, but they might be adjacent to each other.
# So that's not a direct independent set.

# The random search should work.
# To be safe, if the search for 39 takes too long, we might not find it.
# But 39 is the known bound, so it should be findable.

# Let's add a check for time in the main loop.
