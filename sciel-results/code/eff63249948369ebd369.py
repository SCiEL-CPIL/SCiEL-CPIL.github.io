"""Improved ternary covering code K_3(6,1) via graph-theoretic MIS approach."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import networkx as nx


def _all_words(q, n):
    """Generate all q^n words as an (N, n) int array."""
    if n == 0:
        return np.zeros((1, 0), dtype=np.int8)
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    return words


def _build_distance_graph(q, n, radius):
    """
    Build the distance-1 graph: vertices are all q^n words,
    edges connect words at Hamming distance <= radius.
    """
    all_words = _all_words(q, n)
    N = len(all_words)
    
    # For efficiency, build adjacency via vectorized Hamming distance
    # N=729, so N^2 = 531441 comparisons is fine
    adj = np.zeros((N, N), dtype=bool)
    
    for i in range(n):
        adj |= (all_words[:, i:i+1] != all_words[i, None])
    
    # For radius=1: edge if Hamming distance is 1 (or 0 for self-loops, which we ignore)
    # Compute full Hamming distance
    if radius == 1:
        # Edge exists if exactly 1 position differs
        diff_count = np.zeros((N, N), dtype=np.int16)
        for i in range(n):
            diff_count += (all_words[:, i:i+1] != all_words[i, None]).astype(np.int16)
        adj = (diff_count == 1)
    else:
        diff_count = np.zeros((N, N), dtype=np.int16)
        for i in range(n):
            diff_count += (all_words[:, i:i+1] != all_words[i, None]).astype(np.int16)
        adj = (diff_count <= radius)
    
    # Remove self-loops
    np.fill_diagonal(adj, False)
    
    G = nx.from_numpy_array(adj.astype(np.int8))
    return G, all_words


def _mis_to_codewords(G, all_words):
    """
    Find a maximum independent set in G.
    An independent set in the distance-1 graph is a set of codewords
    such that no two are within radius of each other.
    Wait - that's wrong. We need a DOMINATING SET, not independent set.
    
    Actually, let me reconsider:
    - A dominating set D: every vertex is in D or adjacent to some vertex in D
    - In the distance-radius graph, D is a dominating set iff every word is within radius of some codeword
    
    The complement: vertices NOT in D must be adjacent to some vertex in D.
    
    For minimum dominating set, we can use NetworkX's min_weight_dominating_set
    or convert to a hitting set / set cover problem.
    
    Let me use a different approach: greedy + local search on the graph.
    """
    pass


def _greedy_dominating_set(G, all_words, deadline):
    """
    Greedy: repeatedly pick the vertex that dominates the most undominated vertices.
    """
    N = G.number_of_nodes()
    dominated = np.zeros(N, dtype=bool)
    codeword_indices = []
    
    start_time = time.time()
    
    while not dominated.all() and time.time() < deadline:
        # For each vertex, count how many undominated vertices it dominates
        # (including itself)
        best_idx = -1
        best_count = -1
        
        for v in range(N):
            if dominated[v]:
                continue
            # Count undominated vertices in closed neighborhood of v
            # Closed neighborhood: v itself + all neighbors
            count = 0
            if not dominated[v]:
                count += 1
            for u in G.neighbors(v):
                if not dominated[u]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = v
        
        if best_idx == -1:
            break
        
        # Add best_idx to dominating set
        codeword_indices.append(best_idx)
        
        # Mark all vertices in closed neighborhood as dominated
        dominated[best_idx] = True
        for u in G.neighbors(best_idx):
            dominated[u] = True
        
        if time.time() >= deadline:
            break
    
    return [all_words[i].tolist() for i in codeword_indices]


def _local_search_reduce(codewords, G, all_words, deadline):
    """
    Try to remove codewords if coverage is maintained.
    Also try swapping one codeword for another to enable removals.
    """
    N = len(all_words)
    codeword_indices = [np.where((all_words == cw).all(axis=1))[0][0] for cw in codewords]
    codeword_indices = list(codeword_indices)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        dominated = np.zeros(N, dtype=bool)
        for idx in codeword_indices:
            dominated[idx] = True
            for u in G.neighbors(idx):
                dominated[u] = True
        
        for i in range(len(codeword_indices)):
            if time.time() >= deadline:
                break
            # Try removing codeword i
            remaining = codeword_indices[:i] + codeword_indices[i+1:]
            if not remaining:
                continue
            
            # Check if remaining still dominates all
            temp_dominated = np.zeros(N, dtype=bool)
            for idx in remaining:
                temp_dominated[idx] = True
                for u in G.neighbors(idx):
                    temp_dominated[u] = True
            
            if temp_dominated.all():
                codeword_indices = remaining
                improved = True
                break
    
    return [all_words[i].tolist() for i in codeword_indices]


def _swap_local_search(codewords, G, all_words, deadline):
    """
    Try swapping one codeword for another, then try removing others.
    """
    N = len(all_words)
    codeword_indices = [np.where((all_words == cw).all(axis=1))[0][0] for cw in codewords]
    codeword_indices = list(codeword_indices)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        dominated = np.zeros(N, dtype=bool)
        for idx in codeword_indices:
            dominated[idx] = True
            for u in G.neighbors(idx):
                dominated[u] = True
        
        # Try each swap: remove i, add j (j not in current set)
        current_set = set(codeword_indices)
        for i in range(len(codeword_indices)):
            if time.time() >= deadline:
                break
            for j in range(N):
                if j in current_set:
                    continue
                
                new_set = codeword_indices[:i] + codeword_indices[i+1:] + [j]
                
                # Check if new set dominates all
                temp_dominated = np.zeros(N, dtype=bool)
                for idx in new_set:
                    temp_dominated[idx] = True
                    for u in G.neighbors(idx):
                        temp_dominated[u] = True
                
                if temp_dominated.all():
                    # Now try to remove more
                    codeword_indices = new_set
                    improved = True
                    break
            if improved:
                break
    
    return [all_words[i].tolist() for i in codeword_indices]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    all_words = _all_words(q, n)
    N = len(all_words)
    
    # Build the distance-radius graph
    G, _ = _build_distance_graph(q, n, radius)
    
    # Phase 1: Greedy dominating set
    codewords = _greedy_dominating_set(G, all_words, deadline)
    
    # Phase 2: Local search - remove redundant codewords
    if time.time() < deadline:
        codewords = _local_search_reduce(codewords, G, all_words, deadline)
    
    # Phase 3: Swap-based local search
    if time.time() < deadline:
        codewords = _swap_local_search(codewords, G, all_words, deadline)
    
    # Phase 4: Another round of removals
    if time.time() < deadline:
        codewords = _local_search_reduce(codewords, G, all_words, deadline)
    
    # Verify coverage
    cw = np.array(codewords, dtype=np.int8)
    if len(cw) == 0:
        return codewords
    
    # Check all words are within radius of some codeword
    covered = np.zeros(N, dtype=bool)
    for cw_row in cw:
        for i in range(N):
            if np.sum(all_words[i] != cw_row) <= radius:
                covered[i] = True
                break
        if covered.all():
            break
    
    if not covered.all():
        # Fallback: add uncovered words
        for i in range(N):
            if not covered[i]:
                codewords.append(all_words[i].tolist())
                covered[i] = True
                for j in range(N):
                    if np.sum(all_words[j] != all_words[i]) <= radius:
                        covered[j] = True
    
    return codewords
# EVOLVE-BLOCK-END
