"""Optimized search for the largest pairwise non-coprime subset of [1, 255] containing 255."""

import math
import itertools

# Deadline guard
import time

_N = 255
_TARGET = 255  # 255 must be in the set

def _is_valid_clique(S):
    """Check that S is pairwise non-coprime and contains 255."""
    if _TARGET not in S:
        return False
    S = list(S)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if math.gcd(S[i], S[j]) == 1:
                return False
    return True

def _build_candidate_graph():
    """
    Build a graph where edges represent gcd(a,b) > 1 for a, b in [1, 255].
    Returns adjacency as a set of frozensets for speed, and a list of candidates
    that can coexist with 255 (i.e., gcd(x, 255) > 1).
    """
    # Candidates must share a factor with 255 = 3*5*17
    candidates = [x for x in range(1, _N + 1) if math.gcd(x, _TARGET) > 1]
    
    adj = {x: set() for x in candidates}
    for i in range(len(candidates)):
        a = candidates[i]
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            if math.gcd(a, b) > 1:
                adj[a].add(b)
                adj[b].add(a)
    
    return candidates, adj

def _greedy_maximal_clique(adj, start_vertex, ordering=None):
    """
    Greedy approach: start from a vertex, repeatedly add a vertex that
    is connected to all current clique members.
    """
    clique = [start_vertex]
    if ordering is None:
        order = [v for v in adj if v != start_vertex]
    else:
        order = [v for v in ordering if v != start_vertex]
    
    for v in order:
        if all(u in adj[v] for u in clique):
            clique.append(v)
    
    return set(clique)

def _local_improve(clique, adj, max_rounds=100):
    """
    Try to add any vertex not in clique that is adjacent to all clique members.
    Then try to swap: remove one vertex and add two, if possible.
    """
    improved = True
    while improved and max_rounds > 0:
        improved = False
        max_rounds -= 1
        
        # Try adding
        for v in adj:
            if v not in clique and all(u in adj[v] for u in clique):
                clique.add(v)
                improved = True
                break
        
        if improved:
            continue
            
        # Try swap: remove one, add two
        # For small cliques this is expensive, so skip for now
        # Could implement later
    
    return clique

def _max_clique_branch_and_bound(adj, max_time=50.0):
    """
    Branch and bound to find the maximum clique in the candidate graph.
    Uses a simple coloring bound.
    """
    best_clique = set()
    
    # Initial solution: greedy from 255
    greedy = _greedy_maximal_clique(adj, _TARGET)
    greedy = _local_improve(greedy, adj)
    if len(greedy) > len(best_clique):
        best_clique = set(greedy)
    
    # Sort vertices by degree (descending) for better branching
    vertices = sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True)
    vertex_order = {v: i for i, v in enumerate(vertices)}
    
    deadline = time.time() + max_time
    
    # Tomita-style maximum clique algorithm with greedy coloring bound
    def greedy_coloring(P):
        """Color the subgraph induced by P using greedy coloring.
        Returns (coloring_dict, num_colors) where coloring_dict[v] = color.
        """
        color = {}
        u = 0
        # Sort P by degree in full graph (or within P)
        P_sorted = sorted(P, key=lambda v: len(adj[v] & set(P)), reverse=True)
        
        for v in P_sorted:
            used = set()
            for w in adj[v]:
                if w in color:
                    used.add(color[w])
            # Find smallest color not used
            c = 0
            while c in used:
                c += 1
            color[v] = c
            u = max(u, c + 1)
        
        return color, u
    
    best_size = len(best_clique)
    best_set = set(best_clique)
    
    def expand(R, P):
        nonlocal best_size, best_set, deadline
        if time.time() > deadline:
            return
        
        # If R + P cannot beat best, prune
        # Simple bound: |R| + |P| <= best_size
        if len(R) + len(P) <= best_size:
            return
        
        # If P is empty, update best
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_set = set(R)
            return
        
        # Greedy coloring for upper bound
        color, ub = greedy_coloring(P)
        
        if len(R) + ub <= best_size:
            return
        
        # Process vertices in reverse color order (high color first)
        # Actually, Tomita's algorithm processes in a specific order
        # Let's simplify: sort P by color
        P_sorted = sorted(P, key=lambda v: color[v])
        
        for i, v in enumerate(P_sorted):
            if time.time() > deadline:
                return
            if len(R) + ub - color[v] <= best_size:
                continue  # prune
            
            R.append(v)
            P_new = [x for x in P_sorted[i+1:] if x in adj[v]]
            expand(R, set(P_new))
            R.pop()
            
            if len(R) + len(P_new) <= best_size:
                break
        
    expand([], set(adj.keys()))
    
    return best_set

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # 55s budget out of 60s limit
    
    # Build the graph
    candidates, adj = _build_candidate_graph()
    
    # First, try simple multiplies-of-prime solutions
    best = set()
    
    # Multiples of 3
    mult3 = set(x for x in candidates if x % 3 == 0)
    if len(mult3) > len(best):
        best = set(mult3)
    
    # Multiples of 5
    mult5 = set(x for x in candidates if x % 5 == 0)
    if len(mult5) > len(best):
        best = set(mult5)
    
    # Multiples of 17
    mult17 = set(x for x in candidates if x % 17 == 0)
    if len(mult17) > len(best):
        best = set(mult17)
    
    # Try union of multiples of 3 and numbers divisible by 5 or 17 that are also divisible by 3...
    # Actually, a key insight: the set must be a clique. 
    # Consider the set of all multiples of 3 PLUS numbers that are divisible by 5 and 3? No, 
    # a number divisible by 5 but not 3 might not be coprime to all multiples of 3.
    # gcd(5, 3) = 1, so 5 cannot be added to the multiples of 3 set unless we remove 3 or 6, etc.
    
    # Let's run the branch and bound search
    if time.time() < deadline - 5:
        search_result = _max_clique_branch_and_bound(adj, max_time=min(deadline - time.time() - 2, 45.0))
        if len(search_result) > len(best):
            best = search_result
    
    # Verify the best solution
    if _is_valid_clique(best):
        return list(best)
    
    # Fallback to multiples of 3
    fallback = list(range(3, 256, 3))
    return fallback

# EVOLVE-BLOCK-END
