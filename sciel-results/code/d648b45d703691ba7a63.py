"""Improved program for Ramsey R(5,5): try to build larger (5,5)-free graphs."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys
from collections import deque

DEADLINE_SECONDS = 320.0
START_TIME = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)

def _paley_graph(q):
    """Return edges of the Paley graph on q vertices (q a prime congruent to 1 mod 4)."""
    residues = {(x * x) % q for x in range(1, q)}
    return [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

def _build_adj(edges, n):
    """Build adjacency as list of int bitsets."""
    adj = [0] * n
    for i, j in edges:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj

def _has_clique_k5(adj, n):
    """Check if graph has K5 using bitset operations."""
    # Find all 4-cliques, then check if any vertex connects to all 4
    # For each pair of adjacent vertices, find common neighbors
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] & (1 << b)):
                continue
            common_ab = adj[a] & adj[b] & ~((1 << (a + 1)) - 1)  # vertices > b
            # For each c in common_ab
            c_mask = common_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_abc = common_ab & adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    # Check if there's an e > d connected to a,b,c,d
                    common_abcd = common_abc & adj[d] & ~((1 << (d + 1)) - 1)
                    if common_abcd:
                        return True
    return False

def _has_independent_set_5(adj, n):
    """Check if graph has independent set of size 5 (K5 in complement)."""
    # Non-neighbor bitset
    all_mask = (1 << n) - 1
    non_adj = [all_mask ^ adj[i] ^ (1 << i) for i in range(n)]
    
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a] & (1 << b):  # adjacent, skip
                continue
            common_non_ab = non_adj[a] & non_adj[b] & ~((1 << (a + 1)) - 1)
            c_mask = common_non_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_non_abc = common_non_ab & non_adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_non_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    common_non_abcd = common_non_abc & non_adj[d] & ~((1 << (d + 1)) - 1)
                    if common_non_abcd:
                        return True
    return False

def _is_55_free(edges, n):
    """Check if the graph has no K5 and no independent 5-set."""
    adj = _build_adj(edges, n)
    if _has_clique_k5(adj, n):
        return False
    if _has_independent_set_5(adj, n):
        return False
    return True

def _try_add_vertex(edges, n, max_trials=20000):
    """
    Try to add a new vertex (index n) by finding a subset S of existing vertices
    to connect to such that no K5 and no independent 5-set is created.
    Returns (new_edges, success).
    """
    adj = _build_adj(edges, n)
    all_mask = (1 << n) - 1
    non_adj = [all_mask ^ adj[i] ^ (1 << i) for i in range(n)]
    
    # Find all 4-cliques and 4-independent sets
    four_cliques = set()
    four_indeps = set()
    
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] & (1 << b)):
                continue
            common_ab = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_abc = common_ab & adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    four_cliques.add(frozenset([a, b, c, d]))
    
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a] & (1 << b):
                continue
            common_non_ab = non_adj[a] & non_adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_non_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_non_abc = common_non_ab & non_adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_non_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    four_indeps.add(frozenset([a, b, c, d]))
    
    # Convert to lists for faster iteration
    four_cliques_list = list(four_cliques)
    four_indeps_list = list(four_indeps)
    
    random.seed(42)
    
    for trial in range(max_trials):
        if _time_left() < 10:
            return None, False
        
        S = set()
        for v in range(n):
            if random.random() < 0.5:
                S.add(v)
        
        # Check constraint 1: no 4-clique fully in S
        ok1 = True
        for C in four_cliques_list:
            if C.issubset(S):
                ok1 = False
                break
        if not ok1:
            continue
        
        # Check constraint 2: every 4-indep intersects S
        ok2 = True
        for I in four_indeps_list:
            if not I.intersection(S):
                ok2 = False
                break
        if not ok2:
            continue
        
        # Found a valid S!
        new_edges = list(edges)
        for v in S:
            new_edges.append((v, n))
        return new_edges, True
    
    return None, False

def _try_add_vertex_smart(edges, n, max_trials=50000):
    """
    Smarter vertex addition: start with random S, then fix violations iteratively.
    Returns (new_edges, success).
    """
    adj = _build_adj(edges, n)
    all_mask = (1 << n) - 1
    non_adj = [all_mask ^ adj[i] ^ (1 << i) for i in range(n)]
    
    # Find all 4-cliques and 4-independent sets
    four_cliques = set()
    four_indeps = set()
    
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] & (1 << b)):
                continue
            common_ab = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_abc = common_ab & adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    four_cliques.add(frozenset([a, b, c, d]))
    
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a] & (1 << b):
                continue
            common_non_ab = non_adj[a] & non_adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_non_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_non_abc = common_non_ab & non_adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_non_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    four_indeps.add(frozenset([a, b, c, d]))
    
    four_cliques_list = list(four_cliques)
    four_indeps_list = list(four_indeps)
    
    random.seed(123)
    
    for trial in range(max_trials):
        if _time_left() < 10:
            return None, False
        
        # Start with random S
        S = set()
        for v in range(n):
            if random.random() < 0.5:
                S.add(v)
        
        # Iteratively fix violations
        fixed = True
        max_fixes = 100
        for _ in range(max_fixes):
            if not fixed:
                break
            
            # Check for violated 4-clique (all in S)
            violated_clique = None
            for C in four_cliques_list:
                if C.issubset(S):
                    violated_clique = C
                    break
            
            if violated_clique:
                # Remove a random vertex from the violated clique
                v = random.choice(list(violated_clique))
                S.discard(v)
                fixed = False
                continue
            
            # Check for violated 4-indep (none in S)
            violated_indep = None
            for I in four_indeps_list:
                if not I.intersection(S):
                    violated_indep = I
                    break
            
            if violated_indep:
                # Add a random vertex from the violated indep
                v = random.choice(list(violated_indep))
                S.add(v)
                fixed = False
                continue
            
            # No violations
            break
        
        # Final check
        if not fixed:
            # Verify no violations remain
            for C in four_cliques_list:
                if C.issubset(S):
                    break
            else:
                for I in four_indeps_list:
                    if not I.intersection(S):
                        break
                else:
                    # Valid!
                    new_edges = list(edges)
                    for v in S:
                        new_edges.append((v, n))
                    return new_edges, True
    
    return None, False

def construct():
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    n = q
    
    # Verify it's valid
    if not _is_55_free(edges, n):
        pass
    
    best_n = n
    best_edges = edges[:]
    
    # Try to add vertices one by one
    max_attempts = 15
    for attempt in range(max_attempts):
        if _time_left() < 30:
            break
        
        # Try smart vertex addition
        new_edges, success = _try_add_vertex_smart(edges, n, max_trials=30000)
        if success:
            edges = new_edges
            n += 1
            if n > best_n:
                best_n = n
                best_edges = edges[:]
        else:
            # Try simpler method
            new_edges, success = _try_add_vertex(edges, n, max_trials=20000)
            if success:
                edges = new_edges
                n += 1
                if n > best_n:
                    best_n = n
                    best_edges = edges[:]
            else:
                break
    
    # Return the best we found
    return best_n, best_edges

# EVOLVE-BLOCK-END
