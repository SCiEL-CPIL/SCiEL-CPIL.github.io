"""Maximum Independent Set on conflict graph for Property P."""
import random
import sys

N = 256

def _build_conflict_graph():
    """
    Build adjacency list for conflict graph.
    Edge {b,c} (b<=c) exists if there exists a < b such that a | (b+c).
    This ensures that any independent set satisfies the property.
    """
    adj = [set() for _ in range(N + 1)]
    
    for b in range(1, N + 1):
        for c in range(b, N + 1):
            s = b + c
            # Check if any a < b divides s
            has_conflict = False
            for a in range(1, b):
                if s % a == 0:
                    has_conflict = True
                    break
            if has_conflict:
                adj[b].add(c)
                adj[c].add(b)
    
    return adj

def _greedy_mis(adj, order):
    """
    Greedy Maximum Independent Set: process vertices in given order,
    add vertex if none of its neighbors have been added.
    """
    in_set = [False] * (N + 1)
    S = []
    for v in order:
        if in_set[v]:
            continue
        # Check if any neighbor is already in the set
        can_add = True
        for u in adj[v]:
            if in_set[u]:
                can_add = False
                break
        if can_add:
            in_set[v] = True
            S.append(v)
    return S

def _verify(S):
    """Verify the property: for all a<b<=c in S, a does not divide b+c."""
    S_sorted = sorted(S)
    n = len(S_sorted)
    for i in range(n):
        a = S_sorted[i]
        for j in range(i + 1, n):
            b = S_sorted[j]
            for k in range(j, n):
                c = S_sorted[k]
                if (b + c) % a == 0:
                    return False
    return True

def construct():
    """
    Find a large valid subset of [1, 256] using MIS on conflict graph.
    """
    adj = _build_conflict_graph()
    
    best_set = list(range(192, 257))  # Fallback
    best_size = len(best_set)
    
    random.seed(42)
    
    # Try multiple random orderings
    for restart in range(200):
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = _greedy_mis(adj, order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Verify the best set
    if not _verify(best_set):
        # Fallback to a known valid set
        best_set = list(range(192, 257))
    
    return best_set
