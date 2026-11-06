"""Construct a large (5,5)-free graph via vertex growth with fast violation detection."""
import random
import time
from itertools import combinations


def _paley_graph(q):
    """Return adjacency list (list of sets) for Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [set() for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _k4s_in_set(n, adj, vertex_set):
    """Yield all K4s within vertex_set. vertex_set is a set of vertices."""
    vs = list(vertex_set)
    m = len(vs)
    if m < 4:
        return
    vs_set = set(vs)
    for i in range(m):
        a = vs[i]
        for j in range(i + 1, m):
            b = vs[j]
            if b not in adj[a]:
                continue
            common = adj[a] & adj[b] & vs_set
            common_list = [v for v in common if v > b]  # avoid duplicates
            L = len(common_list)
            if L < 2:
                continue
            for k in range(L):
                c = common_list[k]
                for l in range(k + 1, L):
                    d = common_list[l]
                    if d in adj[c]:
                        yield (a, b, c, d)


def _empty4s_in_set(n, adj, vertex_set):
    """Yield all independent 4-sets within vertex_set."""
    vs = list(vertex_set)
    m = len(vs)
    if m < 4:
        return
    vs_set = set(vs)
    # Precompute non-adjacency within vs
    nonadj = {}
    for v in vs:
        nonadj[v] = vs_set - adj[v] - {v}
    for i in range(m):
        va = vs[i]
        na = nonadj[va]
        for j in range(i + 1, m):
            vb = vs[j]
            if vb not in na:
                continue
            nb = nonadj[vb]
            common_ab = na & nb
            common_list = [v for v in common_ab if v > vb]
            for c in common_list:
                nc = nonadj[c]
                common_abc = common_ab & nc
                for d in common_abc:
                    if d > c:
                        yield (va, vb, c, d)


def _has_k5(n, adj):
    """Check if graph contains a K5. Returns True if found."""
    for v in range(n):
        nbrs = adj[v]
        # Check for K4 in neighborhood of v
        for k4 in _k4s_in_set(n, adj, nbrs):
            return True
    return False


def _has_empty5(n, adj):
    """Check if graph contains an independent set of size 5. Returns True if found."""
    for v in range(n):
        non_nbrs = set(range(n)) - adj[v] - {v}
        for ind4 in _empty4s_in_set(n, adj, non_nbrs):
            return True
    return False


def _is_valid(n, adj):
    """Check if graph is (5,5)-free."""
    if _has_k5(n, adj):
        return False
    if _has_empty5(n, adj):
        return False
    return True


def _find_safe_neighbor_set(adj, n, time_limit):
    """
    Find a set S of neighbors for a new vertex such that:
    - S contains no K4
    - V\S contains no independent 4-set
    
    Returns S as a set, or None if no valid S found.
    """
    vertices = set(range(n))
    deadline = time.time() + time_limit
    
    # Try multiple random initializations
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Balanced start
            S = set(random.sample(range(n), n // 2))
        elif attempt == 1:
            # Start with all vertices
            S = set(range(n))
        elif attempt == 2:
            # Start with empty
            S = set()
        else:
            S = set()
            for i in range(n):
                if random.random() < 0.5:
                    S.add(i)
        
        # Iteratively fix violations
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Check if S contains any K4
            violated = False
            for k4 in _k4s_in_set(n, adj, S):
                # Remove one random vertex from this K4 from S
                v = random.choice(k4)
                S.discard(v)
                violated = True
                break
            
            if violated:
                continue
            
            # Check if V\S contains any independent 4-set
            non_S = vertices - S
            violated = False
            for ind4 in _empty4s_in_set(n, adj, non_S):
                # Add one random vertex from this ind4 to S
                v = random.choice(ind4)
                S.add(v)
                violated = True
                break
            
            if not violated:
                # Verify
                ok = True
                for k4 in _k4s_in_set(n, adj, S):
                    ok = False
                    break
                if ok:
                    non_S = vertices - S
                    for ind4 in _empty4s_in_set(n, adj, non_S):
                        ok = False
                        break
                if ok:
                    return S
    
    return None


def _add_vertex(adj, n, S):
    """Add a new vertex n with neighbor set S. Returns new adjacency list."""
    new_adj = [set() for _ in range(n + 1)]
    for i in range(n):
        new_adj[i] = set(adj[i])
    new_adj[n] = set()
    for i in S:
        new_adj[i].add(n)
        new_adj[n].add(i)
    return new_adj


def _local_search(adj, n, deadline):
    """
    Try to improve the graph by flipping edges. Returns (adj, n) possibly modified.
    Only accepts flips that keep the graph valid.
    """
    best_adj = [set(s) for s in adj]
    best_n = n
    best_score = -1
    
    # Score: prefer graphs that are "close" to having a valid extension
    # We'll just try to maintain validity and see if we can add vertices later
    
    for step in range(500):
        if time.time() > deadline:
            break
        
        # Try a random edge flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        
        was_edge = j in adj[i]
        
        # Flip
        if was_edge:
            adj[i].discard(j)
            adj[j].discard(i)
        else:
            adj[i].add(j)
            adj[j].add(i)
        
        # Check if still valid
        if _is_valid(n, adj):
            # Accept this flip
            # Check if we can now add a vertex
            remaining = deadline - time.time()
            if remaining > 3.0:
                S = _find_safe_neighbor_set(adj, n, min(2.0, remaining * 0.3))
                if S is not None:
                    adj = _add_vertex(adj, n, S)
                    n += 1
                    best_n = n
                    best_adj = [set(s) for s in adj]
                    break  # Restart local search with new graph
        else:
            # Revert
            if was_edge:
                adj[i].add(j)
                adj[j].add(i)
            else:
                adj[i].discard(j)
                adj[j].discard(i)
    
    return adj, n


def construct():
    start_time = time.time()
    deadline = start_time + 310  # Safety margin before 330s kill
    
    # Start with Paley graph on 37 vertices
    q = 37
    adj = _paley_graph(q)
    n = q
    
    # Verify initial graph is valid
    if not _is_valid(adj, n):
        return n, [(i, j) for i in range(n) for j in range(i + 1, n) if j in adj[i]]
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to grow the graph by adding vertices
    while time.time() < deadline:
        if n >= 42:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 5.0:
            break
        
        # Find a safe neighbor set for the new vertex
        search_time = min(5.0, remaining_time * 0.4)
        
        S = _find_safe_neighbor_set(adj, n, search_time)
        if S is not None:
            adj = _add_vertex(adj, n, S)
            n += 1
            best_n = n
            best_adj = [set(s) for s in adj]
        
        # If we can't add a vertex, try local search to modify the graph
        else:
            # Try local search with a time budget
            ls_time = min(10.0, remaining_time * 0.3)
            adj, n = _local_search(adj, n, time.time() + ls_time)
            best_n = n
            best_adj = [set(s) for s in adj]
    
    # Final validity check
    if not _is_valid(best_adj, best_n):
        # Fallback to seed
        adj37 = _paley_graph(37)
        return 37, [(i, j) for i in range(37) for j in range(i + 1, 37) if j in adj37[i]]
    
    edges = [(i, j) for i in range(best_n) for j in range(i + 1, best_n) if j in best_adj[i]]
    return best_n, edges
