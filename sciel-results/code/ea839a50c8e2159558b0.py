import time
import random
import sys

def _check_valid(adj, deg, n):
    """
    Check if the graph defined by adj and deg satisfies the constraint:
    No simple 3-edge path s-u-v-t has deg(s) == deg(t).
    """
    for u in range(n):
        du = adj[u]
        for v in du:
            if v <= u:
                continue
            dv = adj[v]
            # Path s - u - v - t
            # s in adj[u], t in adj[v]
            # s != v, t != u, s != t
            for s in du:
                if s == v:
                    continue
                ds = deg[s]
                for t in dv:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _find_violation_edge(adj, deg, n):
    """
    Find an edge (u, v) that is part of a violating path.
    Returns (u, v) or None.
    """
    for u in range(n):
        du = adj[u]
        for v in du:
            if v <= u:
                continue
            dv = adj[v]
            for s in du:
                if s == v:
                    continue
                ds = deg[s]
                for t in dv:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        # Violation found on path s-u-v-t
                        # We can return any edge in this path, e.g., (u, v)
                        return (u, v)
    return None

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin
    n = 15
    rng = random.Random(12345)
    
    # Initialize with complete graph
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            adj[i].add(j)
            adj[j].add(i)
    
    deg = [len(adj[i]) for i in range(n)]
    
    # Phase 1: Prune to make valid
    # We remove edges involved in violations until valid
    # To be efficient, we just pick a random edge from the violation path
    while not _check_valid(adj, deg, n):
        if time.time() > deadline:
            break
        violation = _find_violation_edge(adj, deg, n)
        if violation:
            u, v = violation
            # Remove this edge
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1
        else:
            # Should not happen if check_valid is consistent
            break

    # Now we have a valid graph
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    best_edges = [list(e) for e in edges]
    best_count = len(edges)
    
    def evaluate_and_track(edges_list):
        nonlocal best_edges, best_count
        if len(edges_list) > best_count:
            best_count = len(edges_list)
            best_edges = [list(e) for e in edges_list]

    evaluate_and_track(edges)

    # Phase 2: Local Search (Add, Swap, Kick)
    # We maintain adj, deg, edges list
    # To speed up, we keep a set of present edges
    present = set(edges)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        edges.append((u, v))
        present.add((u, v))

    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1
        edges.remove((u, v))
        present.discard((u, v))

    def get_edges_list():
        return [list(e) for e in edges]

    # Main Loop
    # We try to add edges, then swap, then kick (remove random)
    # Since N=15 is small, we can be very thorough
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add any missing edge
        for u, v in all_pairs:
            if (u, v) in present:
                continue
            # Check if adding (u, v) keeps it valid
            # We can do a fast check:
            # Adding (u, v) creates new paths.
            # Old paths remain valid (degrees of other nodes don't change, 
            # but degrees of u, v change, which might invalidate OLD paths involving u or v).
            # Wait, if we add (u, v), deg(u) and deg(v) increase.
            # This might make deg(u) == deg(s) for some s, creating a new violation
            # on an existing path s-...-u or u-...-s.
            # So we MUST check the whole graph.
            # But _check_valid is fast.
            
            # Tentative add
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            
            if _check_valid(adj, deg, n):
                # Accept
                edges.append((u, v))
                present.add((u, v))
                improved = True
                evaluate_and_track(edges)
            else:
                # Reject
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
                
        if improved:
            no_improve_count = 0
            continue
        else:
            no_improve_count += 1
            
        # 2. Try Swaps: Remove one, Add another
        # Only if we are stuck
        if no_improve_count < max_no_improve:
            # Try a few random swaps
            swaps_tried = 0
            while swaps_tried < 50 and time.time() < deadline:
                swaps_tried += 1
                if not edges:
                    break
                # Pick random edge to remove
                e1_idx = rng.randint(0, len(edges) - 1)
                u1, v1 = edges[e1_idx]
                
                # Pick random missing edge to add
                missing = [(u, v) for u, v in all_pairs if (u, v) not in present]
                if not missing:
                    break
                u2, v2 = rng.choice(missing)
                
                if (u1, v1) == (u2, v2):
                    continue
                
                # Tentative swap
                # Remove e1
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                deg[u1] -= 1
                deg[v1] -= 1
                edges.pop(e1_idx)
                present.discard((u1, v1))
                
                # Add e2
                adj[u2].add(v2)
                adj[v2].add(u2)
                deg[u2] += 1
                deg[v2] += 1
                edges.append((u2, v2))
                present.add((u2, v2))
                
                if _check_valid(adj, deg, n):
                    # Accept swap (edge count same, but might enable future adds)
                    # We don't update best_edges if count is same, unless we track structure?
                    # No, we only care about count. But structure change might help.
                    # So we accept it to explore.
                    improved = True
                    # Do not update best_edges if count is not higher
                    # But we continue with this new structure
                else:
                    # Revert
                    adj[u2].discard(v2)
                    adj[v2].discard(u2)
                    deg[u2] -= 1
                    deg[v2] -= 1
                    edges.pop() # Remove e2
                    present.discard((u2, v2))
                    
                    # Restore e1
                    edges.insert(e1_idx, (u1, v1))
                    present.add((u1, v1))
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    deg[u1] += 1
                    deg[v1] += 1
                    
        if improved:
            no_improve_count = 0
            continue
            
        # 3. Kick: Remove a random edge to allow different structure
        # This decreases edge count, but might lead to a much denser valid graph later
        if edges:
            e1_idx = rng.randint(0, len(edges) - 1)
            u1, v1 = edges[e1_idx]
            remove_edge(u1, v1)
            # After removing, we might be able to add different edges
            # So we reset no_improve to try adding again
            no_improve_count = 0
        else:
            # If no edges, we are stuck at 0? Unlikely.
            break

    # Final result
    result = []
    seen = set()
    for u, v in best_edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
            
    return result

# To ensure we don't crash if called in a weird way, we wrap construct
# But the task requires keeping function names.

if __name__ == "__main__":
    # Optional: Run a quick test
    # edges = construct()
    # print(len(edges))
    pass
