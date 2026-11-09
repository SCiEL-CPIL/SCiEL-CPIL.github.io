"""Rewritten program: construct a 31-vertex, 91-edge graph minimizing the number of C4 subgraphs."""
import numpy as np
import random
import time

N = 31
E = 91
DEADLINE_SECONDS = 145.0  # safety margin under the 150 s limit

def _count_c4s(neighbors):
    """Count the number of C4 subgraphs in the graph.
    For each pair (u,v), count C(common_neighbors(u,v), 2).
    Each C4 is counted twice (once per diagonal), so divide by 2.
    """
    count = 0
    for u in range(N):
        for v in range(u + 1, N):
            common = neighbors[u] & neighbors[v]
            c = len(common)
            if c >= 2:
                count += c * (c - 1) // 2
    return count // 2

def _c4_increase_add_edge(u, v, neighbors):
    """Count new C4s created by adding edge (u,v).
    A new C4 must use (u,v) as one of its edges.
    The C4 is u-v-w-x-u where w~v, x~u, w~x, w!=x, w!=u, x!=v.
    Each such C4 is counted once.
    """
    n_u = neighbors[u] - {v}
    n_v = neighbors[v] - {u}
    count = 0
    for w in n_v:
        for x in n_u:
            if w != x and x in neighbors[w]:
                count += 1
    return count

def _c4_decrease_remove_edge(u, v, neighbors):
    """Count C4s destroyed by removing edge (u,v).
    Same formula as add: C4s that use (u,v) as an edge.
    """
    n_u = neighbors[u] - {v}
    n_v = neighbors[v] - {u}
    count = 0
    for w in n_v:
        for x in n_u:
            if w != x and x in neighbors[w]:
                count += 1
    return count

def _build_greedy_graph():
    """Build a graph with E edges using greedy addition minimizing C4 increase."""
    neighbors = [set() for _ in range(N)]
    edge_set = set()
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    for _ in range(E):
        # Find the edge with minimum C4 increase
        best_edge = None
        best_incr = float('inf')
        random.shuffle(all_edges)
        # Only check a subset for speed
        for u, v in all_edges[:200]:
            if (u, v) in edge_set:
                continue
            incr = _c4_increase_add_edge(u, v, neighbors)
            if incr < best_incr:
                best_incr = incr
                best_edge = (u, v)
            if best_incr == 0:
                break
        if best_edge is None:
            # Fallback: pick any unused edge
            for u, v in all_edges:
                if (u, v) not in edge_set:
                    best_edge = (u, v)
                    break
        u, v = best_edge
        neighbors[u].add(v)
        neighbors[v].add(u)
        edge_set.add((u, v))
    
    return list(edge_set), neighbors

def construct():
    start_time = time.time()
    random.seed(12345)
    
    best_edges = None
    best_c4 = float('inf')
    
    # Phase 1: Try multiple greedy constructions with different randomizations
    for trial in range(8):
        if time.time() - start_time > DEADLINE_SECONDS - 80:
            break
        edges_list, neighbors = _build_greedy_graph()
        c4_count = _count_c4s(neighbors)
        if c4_count < best_c4:
            best_c4 = c4_count
            best_edges = edges_list[:]
    
    if best_edges is None:
        best_edges = [(i, j) for i in range(N) for j in range(i + 1, N)][:E]
        neighbors = [set() for _ in range(N)]
        for u, v in best_edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        best_c4 = _count_c4s(neighbors)
    
    # Phase 2: Local search with edge swaps
    edge_set = set(best_edges)
    neighbors = [set() for _ in range(N)]
    for u, v in best_edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    current_c4 = best_c4
    all_possible_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    deadline_phase2 = start_time + DEADLINE_SECONDS - 5
    
    stagnation_counter = 0
    max_stagnation = 50
    
    while time.time() < deadline_phase2 and stagnation_counter < max_stagnation:
        improved = False
        # Try multiple random swaps
        for _ in range(50):
            if time.time() > deadline_phase2:
                break
            rem_edge = random.choice(list(edge_set))
            add_edge = random.choice(all_possible_edges)
            if add_edge in edge_set:
                continue
            
            u1, v1 = rem_edge
            u2, v2 = add_edge
            
            # Compute delta C4 for removing (u1,v1) and adding (u2,v2)
            # Remove (u1,v1): lose C4s that use this edge
            rem_c4s = _c4_decrease_remove_edge(u1, v1, neighbors)
            # Add (u2,v2): gain C4s that use this edge (in graph without (u1,v1))
            # Temporarily remove (u1,v1) to compute add cost accurately
            neighbors[u1].remove(v1)
            neighbors[v1].remove(u1)
            edge_set.remove(rem_edge)
            
            add_c4s = _c4_increase_add_edge(u2, v2, neighbors)
            
            delta = add_c4s - rem_c4s
            
            if delta < 0:
                # Accept the swap
                neighbors[u2].add(v2)
                neighbors[v2].add(u2)
                edge_set.add(add_edge)
                current_c4 += delta
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = list(edge_set)
                improved = True
                stagnation_counter = 0
            else:
                # Revert
                neighbors[u1].add(v1)
                neighbors[v1].add(u1)
                edge_set.add(rem_edge)
        
        if not improved:
            stagnation_counter += 1
            # Random restart: perturb the current solution
            if stagnation_counter % 5 == 0:
                # Remove a random edge and add a different one regardless of delta
                rem_edge = random.choice(list(edge_set))
                add_edge = random.choice(all_possible_edges)
                if add_edge not in edge_set:
                    u1, v1 = rem_edge
                    u2, v2 = add_edge
                    neighbors[u1].remove(v1)
                    neighbors[v1].remove(u1)
                    edge_set.remove(rem_edge)
                    neighbors[u2].add(v2)
                    neighbors[v2].add(u2)
                    edge_set.add(add_edge)
                    current_c4 = _count_c4s(neighbors)
    
    # Return the best found
    final_edges = list(set(best_edges))
    final_edges = sorted([(min(u, v), max(u, v)) for u, v in final_edges])
    # Ensure exactly E edges
    if len(final_edges) < E:
        all_possible = [(i, j) for i in range(N) for j in range(i + 1, N)]
        existing = set(final_edges)
        for u, v in all_possible:
            if len(final_edges) >= E:
                break
            if (u, v) not in existing:
                final_edges.append((u, v))
                existing.add((u, v))
        final_edges = sorted(final_edges)
    elif len(final_edges) > E:
        final_edges = final_edges[:E]
    
    return final_edges

if __name__ == "__main__":
    edges = construct()
    print("Number of edges:", len(edges))
    neighbors = [set() for _ in range(N)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    print("C4 count:", _count_c4s(neighbors))
