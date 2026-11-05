import time
import random
import sys
import math
import numpy as np

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    
    Strategy:
    1. Model the problem as finding a Maximum Independent Set (MIS) in a conflict graph G_c.
       - Vertices of G_c: The 448 edges of Q7.
       - Edges of G_c: Two vertices are connected if the corresponding Q7 edges belong to the same 2-dimensional face.
       - A valid C4-free edge set in Q7 corresponds to an independent set in G_c.
    
    2. Use Maximum Cardinality Search (MCS) to generate a vertex ordering.
       - MCS is a classic heuristic for graph coloring and can be adapted for MIS.
       - The idea: Order vertices such that each vertex has as many neighbors as possible among previously ordered vertices.
       - This creates a "clique-like" structure in the ordering, which helps in identifying dense regions to avoid or sparse regions to exploit.
       - Alternatively, we can use the reverse MCS ordering for a greedy independent set construction: 
         Process vertices in reverse MCS order. A vertex is added to the independent set if it is not adjacent to any previously added vertex.
         Since MCS orders vertices by increasing "constraint" (number of neighbors in the already ordered set), 
         the reverse order processes "least constrained" vertices first (in the context of the remaining graph? No, MCS is for coloring).
         
    3. Actually, a better interpretation for MIS:
       We want to select a subset of vertices with no edges between them.
       A greedy approach: 
       - Start with empty set S.
       - Repeatedly pick a vertex v that has the fewest neighbors in the *remaining* candidate set? Or fewest neighbors in the *conflict* graph?
       - The prompt suggests: "iteratively select the least constrained edge". 
       - "Least constrained" likely means: An edge e is constrained by the number of other edges it conflicts with. 
         If we pick e, we must exclude all its neighbors in G_c.
         To maximize the set, we should pick vertices that "sacrifice" the fewest other potential vertices.
         So, pick v with minimum degree in the current induced subgraph of candidates.
       
    4. However, the prompt specifically mentions MCS. 
       MCS for Coloring: Order v1, ..., vn such that deg(vi, {v1...vi-1}) is maximized.
       For MIS, a common heuristic is to use the reverse of a minimum degree ordering.
       Let's stick to the "Least Constrained" greedy:
       - Maintain a set of available edges A.
       - At each step, find e in A with the minimum number of neighbors in A.
       - Add e to the solution S.
       - Remove e and all its neighbors from A.
       - Repeat until A is empty.
       
    5. To improve on pure greedy, we will:
       - Use MCS to generate an initial ordering.
       - Use the reverse MCS order as a priority for a greedy pass.
       - Then, apply local search (swap/add/remove) to refine the result.
       
    6. Implementation Details:
       - Precompute the conflict graph adjacency list.
       - Implement MCS ordering.
       - Implement the greedy MIS construction using the MCS order (or min-degree).
       - Implement a local search loop:
         - Try to add an edge not in S if it doesn't conflict with S.
         - Try to swap an edge in S with a set of edges not in S if the net gain is positive (e.g., replace 1 edge with 2 non-conflicting edges).
         - Simulated Annealing style perturbation if stuck.
    
    7. Given the time limit, we will focus on a fast, high-quality greedy + local search.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    d = 7
    n = 2 ** d  # 128 vertices

    # 1. Precompute edges and faces
    edge_list = []
    edge_to_idx = {}
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edge_list)
                edge_list.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    E = len(edge_list)  # 448

    # Precompute faces (C4s)
    faces = []
    face_set = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i+1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_tuple = tuple(sorted([idx1, idx2, idx3, idx4]))
                
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    faces.append([idx1, idx2, idx3, idx4])
    
    F = len(faces)  # 672

    # Build Conflict Graph
    # Two edges conflict if they are in the same face.
    # For each face, all 4 edges form a clique in the conflict graph.
    conflict_adj = [set() for _ in range(E)]
    for face in faces:
        for i in range(4):
            for j in range(i+1, 4):
                e1 = face[i]
                e2 = face[j]
                conflict_adj[e1].add(e2)
                conflict_adj[e2].add(e1)
    
    # Convert to lists for faster iteration if needed, but sets are good for membership.
    # We also need degrees.
    conflict_deg = [len(s) for s in conflict_adj]

    # 2. MCS Ordering
    # Maximum Cardinality Search:
    # Pick a vertex, then repeatedly pick an unvisited vertex with the most neighbors in the visited set.
    # This gives an ordering v1, v2, ..., vE.
    
    def mcs_ordering():
        visited = [False] * E
        order = []
        # Counters for neighbors in visited set
        # Since E is small (448), we can just scan.
        # To speed up, we can maintain a priority queue or just scan each time.
        # Scanning 448 * 448 is ~200k ops, which is fast.
        
        # Initialize with a random start to avoid symmetry bias? Or deterministic 0?
        # Let's use random start for diversity in multiple runs.
        current = random.randint(0, E - 1)
        
        for _ in range(E):
            visited[current] = True
            order.append(current)
            
            # Find next vertex
            best_next = -1
            max_count = -1
            for v in range(E):
                if not visited[v]:
                    # Count neighbors of v in visited set
                    # This is O(deg(v)) if we iterate neighbors and check visited
                    # Or O(E) if we iterate all and check adjacency.
                    # deg is small (~12-15), so iterating neighbors is better.
                    count = 0
                    for nb in conflict_adj[v]:
                        if visited[nb]:
                            count += 1
                    if count > max_count:
                        max_count = count
                        best_next = v
            
            if best_next == -1:
                break
            current = best_next
        
        return order

    # 3. Greedy MIS Construction
    # We want to select an independent set.
    # Heuristic: Process vertices in an order. Add if possible.
    # Which order? 
    # If we use MCS order v1...vE, and process in reverse (vE...v1):
    # vE has the most neighbors in {v1...vE-1}. 
    # Actually, the standard greedy for MIS using MCS is:
    # Order vertices by MCS. 
    # Then, for i from 1 to E:
    #   If v_i is not adjacent to any v_j (j < i) that is in the set, add v_i.
    # This is equivalent to processing in forward order.
    
    # Let's try both forward and reverse MCS orderings and pick the best.
    # Also try "Minimum Degree" greedy on the remaining graph.

    def greedy_mis(order):
        selected = [False] * E
        count = 0
        for v in order:
            # Check if v conflicts with any already selected
            conflict = False
            for nb in conflict_adj[v]:
                if selected[nb]:
                    conflict = True
                    break
            if not conflict:
                selected[v] = True
                count += 1
        return selected, count

    def min_degree_greedy():
        # Maintain a set of available vertices
        available = set(range(E))
        selected = [False] * E
        count = 0
        
        while available:
            # Find vertex with minimum degree in the induced subgraph on 'available'
            min_deg = float('inf')
            best_v = -1
            for v in available:
                deg = 0
                for nb in conflict_adj[v]:
                    if nb in available:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            # Select best_v
            selected[best_v] = True
            count += 1
            available.remove(best_v)
            
            # Remove all neighbors of best_v from available
            to_remove = []
            for nb in conflict_adj[best_v]:
                if nb in available:
                    to_remove.append(nb)
            for nb in to_remove:
                available.remove(nb)
                
        return selected, count

    # 4. Local Search Refinement
    # Given a valid independent set (C4-free edge set), try to improve it.
    # Moves:
    # 1. Add: If there is an edge not in S that doesn't conflict with S, add it. (Shouldn't happen if S is maximal, but good to check).
    # 2. Swap: Remove one edge e from S. This "frees up" its neighbors. 
    #    Then, try to add as many of the newly available neighbors as possible (greedy).
    #    If the net change in size is positive, keep the move.
    
    def refine_solution(state, full_count, time_budget):
        # state: list of bools
        # full_count: list of ints (count of selected edges in each face)
        # We maintain validity: full_count[f] < 4 for all f.
        
        # Rebuild conflict info for fast "can add" check
        # An edge e can be added if for all faces f containing e, full_count[f] < 3.
        # (Because adding e makes count 4 if it was 3).
        
        best_state = state[:]
        best_score = sum(state)
        
        # Precompute face membership for edges
        edge_faces = [[] for _ in range(E)]
        for f_idx, face in enumerate(faces):
            for e_idx in face:
                edge_faces[e_idx].append(f_idx)
        
        # Local Search Loop
        while time.time() - start_time < time_budget:
            improved = False
            
            # Phase 1: Try to add edges
            # If S is maximal, no edge can be added. 
            # But after swaps, it might not be maximal.
            for e in range(E):
                if best_state[e]:
                    continue
                can_add = True
                for f_idx in edge_faces[e]:
                    if full_count[f_idx] == 3:
                        can_add = False
                        break
                if can_add:
                    best_state[e] = True
                    for f_idx in edge_faces[e]:
                        full_count[f_idx] += 1
                    best_score += 1
                    improved = True
            
            if improved:
                continue
            
            # Phase 2: Try 1-for-k swaps
            # For each edge e in S, try removing it and then greedily adding back as many neighbors as possible.
            # This is expensive if done naively. 
            # Optimization: Only try edges that are "high cost" (many neighbors) or "low gain" (few alternatives).
            # For simplicity, iterate all edges in S.
            
            found_swap = False
            for e in range(E):
                if not best_state[e]:
                    continue
                
                # Simulate removing e
                # Neighbors of e in conflict graph are edges that share a face with e.
                # When e is removed, these neighbors become *potentially* addable.
                # However, a neighbor might still conflict with other edges in S.
                
                # Let's identify the "candidates" to add: neighbors of e that are NOT in S.
                candidates = []
                for nb in conflict_adj[e]:
                    if not best_state[nb]:
                        candidates.append(nb)
                
                if not candidates:
                    continue
                
                # Greedily add candidates
                # We need to check if they can be added given the current S (with e removed).
                # Let's simulate.
                temp_state = best_state[:]
                temp_full = full_count[:]
                
                # Remove e
                temp_state[e] = False
                for f_idx in edge_faces[e]:
                    temp_full[f_idx] -= 1
                
                added_count = 0
                # Sort candidates by degree? Or just random?
                # Let's try adding them in order of fewest conflicts with current temp_state.
                # For speed, just try all.
                random.shuffle(candidates)
                
                for nb in candidates:
                    if temp_state[nb]:
                        continue
                    can_add = True
                    for f_idx in edge_faces[nb]:
                        if temp_full[f_idx] == 3:
                            can_add = False
                            break
                    if can_add:
                        temp_state[nb] = True
                        for f_idx in edge_faces[nb]:
                            temp_full[f_idx] += 1
                        added_count += 1
                
                net_gain = added_count - 1
                if net_gain > 0:
                    # Accept swap
                    best_state = temp_state
                    full_count = temp_full
                    best_score += net_gain
                    found_swap = True
                    break # Restart search with new state
            
            if not found_swap:
                # Local optimum reached for 1-for-k swaps.
                # Try perturbation: Remove 2-3 random edges to escape local optimum.
                if best_score < 308:
                    # Perturb
                    selected_edges = [i for i in range(E) if best_state[i]]
                    if len(selected_edges) > 3:
                        to_remove = random.sample(selected_edges, 3)
                        for e in to_remove:
                            best_state[e] = False
                            for f_idx in edge_faces[e]:
                                full_count[f_idx] -= 1
                        best_score -= 3
                        # Continue loop to try to add back

    # Main Execution
    random.seed(42)
    
    best_overall_state = None
    best_overall_score = 0
    
    # Run multiple strategies
    num_runs = 50 # Limit by time
    
    for run in range(num_runs):
        if time.time() - start_time > time_limit - 20:
            break
            
        # Generate MCS Order
        order = mcs_ordering()
        
        # Try forward MCS
        state_fwd, score_fwd = greedy_mis(order)
        # Try reverse MCS
        state_rev, score_rev = greedy_mis(order[::-1])
        
        # Try Min-Degree Greedy
        state_md, score_md = min_degree_greedy()
        
        # Pick best initial
        candidates = [
            (state_fwd, score_fwd),
            (state_rev, score_rev),
            (state_md, score_md)
        ]
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_state, best_score = candidates[0]
        
        # Prepare full_count for refinement
        full_count = [0] * F
        for f_idx in range(F):
            f = faces[f_idx]
            full_count[f_idx] = best_state[f[0]] + best_state[f[1]] + best_state[f[2]] + best_state[f[3]]
        
        # Refine
        time_budget = time_limit - 5
        best_state, best_score = refine_solution(best_state, full_count, time_budget)
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_state = best_state[:]
            
        if best_overall_score >= 308:
            break

    # Final Output
    if best_overall_state is None:
        # Fallback: Just return empty or a trivial set
        return []
        
    result = []
    for i in range(E):
        if best_overall_state[i]:
            u, v = edge_list[i]
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")
