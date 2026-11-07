"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Conflict Graph Construction
    # Node i in conflict graph represents edge all_edges[i] in candidate graph.
    # Edge (i, j) in conflict graph means edge i and edge j CANNOT be in the same valid candidate graph.
    
    # Why conflict?
    # If we have a set of edges E, it is valid iff for every path s-u-v-t in E, deg(s) != deg(t).
    # This is a global constraint. It is hard to express as pairwise conflicts directly because 
    # "deg(s)" depends on the whole set E.
    # 
    # However, the constraint is: For any u, v in E, and any s in N_E(u)\{v}, t in N_E(v)\{u}, deg_E(s) != deg_E(t).
    # This is not a simple 2-SAT or pairwise constraint. It depends on degrees which are sums.
    #
    # Therefore, a simple Maximum Independent Set on a pre-computed conflict graph is NOT directly applicable 
    # because validity is not a pairwise property of edges. Two edges might be fine together, but 
    # adding a third changes degrees and makes the first two invalid.
    #
    # RE-THINKING:
    # The current program uses a heuristic approach because the problem is likely NP-hard or at least 
    # complex to solve exactly via simple reductions.
    #
    # Let's look at the "Inspiration" which scored 3.571. It used "Randomized greedy edge insertion 
    # with a fast per-edge validity check... plus drop-one local search".
    # The current program (score 1.0) is similar but perhaps less effective or stuck in a local optima 
    # (star graph has 14 edges, score 1.0 implies seed value is 14? No, score is value/seed. 
    # If seed is 14, score 1.0 means value 14. If improved to 3.571, value is ~50?).
    #
    # Let's try a different strategy: **Simulated Annealing** with a very strong move set.
    # The current program does Greedy + Simple Swaps.
    # Simulated Annealing allows accepting worse solutions temporarily to escape local optima.
    #
    # Strategy:
    # 1. Define a state as a set of edges.
    # 2. Define moves:
    #    - Add an edge
    #    - Remove an edge
    #    - Swap an edge (remove one, add another)
    #    - 2-Swap (remove two, add two)
    # 3. Objective: Maximize edge count.
    # 4. Constraint: Validity.
    #    - If invalid, we can either reject the move or accept it with a penalty.
    #    - Better: Only consider valid states? That makes search space disconnected.
    #    - Alternative: Define a score function that penalizes invalidity heavily.
    #      Score = num_edges - 100 * num_violations.
    #      Then we can move between invalid states if it helps escape a local minimum in the valid region.
    #
    # Let's implement a fast validator that counts violations, not just boolean.
    # Violation: A path s-u-v-t where deg(s) == deg(t).
    #
    # Let's try a **Local Search with Random Restarts and 2-Swaps**.
    # The key insight from the inspiration is that "drop-one" (removing an edge to fix a violation 
    # or to allow adding two) is powerful.
    #
    # Actually, let's look at the structure of the constraint again.
    # deg(s) != deg(t) for all s~u, t~v, u~v, s!=t, s!=v, t!=u.
    #
    # If we fix the degrees of all vertices, the constraint becomes:
    # For each edge (u,v), let S = {deg(s) | s~u, s!=v} and T = {deg(t) | t~v, t!=u}.
    # We require S ∩ T = ∅.
    #
    # This suggests that if we can find a degree sequence that is "compatible", we can try to 
    # realize it. But realizing a degree sequence is itself a hard problem (graph realization).
    #
    # Let's stick to the heuristic search but improve it significantly.
    # 1. Fast validity check using bitmasks or sets of degrees.
    # 2. Move set: Add, Remove, Swap, 2-Add-1-Remove, 1-Add-2-Remove.
    # 3. Simulated Annealing to escape local optima.
    
    def get_degrees_and_adj(edges):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        return deg, adj

    def count_violations(edges):
        """Returns the number of violating paths."""
        if not edges:
            return 0
        deg, adj = get_degrees_and_adj(edges)
        violations = 0
        for u in range(n):
            # Neighbors of u excluding v
            # We iterate over edges (u, v) in the graph
            for v in adj[u]:
                # S = {deg(s) for s in adj[u] if s != v}
                # T = {deg(t) for t in adj[v] if t != u}
                # If S and T intersect, there is a violation.
                # Specifically, for each d in S ∩ T, there is at least one s with deg(s)=d and one t with deg(t)=d.
                # The number of violating paths for this (u,v) is the sum over d in S∩T of (count_s[d] * count_t[d]).
                
                deg_s_counts = {}
                for s in adj[u]:
                    if s != v:
                        d = deg[s]
                        deg_s_counts[d] = deg_s_counts.get(d, 0) + 1
                
                if not deg_s_counts:
                    continue
                
                for t in adj[v]:
                    if t != u:
                        dt = deg[t]
                        if dt in deg_s_counts:
                            violations += deg_s_counts[dt]
        return violations

    def is_valid(edges):
        return count_violations(edges) == 0

    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.01
    max_steps = 50000
    
    def random_initial_state():
        # Start with a random sparse graph
        k = random.randint(10, 20)
        random.shuffle(all_edges)
        return all_edges[:k]
    
    def neighbor_state(state):
        """Generate a random neighbor state."""
        state_set = set(state)
        cur_edges = list(state)
        
        # Choose move type
        r = random.random()
        
        if r < 0.4:
            # Add an edge
            missing = [e for e in all_edges if e not in state_set]
            if not missing:
                return cur_edges
            e = random.choice(missing)
            cur_edges.append(e)
        elif r < 0.7:
            # Remove an edge
            if not cur_edges:
                return cur_edges
            e = random.choice(cur_edges)
            cur_edges.remove(e)
        elif r < 0.9:
            # Swap
            if not cur_edges:
                return cur_edges
            missing = [e for e in all_edges if e not in state_set]
            if not missing:
                return cur_edges
            e1 = random.choice(cur_edges)
            e2 = random.choice(missing)
            cur_edges.remove(e1)
            cur_edges.append(e2)
        else:
            # 2-Swap: Remove 2, Add 2
            if len(cur_edges) < 2:
                return cur_edges
            missing = [e for e in all_edges if e not in state_set]
            if len(missing) < 2:
                return cur_edges
            e1, e2 = random.sample(cur_edges, 2)
            e3, e4 = random.sample(missing, 2)
            cur_edges.remove(e1)
            cur_edges.remove(e2)
            cur_edges.append(e3)
            cur_edges.append(e4)
            
        return cur_edges

    best_edges = []
    best_score = -1
    
    # Run multiple restarts
    num_restarts = 1000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        cur_edges = random_initial_state()
        cur_score = len(cur_edges) - 100 * count_violations(cur_edges)
        
        if len(cur_edges) > len(best_edges) and is_valid(cur_edges):
            best_edges = cur_edges[:]
            best_score = len(best_edges)
            
        T = 10.0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            next_edges = neighbor_state(cur_edges)
            next_score = len(next_edges) - 100 * count_violations(next_edges)
            
            delta = next_score - cur_score
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                cur_edges = next_edges
                cur_score = next_score
                
                # Update best if valid
                if is_valid(cur_edges):
                    if len(cur_edges) > len(best_edges):
                        best_edges = cur_edges[:]
                        best_score = len(best_edges)
            
            T *= cooling_rate
            if T < min_T:
                break
        
        # Local Search Refinement on best found so far?
        # Let's do a quick greedy improvement on the current best
        # Actually, the SA loop already does this.

    # Final Greedy Polish on the best valid edge set found
    # Try to add any remaining edge that keeps it valid
    best_set = set(best_edges)
    improved = True
    while improved:
        improved = False
        for e in all_edges:
            if e in best_set:
                continue
            new_edges = best_edges + [e]
            if is_valid(new_edges):
                best_edges = new_edges
                best_set.add(e)
                improved = True
                if time.time() > deadline:
                    return [list(x) for x in best_edges]

    # Ensure types are correct
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
