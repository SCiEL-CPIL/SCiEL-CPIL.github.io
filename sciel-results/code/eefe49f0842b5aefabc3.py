import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill
    
    n = 20
    total_pairs = n * (n - 1) // 2
    
    # 1. Construct the Conflict Graph
    # Nodes in the conflict graph represent potential edges (u, v) in the original graph.
    # Two nodes (u1, v1) and (u2, v2) are connected in the conflict graph if including
    # both edges in the original graph would create a K2,3.
    # A K2,3 is created if:
    # a) The edges share a vertex and the other two vertices are adjacent to both shared vertices? 
    #    Actually, the condition is: No two vertices have 3 common neighbors.
    #    Let the set of edges be E.
    #    For any pair of vertices {u, v} in the original graph, let N(u) be neighbors of u.
    #    We require |N(u) ∩ N(v)| <= 2 for all u != v.
    #    
    #    How does adding an edge (a,b) affect this?
    #    Adding (a,b) adds b to N(a) and a to N(b).
    #    This increases |N(a) ∩ N(w)| by 1 if b in N(w) (i.e., w in N(b)).
    #    This increases |N(b) ∩ N(w)| by 1 if a in N(w) (i.e., w in N(a)).
    #    This increases |N(a) ∩ N(b)| by 1 (the edge itself is not a neighbor, but wait).
    #    Common neighbors of a and b are vertices x such that x in N(a) and x in N(b).
    #    The edge (a,b) does not add a common neighbor to the pair (a,b) itself in terms of 
    #    "common neighbors" definition (neighbors are distinct from u and v).
    #    However, the constraint is on the *final* graph.
    #    
    #    Conflict Condition for two edges e1=(u1,v1) and e2=(u2,v2):
    #    If we include both, do they cause any pair of vertices to have >2 common neighbors?
    #    
    #    Let's analyze which pairs of edges are incompatible.
    #    Let e1 = {u1, v1} and e2 = {u2, v2}.
    #    
    #    Case 1: Disjoint edges {u1,v1} and {u2,v2} with u1,v1,u2,v2 distinct.
    #    If we include both, consider the pair of vertices {u1, u2}.
    #    Common neighbors of u1 and u2 in the graph containing e1, e2:
    #    - v1 is a neighbor of u1. Is v1 a neighbor of u2? Only if edge (v1,u2) is in E.
    #    - v2 is a neighbor of u2. Is v2 a neighbor of u1? Only if edge (v2,u1) is in E.
    #    - u1 is not a neighbor of u1.
    #    - u2 is not a neighbor of u2.
    #    
    #    This line of reasoning is complex for pairwise conflicts because the violation 
    #    might involve 3 edges or more. 
    #    
    #    HOWEVER, the constraint is local: |N(u) ∩ N(v)| <= 2.
    #    If we have a set of edges E, the condition holds iff for every u<v, 
    #    the number of w such that (u,w) in E and (v,w) in E is <= 2.
    #    
    #    This is NOT a simple pairwise conflict graph problem in the standard sense 
    #    where "conflict" means "cannot be in the same independent set". 
    #    An independent set in the conflict graph corresponds to a set of edges with 
    #    NO pairwise conflicts. But K2,3 violations are often global (3 edges forming the K2,3).
    #    
    #    Wait, let's re-read the K2,3 definition. 
    #    K2,3 is a subgraph. It consists of 2 vertices in one part (say A={u,v}) 
    #    and 3 in the other (B={w1,w2,w3}), with all 6 edges present.
    #    So, if the graph contains edges (u,w1), (u,w2), (u,w3), (v,w1), (v,w2), (v,w3), 
    #    it contains K2,3.
    #    This means if we pick edges, we must avoid picking a set of 6 edges that forms this pattern.
    #    This is a hypergraph independent set problem (3-uniform? No, the forbidden configuration has 6 edges).
    #    
    #    So, standard MCIS on a pairwise conflict graph is NOT sufficient because 
    #    a set of edges can be pairwise compatible but collectively form a K2,3.
    #    Example: 
    #    e1=(u,w1), e2=(v,w1), e3=(u,w2), e4=(v,w2), e5=(u,w3), e6=(v,w3).
    #    Do any two of these conflict in a pairwise sense?
    #    If we define conflict as "adding both causes immediate violation", 
    #    adding e1 and e2 doesn't violate. Adding e1..e5 doesn't violate. 
    #    Only when all 6 are present does it violate.
    #    
    #    Therefore, the problem is a Maximum Independent Set in a 3-uniform (or higher) hypergraph, 
    #    or equivalently, a 0-1 Integer Programming problem with higher-order constraints.
    #    
    #    Branch and Bound can still be used, but the bound must account for these higher-order constraints.
    #    A simple coloring bound on a pairwise graph would be invalid/loose.
    #
    #    Alternative Strategy: 
    #    Given n=20 is very small, we can use a more direct search.
    #    The current SA approach is stochastic. 
    #    Let's try a **Deterministic Greedy with Backtracking** or **Local Search with Stronger Moves**.
    #    
    #    Actually, let's look at the "Inspirations". 
    #    Inspiration 1 mentions "deterministic exhaustive search with symmetry breaking".
    #    Inspiration 2 mentions "Analysis of the Problem".
    #    
    #    Let's try a **Simulated Annealing** but with a much better initialization and 
    #    **Edge Swap** moves (remove one, add one) instead of just add/remove, 
    #    and a **Greedy Coloring** based bound is not applicable directly.
    #    
    #    Instead, let's use **Local Search** with **2-opt** style moves (swap edges) 
    #    and a **Tabu Search** to escape local optima. 
    #    The current program only does add/remove. Swapping allows maintaining the edge count 
    #    while exploring the space, which is crucial for maximizing cardinality under hard constraints.
    #
    #    Strategy: 
    #    1. Start with a dense random graph.
    #    2. Clean it by removing edges involved in violations (greedy).
    #    3. Local Search Loop:
    #       - Try to add a random missing edge. If valid, accept.
    #       - Try to swap a random existing edge with a random missing edge. 
    #         If the resulting graph is valid, accept (even if edge count stays same, 
    #         it might enable future additions).
    #       - Use a cooling schedule.
    #
    #    This is a different design family: **Tabu/SA with Swap Moves** vs **SA with Add/Remove**.
    #    Swap moves allow "moving" the bottleneck edges rather than just shrinking the graph.
    
    def get_common_count(adj, u, v):
        return (adj[u] & adj[v]).bit_count()
    
    def is_valid_add(adj, u, v):
        if get_common_count(adj, u, v) > 2:
            return False
        for w in range(n):
            if w == u or w == v:
                continue
            inc_uw = 1 if (adj[v] >> w) & 1 else 0
            if (adj[u] & adj[w]).bit_count() + inc_uw > 2:
                return False
            inc_vw = 1 if (adj[u] >> w) & 1 else 0
            if (adj[v] & adj[w]).bit_count() + inc_vw > 2:
                return False
        return True

    def edge_count(adj):
        return sum(a.bit_count() for a in adj) // 2

    def to_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def clean_graph(adj):
        """Remove edges to make graph valid."""
        changed = True
        while changed:
            changed = False
            violation = None
            for u in range(n):
                for v in range(u + 1, n):
                    if get_common_count(adj, u, v) > 2:
                        violation = (u, v)
                        break
                if violation:
                    break
            
            if not violation:
                break
            
            u, v = violation
            common = (adj[u] & adj[v])
            common_list = []
            w = 0
            c = common
            while c:
                if c & 1:
                    common_list.append(w)
                c >>= 1
                w += 1
            
            if not common_list:
                break
            
            # Remove an edge incident to one of the common neighbors
            # To maximize density, maybe remove the edge that is part of the most violations?
            # For now, random choice from common neighbors, then random side.
            w = random.choice(common_list)
            if random.random() < 0.5:
                adj[u] &= ~(1 << w)
                adj[w] &= ~(1 << u)
            else:
                adj[v] &= ~(1 << w)
                adj[w] &= ~(1 << v)
            changed = True
        
        return adj

    best_edges = []
    best_count = 0
    
    # Run multiple restarts
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        # Initialize
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.7: # Start denser
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        adj = clean_graph(adj)
        
        current_count = edge_count(adj)
        best_adj_restart = adj[:]
        best_count_restart = current_count
        
        T = 1.0
        cooling = 0.999
        min_T = 0.001
        steps = 0
        max_steps = 500000 # High step count, controlled by time
        
        # Precompute all possible edges for swap moves
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        while T > min_T and time.time() < deadline - 5:
            move_type = random.random()
            
            if move_type < 0.4:
                # Try Add
                u, v = random.choice(all_edges)
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        # Always accept valid add
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_count += 1
                        if current_count > best_count_restart:
                            best_count_restart = current_count
                            best_adj_restart = adj[:]
            elif move_type < 0.8:
                # Try Swap: Remove one, Add one
                # Pick a random existing edge to remove
                existing_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj[u] >> v) & 1:
                            existing_edges.append((u,v))
                
                if existing_edges:
                    ru, rv = random.choice(existing_edges)
                    # Pick a random missing edge to add
                    missing_edges = []
                    for u, v in all_edges:
                        if not ((adj[u] >> v) & 1):
                            missing_edges.append((u,v))
                    
                    if missing_edges:
                        au, av = random.choice(missing_edges)
                        
                        # Temporarily remove (ru, rv)
                        adj[ru] &= ~(1 << rv)
                        adj[rv] &= ~(1 << ru)
                        
                        # Check if adding (au, av) is valid in the new state
                        if is_valid_add(adj, au, av):
                            adj[au] |= (1 << av)
                            adj[av] |= (1 << au)
                            # Count remains same, but structure changed
                            # Accept with some probability to allow exploration? 
                            # Since count is same, we usually accept to escape local optima.
                            # In SA, delta = 0, so accept with prob 1.
                            # But we should track best count.
                        else:
                            # Revert removal
                            adj[ru] |= (1 << rv)
                            adj[rv] |= (1 << ru)
            else:
                # Try Remove (to escape bad states if stuck, though we don't track energy strictly)
                # Actually, since we always accept valid adds, we might get stuck if no valid adds exist.
                # Removing an edge opens up possibilities for swaps/adds.
                existing_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj[u] >> v) & 1:
                            existing_edges.append((u,v))
                if existing_edges:
                    ru, rv = random.choice(existing_edges)
                    # Accept removal with probability exp(-1/T) ? 
                    # Removing decreases count, so it's a "bad" move.
                    if random.random() < pow(2.71828, -1.0 / T):
                        adj[ru] &= ~(1 << rv)
                        adj[rv] &= ~(1 << ru)
                        current_count -= 1
            
            T *= cooling
            steps += 1
            
            # Periodic check
            if steps % 1000 == 0 and time.time() > deadline - 10:
                break
        
        if best_count_restart > best_count:
            best_count = best_count_restart
            best_edges = to_edge_list(best_adj_restart)
            
    # Final fallback
    if not best_edges:
        for i in range(n):
            best_edges.append([i, (i + 1) % n])
            
    return best_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    if valid:
        print("Valid")
    else:
        print("Invalid")
