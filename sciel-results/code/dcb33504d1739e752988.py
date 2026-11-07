import time
import random
from itertools import combinations

def construct():
    """Find clique partitions of three split graphs, minimizing total clique count."""
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve_split(n, k, p, seed, deadline):
        """
        Solve a single split graph instance.
        C = {0..k-1} is the clique part, I = {k..n-1} is the independent set.
        Cross edges: (u, v) where u in C, v in I, with probability p.
        
        Strategy:
        - Every edge must be in exactly one clique.
        - Edges within C: need to be covered.
        - Cross edges (u,v) with u in C, v in I: need to be covered.
        - No edges within I.
        
        Key observations:
        1. A clique containing vertices from both C and I can cover cross edges.
        2. A clique containing only C vertices can cover C-C edges.
        3. Since I is independent, no clique can contain two I vertices.
        
        Approach:
        - For each I vertex v, let S_v = {u in C : (u,v) is an edge}.
        - We need to cover all C-C edges and all cross edges.
        - A clique {a, b, v} where a,b in C and v in I covers edges (a,b), (a,v), (b,v)
          if all three exist.
        - More generally, a clique {c1, c2, ..., cm, v} covers all C-C edges among 
          {c1,...,cm} and all cross edges from v to {c1,...,cm}.
        
        Greedy/heuristic approach:
        - Cover C-C edges using cliques that may include I vertices.
        - For each I vertex v, we can form cliques involving v to cover some C-C edges
          and the cross edges from v.
        
        Let's think differently:
        - Each clique either:
          (a) Contains exactly one I vertex v and some subset S of C. It covers:
              - All edges within S (C-C edges)
              - All cross edges (s, v) for s in S
          (b) Contains only C vertices. It covers C-C edges.
        
        To minimize number of cliques:
        - We want to cover as many edges as possible per clique.
        - A clique with one I vertex v and m C vertices covers m*(m-1)/2 C-C edges 
          and m cross edges.
        
        Simple approach:
        1. Identify for each I vertex v, its neighbor set S_v in C.
        2. We need to cover all C-C edges. The C part has k vertices forming a complete graph.
           Total C-C edges: k*(k-1)/2.
        3. Cross edges: sum of |S_v| over all v in I.
        
        Heuristic:
        - For each I vertex v, if S_v is non-empty, we can use v to help cover some C-C edges.
        - Form cliques that include v and pairs of C vertices from S_v.
        
        Actually, let's think about this more carefully.
        
        A clique can contain at most one I vertex. So cliques are of the form:
        - {v} ∪ S where v in I, S ⊆ S_v (since all vertices in S must be adjacent to v)
          But wait, for {v} ∪ S to be a clique, we need all pairs in S to be adjacent (they are, 
          since C is a clique) AND all s in S to be adjacent to v (which means S ⊆ S_v).
        - T where T ⊆ C
        
        So valid cliques are:
        1. T ⊆ C (any subset of C)
        2. {v} ∪ S where v in I and S ⊆ S_v
        
        We need to cover all edges exactly once.
        
        This is a set cover / exact cover problem, which is NP-hard in general, but for 
        split graphs with the structure given, we can use heuristics.
        
        Greedy approach:
        - Start with all edges uncovered.
        - Repeatedly find the clique that covers the most uncovered edges, add it, 
          mark those edges as covered.
        - Continue until all edges are covered.
        
        But we need EXACTLY one covering per edge, so this is an exact cover problem.
        
        Let me think of a simpler structure:
        
        Observation: In a split graph, the minimum clique partition has a specific structure.
        
        Let's try a constructive approach:
        - For the C part, we need to cover k*(k-1)/2 edges.
        - For cross edges, each I vertex v has |S_v| edges to cover.
        
        Key insight: If we use a clique {v, u1, u2, ..., um} where v in I and 
        {u1,...,um} ⊆ S_v, it covers:
        - m*(m-1)/2 C-C edges (among u1,...,um)
        - m cross edges (from v to each ui)
        
        So larger cliques are more efficient.
        
        Greedy exact cover heuristic:
        1. Generate candidate cliques.
        2. Use a greedy algorithm that picks the clique covering the most uncovered edges,
           ensuring no edge is covered twice.
        
        Given the time constraint, let's use a simpler approach:
        
        Approach A: Pairwise coverage
        - For each C-C edge (u,v), cover it with a clique {u,v} (2-clique).
        - For each cross edge (u,v), cover it with a clique {u,v} (2-clique).
        - Total cliques: k*(k-1)/2 + sum(|S_v| for v in I)
        - This is simple but not optimal.
        
        Approach B: Use I vertices to cover C-C edges
        - For each I vertex v with |S_v| >= 2, we can form a clique {v} ∪ S_v 
          (if S_v is the entire neighbor set). This covers all C-C edges within S_v 
          and all cross edges from v.
        - Then for remaining C-C edges not covered, use 2-cliques.
        - For cross edges from v not covered (if we didn't use the full S_v), use 2-cliques.
        
        Let's implement a greedy approach:
        1. Sort I vertices by |S_v| in descending order.
        2. For each I vertex v, consider using cliques involving v to cover as many edges as possible.
        3. Track which edges are covered.
        
        Actually, let me think about this more carefully with a cleaner formulation.
        
        The problem is: partition the edge set into cliques, minimizing the number of cliques.
        Each clique has size >= 2.
        
        For split graphs, I'll use the following heuristic:
        1. For each I vertex v, let S_v be its neighbors in C.
        2. We'll process I vertices in order of decreasing |S_v|.
        3. For each v, we try to form one or more cliques involving v that cover 
           a good fraction of the remaining uncovered edges.
        
        Let's just go with a simple but effective approach:
        - Use the fact that within C, we can cover edges efficiently.
        - For each I vertex v, if |S_v| >= 2, form a clique {v} ∪ S_v to cover 
          all edges incident to v and all C-C edges within S_v.
        - For any remaining C-C edges, cover them with 2-cliques.
        - For any remaining cross edges (from v to u where u in S_v but u is not in 
          the clique formed with v), cover with 2-cliques.
        
        Wait, if we form {v} ∪ S_v, it covers ALL cross edges from v and ALL C-C edges 
        within S_v. That's good.
        
        But different I vertices share C vertices, so C-C edges might be covered multiple times 
        if we're not careful. We need to track this.
        
        Let me implement a proper greedy algorithm:
        
        1. Compute all edges and their coverage status.
        2. For each I vertex v (sorted by |S_v| desc), if |S_v| >= 2:
           - Consider the clique {v} ∪ S_v.
           - Count how many currently uncovered edges it would cover.
           - If it covers at least 1 uncovered edge, add it and mark those edges as covered.
        3. For remaining uncovered C-C edges, add 2-cliques.
        4. For remaining uncovered cross edges, add 2-cliques.
        
        This should be reasonably efficient and give a good solution.
        """
        # Build adjacency for C and cross edges
        rng = random.Random(seed)
        
        # C-C edges (all pairs within C)
        cc_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        
        # Cross edges
        cross_edges = []
        S = [set() for _ in range(n - k)]  # S[i] = neighbors of I vertex (k+i) in C
        for v_idx in range(n - k):
            v = k + v_idx
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
                    S[v_idx].add(u)
        
        all_edges = set(cc_edges) | set(cross_edges)
        uncovered = set(all_edges)
        
        cliques = []
        
        # Process I vertices in order of decreasing |S_v|
        i_order = sorted(range(n - k), key=lambda idx: len(S[idx]), reverse=True)
        
        for idx in i_order:
            if time.time() > deadline:
                break
            v = k + idx
            Sv = S[idx]
            if len(Sv) < 2:
                continue
            
            # Consider clique {v} ∪ Sv
            clique = list(Sv) + [v]
            clique.sort()
            
            # Find which edges in this clique are currently uncovered
            clique_edges = set()
            for a, b in combinations(clique, 2):
                if a < b:
                    clique_edges.add((a, b))
                else:
                    clique_edges.add((b, a))
            
            new_covered = clique_edges & uncovered
            if len(new_covered) > 0:
                cliques.append(clique)
                uncovered -= new_covered
        
        # Cover remaining C-C edges with 2-cliques
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) in uncovered:
                    cliques.append([u, v])
                    uncovered.discard((u, v))
        
        # Cover remaining cross edges with 2-cliques
        for u in range(k):
            for v in range(k, n):
                if (u, v) in uncovered:
                    cliques.append([u, v])
                    uncovered.discard((u, v))
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > DEADLINE:
            # Fallback: return 2-cliques for all edges
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
        else:
            cliques = solve_split(n, k, p, seed, DEADLINE)
            results.append(cliques)
    
    # Ensure all elements are Python ints and lists
    final_result = []
    for instance_cliques in results:
        cleaned = []
        for clique in instance_cliques:
            cleaned.append([int(x) for x in clique])
        final_result.append(cleaned)
    
    return final_result
