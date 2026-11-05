"""Rewritten program: maximum independent set in conflict graph for 4-uniform sunflower-free families."""
import random
from itertools import combinations

def construct():
    # Universe is [1, 16], k=4
    n = 16
    k = 4
    
    # Generate all 4-subsets
    all_sets = list(combinations(range(1, n + 1), k))
    set_to_idx = {s: i for i, s in enumerate(all_sets)}
    m = len(all_sets)  # 1820
    
    # Build conflict graph
    # Two sets A, B conflict if there exists a third set C (distinct from A, B)
    # such that |A∩B| = |A∩C| = |B∩C|
    # 
    # Key insight: For fixed A and B, the condition is that there exists C ≠ A, B
    # with |A∩C| = |B∩C| = |A∩B|.
    #
    # Let's precompute for each pair (A, B) whether they conflict.
    # A and B conflict if ∃ C ≠ A, B: |A∩C| = |B∩C| = |A∩B|.
    
    # Precompute intersection sizes for all pairs
    # inter[i][j] = |set_i ∩ set_j|
    inter = [[0] * m for _ in range(m)]
    for i in range(m):
        si = set(all_sets[i])
        for j in range(i + 1, m):
            sj = set(all_sets[j])
            inter[i][j] = len(si & sj)
            inter[j][i] = inter[i][j]
    
    # For each pair (i, j), check if there exists k ≠ i, j with inter[i][k] == inter[j][k] == inter[i][j]
    # To make this efficient, for each pair (i, j), we need to count how many k satisfy the condition.
    
    # Let's think differently. For a fixed target intersection size t = inter[i][j],
    # we need to find if there exists k ≠ i, j such that inter[i][k] == t and inter[j][k] == t.
    
    # Precompute for each set i and each possible intersection size t (0,1,2,3,4),
    # the list of sets j such that inter[i][j] == t.
    
    # Actually, let's just build the conflict graph directly but efficiently.
    # For each pair (i, j), check if conflict exists.
    
    # We'll use a smarter approach:
    # For each triple of intersection sizes... actually let's just do it with bitsets or lists.
    
    # For each i, for each t in 0..4, store the list of j where inter[i][j] == t
    # Then for pair (i, j) with t = inter[i][j], check if there's a k in (list_i[t] ∩ list_j[t]) \ {i, j}
    
    adj = [set() for _ in range(m)]
    
    # Precompute buckets
    buckets = [[[] for _ in range(5)] for _ in range(m)]  # buckets[i][t] = list of j with inter[i][j] == t
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            t = inter[i][j]
            buckets[i][t].append(j)
    
    # For each pair (i, j), i < j, check conflict
    for i in range(m):
        for j in range(i + 1, m):
            t = inter[i][j]
            # Check if there exists k ≠ i, j with inter[i][k] == t and inter[j][k] == t
            # Use the smaller list for iteration
            list_i = set(buckets[i][t])
            list_j = set(buckets[j][t])
            common = list_i & list_j
            # Remove i and j from common
            common.discard(i)
            common.discard(j)
            if common:
                adj[i].add(j)
                adj[j].add(i)
    
    # Now solve maximum independent set using branch and bound
    # Greedy vertex ordering by degree centrality (descending)
    # Use a heuristic: pick vertex with highest degree first in ordering
    
    # Order vertices by degree (descending)
    order = sorted(range(m), key=lambda x: -len(adj[x]))
    pos_in_order = [0] * m
    for idx, v in enumerate(order):
        pos_in_order[v] = idx
    
    # Reindex vertices according to order
    # new_idx[v] = position of v in order
    new_to_old = order
    old_to_new = [0] * m
    for i, v in enumerate(order):
        old_to_new[v] = i
    
    # Build new adjacency in reordered indices
    adj_new = [set() for _ in range(m)]
    for i in range(m):
        for j in adj[i]:
            adj_new[old_to_new[i]].add(old_to_new[j])
    
    # Branch and bound for maximum independent set
    # We use a simple but effective approach:
    # - Process vertices in order (highest degree first)
    # - At each step, either include or exclude the vertex
    # - Use upper bound: current_size + number of remaining vertices that can still be added
    
    best_size = 0
    best_set = set()
    
    def max_independent_set(vertices, current_set, current_size):
        nonlocal best_size, best_set
        
        if not vertices:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound: current_size + len(vertices)
        if current_size + len(vertices) <= best_size:
            return
        
        # Pick the first vertex in our ordering
        v = min(vertices, key=lambda x: pos_in_order[x])
        # Actually, since we reordered, vertices are already in order. Let's just pick the first.
        # But vertices is a set, so let's pick the one with smallest pos_in_order
        # Since we reordered, the smallest index in the new ordering should be picked first
        # Let's just pick min(vertices) which corresponds to highest degree in original
        
        # Option 1: Include v
        # Remove v and its neighbors from vertices
        neighbors = adj_new[v]
        remaining_after_include = vertices - {v} - neighbors
        current_set.add(v)
        max_independent_set(remaining_after_include, current_set, current_size + 1)
        current_set.remove(v)
        
        # Option 2: Exclude v
        remaining_after_exclude = vertices - {v}
        max_independent_set(remaining_after_exclude, current_set, current_size)
    
    # This naive branch and bound might be too slow. Let's use a better heuristic.
    # Use a greedy + local search approach instead.
    
    # Actually, let's try a different approach: greedy construction with conflict checking,
    # then try to improve via local search.
    
    # Greedy: add sets in order of... let's try random orderings and take the best.
    
    # But first, let's try a simple greedy: add sets that have fewest conflicts with already chosen.
    
    # Actually, the problem is NP-hard, but with m=1820, we might be able to do a decent
    # greedy + local search.
    
    # Let's do: greedy add, then try to add more, then local search by swapping.
    
    def greedy_construct(ordering):
        chosen = set()
        for v in ordering:
            # Check if v conflicts with any in chosen
            if not any(w in chosen for w in adj_new[v]):
                chosen.add(v)
        return chosen
    
    def local_search(chosen, max_iter=10000):
        chosen = set(chosen)
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            # Try to add a vertex not in chosen
            for v in range(m):
                if v in chosen:
                    continue
                if not any(w in chosen for w in adj_new[v]):
                    chosen.add(v)
                    improved = True
                    break
            if improved:
                continue
            # Try swaps: remove one, add one or more
            for v in list(chosen):
                for u in range(m):
                    if u in chosen:
                        continue
                    # Remove v, add u: check if u conflicts with chosen - {v}
                    new_chosen = chosen - {v}
                    if not any(w in new_chosen for w in adj_new[u]):
                        chosen = new_chosen | {u}
                        improved = True
                        break
                if improved:
                    break
        return chosen
    
    # Try multiple random orderings
    best_chosen = set()
    best_size_final = 0
    
    for trial in range(50):
        ordering = list(range(m))
        random.shuffle(ordering)
        chosen = greedy_construct(ordering)
        chosen = local_search(chosen, max_iter=2000)
        if len(chosen) > best_size_final:
            best_size_final = len(chosen)
            best_chosen = set(chosen)
    
    # Convert back to original indices and then to sets
    result = []
    for v in best_chosen:
        result.append(list(all_sets[v]))
    
    result.sort()
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Family size: {len(res)}")
