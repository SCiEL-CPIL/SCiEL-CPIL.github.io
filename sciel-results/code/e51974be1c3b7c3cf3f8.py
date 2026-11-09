"""Direct bootstrap for R(5,5): extend Paley(37) with a time-boxed heuristic search."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE = time.time() + 270  # safety margin: 330s hard, aim to finish by 270s


def _build_paley37():
    """Return adjacency bitset list for Paley graph on 37 vertices (vertices 0..36)."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _has_k5(adj, candidates):
    """Check if any 5-tuple of candidates plus new vertex contains K5.
    Candidates are already mutually connected (K4 in G).
    We need: new vertex connected to all 5, and all 5 pairs connected among themselves.
    Returns True if any K5 found."""
    n = len(candidates)
    if n < 5:
        return False
    # candidates list contains indices
    for combo in combinations(candidates, 5):
        # Check all pairs in combo are connected
        ok = True
        for i_idx in range(5):
            a = combo[i_idx]
            for j_idx in range(i_idx + 1, 5):
                b = combo[j_idx]
                if not (adj[a] & (1 << b)):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            # K5 would be combo itself (the 5 vertices) — but wait, if all pairs among combo are
            # connected, that's a K5 in G. But G is (5,5)-free, so this should never happen.
            # Actually, if G is valid (5,5)-free, no K5 exists in G. So this check is just
            # defensive. The real constraint is: new vertex + 4 mutually connected candidates = K5.
            pass
    return False


def _has_k5_with_new(adj, new_idx, combo4):
    """Check if new vertex connected to all 4 in combo4, and all pairs in combo4 connected.
    This would form K5 = {new_idx} + combo4."""
    for a in combo4:
        if not (adj[a] & (1 << new_idx)):
            return False
    for i in range(4):
        a = combo4[i]
        for j in range(i + 1, 4):
            b = combo4[j]
            if not (adj[a] & (1 << b)):
                return False
    return True


def _has_ind5_with_new(adj, new_idx, combo4):
    """Check if new vertex NOT connected to all 4 in combo4, and all pairs in combo4 NOT connected.
    This would form independent set = {new_idx} + combo4."""
    for a in combo4:
        if (adj[a] & (1 << new_idx)):
            return False
    for i in range(4):
        a = combo4[i]
        for j in range(i + 1, 4):
            b = combo4[j]
            if (adj[a] & (1 << b)):
                return False
    return True


def _find_k4_sets(adj, n):
    """Find all K4 cliques in graph with n vertices. Return list of 4-tuples."""
    k4_list = []
    for i in range(n):
        neigh_i = [j for j in range(n) if j > i and (adj[i] & (1 << j))]
        for j in range(n):
            if j <= i or not (adj[i] & (1 << j)):
                continue
            neigh_ij = [k for k in range(n) if k > j and (adj[i] & (1 << k)) and (adj[j] & (1 << k))]
            for k1 in range(len(neigh_ij)):
                a = neigh_ij[k1]
                for k2 in range(k1 + 1, len(neigh_ij)):
                    b = neigh_ij[k2]
                    if adj[a] & (1 << b):
                        k4_list.append((i, j, a, b))
    return k4_list


def _find_ind4_sets(adj, n):
    """Find all independent 4-sets in graph with n vertices. Return list of 4-tuples."""
    ind4_list = []
    for i in range(n):
        non_neigh_i = [j for j in range(n) if j > i and not (adj[i] & (1 << j))]
        for j in non_neigh_i:
            non_neigh_ij = [k for k in range(n) if k > j and not (adj[i] & (1 << k)) and not (adj[j] & (1 << k))]
            for k1 in range(len(non_neigh_ij)):
                a = non_neigh_ij[k1]
                for k2 in range(k1 + 1, len(non_neigh_ij)):
                    b = non_neigh_ij[k2]
                    if not (adj[a] & (1 << b)):
                        ind4_list.append((i, j, a, b))
    return ind4_list


def _try_add_vertex(adj, n, k4_list, ind4_list, order):
    """Try to add vertex n (new index = n) by choosing edges to 0..n-1.
    Uses constraint-driven greedy assignment with local repair.
    Returns new_adj or None if failed."""
    new_n = n + 1
    
    # Build constraint maps: for each vertex, which K4s and ind4s it belongs to
    k4_by_vertex = {v: [] for v in range(n)}
    for idx, (a, b, c, d) in enumerate(k4_list):
        for v in (a, b, c, d):
            k4_by_vertex[v].append(idx)
    
    ind4_by_vertex = {v: [] for v in range(n)}
    for idx, (a, b, c, d) in enumerate(ind4_list):
        for v in (a, b, c, d):
            ind4_by_vertex[v].append(idx)
    
    def check_validity(assign):
        """Check if assignment satisfies all constraints."""
        for (a, b, c, d) in k4_list:
            if assign[a] and assign[b] and assign[c] and assign[d]:
                return False
        for (a, b, c, d) in ind4_list:
            if not assign[a] and not assign[b] and not assign[c] and not assign[d]:
                return False
        return True
    
    def count_violations(assign):
        """Count number of violated constraints."""
        count = 0
        for (a, b, c, d) in k4_list:
            if assign[a] and assign[b] and assign[c] and assign[d]:
                count += 1
        for (a, b, c, d) in ind4_list:
            if not assign[a] and not assign[b] and not assign[c] and not assign[d]:
                count += 1
        return count
    
    def local_repair(assign, max_iters=50):
        """Try to fix violations by flipping individual bits."""
        assign = assign[:]
        for _ in range(max_iters):
            if check_validity(assign):
                return assign
            # Find a violated constraint and flip a bit to help
            # Try flipping each vertex's assignment to reduce violations
            best_flip = None
            best_score = count_violations(assign)
            for v in range(n):
                old_val = assign[v]
                assign[v] = 1 - old_val
                score = count_violations(assign)
                if score < best_score:
                    best_score = score
                    best_flip = v
                assign[v] = old_val
            if best_flip is None:
                break
            assign[best_flip] = 1 - assign[best_flip]
        if check_validity(assign):
            return assign
        return None
    
    # Strategy 1: Greedy constraint-driven assignment
    for attempt in range(50):
        assign = [None] * n
        # Process vertices in order, choosing assignment that minimizes future conflicts
        for pos in range(n):
            v = order[pos]
            # For each possible value (0 or 1), estimate how many constraints would be "at risk"
            scores = []
            for val in (0, 1):
                # Count K4s containing v where all other 3 are already assigned to 1
                k4_risk = 0
                for idx in k4_by_vertex[v]:
                    a, b, c, d = k4_list[idx]
                    others = [x for x in (a, b, c, d) if x != v]
                    all_assigned_1 = all(assign[x] == 1 for x in others if assign[x] is not None)
                    if all_assigned_1 and val == 1:
                        k4_risk += 1
                
                # Count ind4s containing v where all other 3 are already assigned to 0
                ind4_risk = 0
                for idx in ind4_by_vertex[v]:
                    a, b, c, d = ind4_list[idx]
                    others = [x for x in (a, b, c, d) if x != v]
                    all_assigned_0 = all(assign[x] == 0 for x in others if assign[x] is not None)
                    if all_assigned_0 and val == 0:
                        ind4_risk += 1
                
                scores.append((k4_risk + ind4_risk, val))
            
            # Pick the value with lower risk; break ties randomly
            scores.sort()
            if len(scores) > 1 and scores[0][0] == scores[1][0]:
                chosen_val = random.choice([0, 1])
            else:
                chosen_val = scores[0][1]
            assign[v] = chosen_val
        
        if check_validity(assign):
            new_adj = adj[:] + [0]
            for v in range(n):
                if assign[v]:
                    new_adj[new_n] |= (1 << v)
                    new_adj[v] |= (1 << new_n)
            return new_adj
        
        # Try local repair
        repaired = local_repair(assign)
        if repaired is not None:
            new_adj = adj[:] + [0]
            for v in range(n):
                if repaired[v]:
                    new_adj[new_n] |= (1 << v)
                    new_adj[v] |= (1 << new_n)
            return new_adj
    
    # Strategy 2: Random assignments with repair
    for attempt in range(300):
        assign = [random.choice([0, 1]) for _ in range(n)]
        if check_validity(assign):
            new_adj = adj[:] + [0]
            for v in range(n):
                if assign[v]:
                    new_adj[new_n] |= (1 << v)
                    new_adj[v] |= (1 << new_n)
            return new_adj
        repaired = local_repair(assign, max_iters=30)
        if repaired is not None:
            new_adj = adj[:] + [0]
            for v in range(n):
                if repaired[v]:
                    new_adj[new_n] |= (1 << v)
                    new_adj[v] |= (1 << new_n)
            return new_adj
    
    return None


def construct():
    adj = _build_paley37()
    n = 37
    
    best_n = 37
    best_adj = adj[:]
    
    # Precompute K4 and independent 4-sets for the base graph
    k4_list = _find_k4_sets(adj, n)
    ind4_list = _find_ind4_sets(adj, n)
    
    # Try to extend the graph by adding vertices one at a time
    for target_n in range(38, 43):
        if time.time() > DEADLINE:
            break
        
        # Recompute k4 and ind4 for current graph
        k4_list = _find_k4_sets(adj, n)
        ind4_list = _find_ind4_sets(adj, n)
        
        # Try multiple random orderings for robustness
        success = False
        for trial in range(5):
            if time.time() > DEADLINE:
                break
            order = list(range(n))
            random.shuffle(order)
            
            new_adj = _try_add_vertex(adj, n, k4_list, ind4_list, order)
            
            if new_adj is not None:
                adj = new_adj
                n = target_n
                best_n = n
                best_adj = adj[:]
                print(f"Extended to n={n}")
                success = True
                break
        
        if not success:
            break
    
    # Build edge list from best_adj
    n_best = best_n
    edges = []
    for i in range(n_best):
        for j in range(i + 1, n_best):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
    
    return n_best, edges

# EVOLVE-BLOCK-END
