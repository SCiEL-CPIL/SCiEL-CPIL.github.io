"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_N = 256
_KILL = 90.0
_MARGIN = 8.0

def _build_conflict_graph():
    """
    Build the conflict graph where an edge exists between a and b if there exists c
    such that {a, b, c} is a 3-AP.
    We represent the graph using bitmasks for efficiency.
    """
    # adj[i] is a bitmask where bit j is 1 if i and j form a 3-AP with some k
    adj = [0] * (_N + 1)
    
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            # Check if a, b, c is a 3-AP for some c in [1, N]
            # The three possibilities for the third element c given a, b:
            # 1. a, b, c => c = 2b - a
            # 2. a, c, b => c = (a+b)/2 (if a+b is even)
            # 3. c, a, b => c = 2a - b
            
            c1 = 2 * b - a
            if 1 <= c1 <= _N and c1 != a and c1 != b:
                # a and b are in a 3-AP with c1
                adj[a] |= 1 << b
                adj[b] |= 1 << a
                # Also, c1 conflicts with a and b
                adj[c1] |= 1 << a
                adj[c1] |= 1 << b
                
            s = a + b
            if s % 2 == 0:
                c2 = s // 2
                if 1 <= c2 <= _N and c2 != a and c2 != b:
                    adj[a] |= 1 << b
                    adj[b] |= 1 << a
                    adj[c2] |= 1 << a
                    adj[c2] |= 1 << b
                    
            c3 = 2 * a - b
            if 1 <= c3 <= _N and c3 != a and c3 != b:
                adj[a] |= 1 << b
                adj[b] |= 1 << a
                adj[c3] |= 1 << a
                adj[c3] |= 1 << b
                
    return adj

def _mwis_branch_and_bound(adj, weights, deadline):
    """
    Solves Maximum Weight Independent Set using branch and bound.
    weights: list of weights for nodes 1..N
    adj: list of bitmasks for adjacency
    """
    # We want to select a subset of nodes with no edges between them.
    # This is the Independent Set problem.
    
    best_weight = 0
    best_set = set()
    
    # Precompute weights for quick access
    w = [0] + weights[1:]
    
    # Sort nodes by weight descending to improve bound quality early? 
    # Or just use standard ordering. Standard B&B for MWIS often benefits from 
    # ordering by weight or degree. Let's stick to index order but use a good bound.
    
    # Upper bound function: Greedy coloring of the induced subgraph on candidates.
    # The number of colors needed is an upper bound on the size of the independent set.
    # For weighted case, we can use a weighted version or just sum of weights of 
    # nodes in the same color class? No, standard bound is:
    # Sort candidates by weight. Assign colors greedily.
    # The bound is the sum of the maximum weight in each color class? 
    # Actually, a simpler bound: 
    # If we color the graph, any independent set can contain at most one node from each color.
    # So the max weight independent set weight <= sum over colors of (max weight in that color).
    
    def color_bound(candidates):
        """
        candidates: bitmask of available nodes.
        Returns an upper bound on the weight of an independent set within candidates.
        """
        if candidates == 0:
            return 0
        
        # We need to color the subgraph induced by candidates.
        # Greedy coloring:
        # 1. Order nodes (e.g., by weight descending or index).
        # 2. Assign smallest available color.
        
        # To make the bound tight, we should order nodes by weight descending?
        # If we put high weight nodes in different colors, the sum of max weights per color 
        # will be higher, which is a looser bound (worse for pruning).
        # We want a TIGHT bound (low value).
        # So we should try to pack high weight nodes into the SAME color if possible?
        # No, coloring assigns colors such that adjacent nodes have DIFFERENT colors.
        # We want to minimize the sum of max(weights) per color.
        # This is achieved by assigning high weight nodes to colors that already have 
        # high weight nodes? No, that's not how greedy works easily.
        
        # Standard approach for weighted coloring bound:
        # Sort vertices by weight descending.
        # Assign each vertex to the first color class where it doesn't conflict.
        # This tends to put high weight vertices in early colors.
        # The bound is sum of max weight in each color class.
        
        # Let's extract vertices from bitmask
        verts = []
        c = candidates
        while c:
            # Get lowest set bit
            lsb = c & -c
            v = lsb.bit_length() - 1
            verts.append(v)
            c ^= lsb
        
        # Sort by weight descending
        verts.sort(key=lambda x: w[x], reverse=True)
        
        # Color classes: list of bitmasks or lists of nodes
        color_classes = [] # list of lists of nodes
        color_conflicts = [] # list of bitmasks of nodes in that color? No, we need to check adjacency.
        
        # To check if v can join color k, v must not be adjacent to any node in color k.
        # We can maintain a bitmask of nodes in each color class.
        class_masks = []
        
        for v in verts:
            placed = False
            for k in range(len(class_masks)):
                # Check if v is adjacent to any node in class_masks[k]
                # adj[v] & class_masks[k] == 0
                if (adj[v] & class_masks[k]) == 0:
                    class_masks[k] |= (1 << v)
                    placed = True
                    break
            if not placed:
                class_masks.append(1 << v)
                
        # Bound is sum of max weight in each class
        bound = 0
        for mask in class_masks:
            # Find max weight in mask
            max_w = 0
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if w[v] > max_w:
                    max_w = w[v]
                m ^= lsb
            bound += max_w
            
        return bound

    def solve(candidates, current_weight, current_set):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
        
        if candidates == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
        
        # Prune
        bound = color_bound(candidates)
        if current_weight + bound <= best_weight:
            return
            
        # Select a pivot vertex
        # Heuristic: pick vertex with max weight or max degree in candidates?
        # Let's pick the vertex with max weight in candidates to branch on.
        # Or pick one that appears in many conflicts.
        
        # Find a vertex to branch on
        # We can iterate through candidates.
        # Let's pick the first one (lowest index) or the one with highest weight.
        # Highest weight is usually better for weighted problems.
        
        c = candidates
        pivot = -1
        max_w = -1
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            if w[v] > max_w:
                max_w = w[v]
                pivot = v
            c ^= lsb
            
        if pivot == -1:
            return
            
        pivot_bit = 1 << pivot
        
        # Branch 1: Include pivot
        new_candidates = candidates & ~pivot_bit & ~adj[pivot]
        new_set = current_set | {pivot}
        solve(new_candidates, current_weight + w[pivot], new_set)
        
        # Branch 2: Exclude pivot
        new_candidates = candidates & ~pivot_bit
        solve(new_candidates, current_weight, current_set)
        
    # Initial candidates: all nodes 1..N
    # Bitmask: bits 1 to N are set.
    # Note: Python ints are arbitrary precision, so 1<<256 is fine.
    initial_candidates = 0
    for i in range(1, _N + 1):
        initial_candidates |= (1 << i)
        
    solve(initial_candidates, 0, set())
    
    return best_weight, best_set

def _greedy_init(adj, weights, rng, deadline):
    """
    Greedy construction: pick highest weight node, remove it and its neighbors.
    """
    S = set()
    available = set(range(1, _N + 1))
    
    while available and time.time() < deadline:
        # Pick node with max weight
        max_node = max(available, key=lambda x: weights[x])
        S.add(max_node)
        
        # Remove max_node and its neighbors from available
        neighbors = set()
        mask = adj[max_node]
        # Extract neighbors from mask
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if v in available:
                neighbors.add(v)
            m ^= lsb
            
        available.discard(max_node)
        available.difference_update(neighbors)
        
    return S

def _local_search(S, adj, weights, deadline):
    """
    Simple local search: try to add nodes, or swap.
    """
    S = set(S)
    current_weight = sum(weights[x] for x in S)
    best_S = set(S)
    best_w = current_weight
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a node
        best_gain = 0
        best_add = None
        for x in range(1, _N + 1):
            if x in S:
                continue
            # Check if x can be added (no neighbors in S)
            # x is compatible with S if (adj[x] & mask_of_S) == 0
            # Let's compute mask of S
            mask_S = 0
            for y in S:
                mask_S |= (1 << y)
            
            if (adj[x] & mask_S) == 0:
                gain = weights[x]
                if gain > best_gain:
                    best_gain = gain
                    best_add = x
                    
        if best_add is not None:
            S.add(best_add)
            current_weight += weights[best_add]
            if current_weight > best_w:
                best_w = current_weight
                best_S = set(S)
            improved = True
            continue # Restart loop with new S
            
        # Try swapping: remove x, add y
        # This is expensive. Skip for now or do simple version.
        # If no add possible, try swap.
        best_swap_gain = 0
        best_swap_out = None
        best_swap_in = None
        
        # Compute mask of S once
        mask_S = 0
        for y in S:
            mask_S |= (1 << y)
            
        for x_out in S:
            # If we remove x_out, mask becomes mask_S without x_out
            mask_without_x = mask_S & ~(1 << x_out)
            
            # Find best y to add that is not in S and not neighbor of any in S\{x_out}
            # And y != x_out (trivially true if y not in S)
            # Actually y can be x_out? No, we are swapping.
            # y must not be in S.
            # y must not be adjacent to any node in S\{x_out}.
            
            # We can iterate y
            for y_in in range(1, _N + 1):
                if y_in in S:
                    continue
                # Check compatibility with S\{x_out}
                # adj[y_in] & mask_without_x == 0
                if (adj[y_in] & mask_without_x) == 0:
                    gain = weights[y_in] - weights[x_out]
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_out = x_out
                        best_swap_in = y_in
                        
        if best_swap_gain > 0:
            S.discard(best_swap_out)
            S.add(best_swap_in)
            current_weight += best_swap_gain
            if current_weight > best_w:
                best_w = current_weight
                best_S = set(S)
            improved = True
            
    return best_S

def construct():
    start = time.time()
    deadline = start + _KILL - _MARGIN
    
    # 1. Build conflict graph
    adj = _build_conflict_graph()
    
    # 2. Weights: reciprocal of index
    weights = [0.0] + [1.0 / i for i in range(1, _N + 1)]
    
    best_S = None
    best_w = -1.0
    
    # 3. Run MWIS Branch and Bound
    # Give it a significant chunk of time, e.g., 60s
    bnb_deadline = min(start + 60.0, deadline)
    try:
        bnb_w, bnb_S = _mwis_branch_and_bound(adj, weights, bnb_deadline)
        if bnb_S and len(bnb_S) > 0:
            # Verify
            if bnb_w > best_w:
                best_w = bnb_w
                best_S = bnb_S
    except Exception:
        pass
        
    # 4. Greedy + Local Search as fallback/improvement
    if time.time() < deadline - 5.0:
        rng = random.Random(42)
        S_greedy = _greedy_init(adj, weights, rng, deadline - 5.0)
        S_opt = _local_search(S_greedy, adj, weights, deadline - 1.0)
        w_opt = sum(weights[x] for x in S_opt)
        if w_opt > best_w:
            best_w = w_opt
            best_S = S_opt
            
    if best_S is None or len(best_S) == 0:
        best_S = {1}
        
    res = sorted(best_S)
    
    # Defensive verification
    Sset = set(res)
    ok = True
    for i, a in enumerate(res):
        for b in res[i+1:]:
            c = 2 * b - a
            if c in Sset and c > b:
                ok = False
            d = 2 * a - b
            if d in Sset and d < a:
                ok = False
            s = a + b
            if s % 2 == 0:
                m = s // 2
                if m in Sset and m != a and m != b:
                    ok = False
    if not ok:
        # Fallback to simple valid set
        S = _greedy_init(adj, weights, random.Random(1), deadline)
        res = sorted(S)
        
    return res
# EVOLVE-BLOCK-END
