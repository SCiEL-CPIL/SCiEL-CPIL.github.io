"""
Strategy: Maximum Independent Set via Branch and Bound (Tomita's algorithm) on a conflict graph.
Nodes are (modulus, residue) pairs. Edges represent conflicts (non-disjoint classes).
We find the largest set of non-conflicting nodes, constrained to at most one node per modulus.
"""
# EVOLVE-BLOCK-START
import math
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    MAX_MOD = 128

    # 1. Generate candidate nodes
    # Node ID: index in the list
    # Structure: list of (modulus, residue)
    candidates = []
    mod_to_indices = {}
    
    for m in range(2, MAX_MOD + 1):
        indices_for_m = []
        for r in range(m):
            idx = len(candidates)
            candidates.append((m, r))
            indices_for_m.append(idx)
        mod_to_indices[m] = indices_for_m

    n_nodes = len(candidates)
    
    # 2. Build Conflict Graph
    # Two nodes (m1, r1) and (m2, r2) conflict if their residue classes intersect.
    # Intersection exists iff (r1 - r2) % gcd(m1, m2) == 0.
    # We want an Independent Set, which is equivalent to a Clique in the complement graph.
    # However, the complement graph includes edges between nodes of the SAME modulus (which we cannot pick together).
    # It is easier to work with the Conflict Graph directly and find an Independent Set.
    # Tomita's algorithm is typically for Maximum Clique. 
    # Let's build the "Compatibility" graph (Complement of Conflict Graph).
    # Edge in Compat Graph: (u, v) if u and v are compatible (disjoint classes).
    # We want Max Clique in Compat Graph, BUT with the constraint that no two nodes have the same modulus.
    # Actually, if we just find Max Clique in Compat Graph, it might pick multiple nodes with the same modulus 
    # IF those nodes are compatible with each other. 
    # Are two nodes with the same modulus compatible?
    # m1 == m2 = m. gcd(m, m) = m.
    # Condition: (r1 - r2) % m == 0.
    # Since 0 <= r < m, r1 - r2 is in (-m, m).
    # (r1 - r2) % m == 0 implies r1 == r2.
    # So distinct nodes with the same modulus are ALWAYS in conflict (not compatible).
    # Therefore, any Clique in the Compat Graph automatically has distinct moduli!
    # So the problem reduces exactly to: Find Maximum Clique in the Compatibility Graph.

    # Build Adjacency List for Compatibility Graph
    # n_nodes is around sum_{m=2}^{128} m = (128*129)/2 - 1 = 8255 nodes.
    # This is a bit large for naive O(N^2) bitset construction in Python if not optimized, but N=8255 is manageable.
    # N^2 ~ 68 million operations. In Python, this might take a few seconds.
    # Let's optimize the construction.
    
    adj = [0] * n_nodes  # Bitsets represented as Python integers for fast bitwise ops
    
    # Precompute gcd table for moduli
    moduli_list = list(range(2, MAX_MOD + 1))
    gcd_cache = {}
    for i in range(len(moduli_list)):
        m1 = moduli_list[i]
        for j in range(i, len(moduli_list)):
            m2 = moduli_list[j]
            g = math.gcd(m1, m2)
            gcd_cache[(m1, m2)] = g
            if m1 != m2:
                gcd_cache[(m2, m1)] = g

    # Build adjacency bitsets
    # For each node i, check against j > i
    # If compatible, set bit j in adj[i] and bit i in adj[j]
    
    # To speed up, we can group by modulus.
    # Nodes of modulus m1 are compatible with nodes of modulus m2 if (r1 - r2) % gcd(m1, m2) != 0.
    
    # Let's iterate pairs of moduli to build edges, which might be faster than node-pair if we use vectorization or just tight loops.
    # Actually, pure Python loops for 8k nodes squared is 64M iterations. That's ~10-20 seconds in CPython.
    # We have 80s budget. This is feasible.
    
    # Optimization: Use NumPy? No, bitsets are easier with Python ints.
    # Let's try to make the inner loop as fast as possible.
    
    node_mod = [0] * n_nodes
    node_res = [0] * n_nodes
    for idx, (m, r) in enumerate(candidates):
        node_mod[idx] = m
        node_res[idx] = r

    # Build graph
    # We only need to check pairs (i, j) where i < j.
    # If node_mod[i] == node_mod[j], they are incompatible (unless same node, but i!=j).
    # If node_mod[i] != node_mod[j], check gcd condition.
    
    # To reduce overhead, pre-group indices by modulus
    # But we need to connect across moduli.
    
    # Let's just do the double loop but break early if time is tight.
    # 8255 nodes.
    
    # Alternative: Build the graph on the fly in the B&B algorithm? No, preprocessing is better.
    
    # Let's use a slightly optimized approach:
    # For each modulus m1, for each modulus m2 > m1:
    #   g = gcd(m1, m2)
    #   For each r1 in m1, r2 in m2:
    #      if (r1 - r2) % g != 0: compatible
    
    # This structure allows us to skip the gcd lookup in the inner loop.
    
    # Let's restructure the bitset building.
    # adj[i] is an int.
    
    # We will iterate moduli pairs.
    # Get indices for m1 and m2.
    
    # This is O(N^2) in the worst case but with better constants.
    
    # However, 8255 nodes is small enough that a simple O(N^2) with precomputed gcds might be fine.
    # Let's try the simple O(N^2) first, but with a time check.
    
    # Actually, 64 million iterations in Python is risky (could be 10-30s).
    # Let's try to be smarter.
    # We can represent the "compatible residues" for a given (m1, r1) with respect to m2.
    # For fixed m1, r1, m2:
    # r2 is compatible if r2 % g != r1 % g.
    # So for a fixed m2, the set of compatible residues is the union of (g-1) residue classes mod g.
    
    # Let's stick to the O(N^2) but optimized.
    # We'll use a list of lists for adjacency during construction, then convert to bitsets?
    # No, Python ints are best for bitsets.
    
    # Let's try to build the bitsets directly.
    
    # To save time, we can parallelize? No, single thread.
    
    # Let's estimate: 8255^2 / 2 = 34 million pairs.
    # Each pair: get m1, m2, g, r1, r2, mod, compare.
    # In Python, ~100ns per op? 34M * 100ns = 3.4s.
    # Plus overhead. Maybe 5-10s. Acceptable.
    
    # Implementation:
    # 1. Loop i from 0 to N-1
    # 2. Loop j from i+1 to N-1
    # 3. Check compatibility
    # 4. Update bitsets
    
    # We can speed this up by precomputing the "conflict" or "compat" status for modulus pairs.
    
    # Let's do this:
    # Precompute for each pair of moduli (m1, m2), a table compat[m1][m2][r1][r2] -> bool?
    # That's too big.
    
    # Just do the double loop.
    
    # To ensure we don't TLE on graph construction, we add a time check.
    # If graph construction takes too long, we fall back to a heuristic.
    
    # Let's write the graph construction carefully.
    
    # Optimization: 
    # Group nodes by modulus.
    # For m1 in moduli:
    #   for m2 in moduli where m2 > m1:
    #      g = gcd(m1, m2)
    #      for i in indices[m1]:
    #         r1 = res[i]
    #         for j in indices[m2]:
    #            r2 = res[j]
    #            if (r1 - r2) % g != 0:
    #                adj[i] |= 1 << j
    #                adj[j] |= 1 << i
    
    # This avoids checking m1==m2 cases and gcd lookups in inner loop.
    
    # Let's implement this.
    
    indices_by_mod = {m: [] for m in range(2, MAX_MOD + 1)}
    for idx, (m, r) in enumerate(candidates):
        indices_by_mod[m].append(idx)
        
    # Precompute gcd for modulus pairs
    gcd_table = {}
    for m1 in range(2, MAX_MOD + 1):
        for m2 in range(m1 + 1, MAX_MOD + 1):
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
            
    # Build bitsets
    # We need to be careful with bit shifting for large N (8255 bits). Python handles big ints natively.
    
    # Time check
    t_start_graph = time.time()
    
    for m1 in range(2, MAX_MOD + 1):
        if time.time() > deadline - 20: # Leave 20s for search
            # Fallback: Random greedy if graph construction is too slow
            # But 8000 nodes should be fast enough.
            pass
            
        idxs1 = indices_by_mod[m1]
        for m2 in range(m1 + 1, MAX_MOD + 1):
            idxs2 = indices_by_mod[m2]
            g = gcd_table[(m1, m2)]
            
            # For each r1 in m1
            for i in idxs1:
                r1 = node_res[i]
                bit_i = 1 << i
                # For each r2 in m2
                for j in idxs2:
                    r2 = node_res[j]
                    if (r1 - r2) % g != 0:
                        adj[i] |= 1 << j
                        adj[j] |= bit_i
                        
    # 3. Maximum Clique Algorithm (Tomita's B&B)
    # We work on the Compatibility Graph.
    # We want to find the largest clique.
    
    # Tomita's algorithm:
    # expand(R, P)
    #   if P is empty: update best
    #   order P by degree
    #   for v in P:
    #       R' = R U {v}
    #       P' = P intersect N(v)
    #       expand(R', P')
    #       P = P \ {v}
    #       if |R| + |P| <= best_size: break
    
    # We represent P as a set of node indices.
    # To speed up, we can use bitsets for P as well?
    # Python sets are okay for N=8000, but bitsets are faster for intersection.
    # Let's use Python ints for P too.
    
    best_clique = []
    best_size = 0
    
    # Initial P is all nodes.
    # We need to order nodes. A good heuristic is to sort by degree in descending order.
    # Degree of node i = popcount(adj[i])
    
    # Calculate degrees
    degrees = [0] * n_nodes
    for i in range(n_nodes):
        # Python int bit_count is available in 3.10+
        # If not, bin(x).count('1')
        degrees[i] = adj[i].bit_count() if hasattr(adj[i], 'bit_count') else bin(adj[i]).count('1')
        
    # Sort nodes by degree descending
    # We will iterate in this order.
    # However, the standard Tomita algorithm orders P dynamically.
    # For simplicity and speed in Python, we can use a static order based on initial degree,
    # or re-sort P at each step (expensive).
    # Let's try static order first, it often works well.
    # Actually, dynamic ordering is crucial for B&B performance.
    # But re-sorting a set of size ~8000 at every node in the recursion tree is O(N log N) per node.
    # Depth is ~size of clique (maybe 20-30?).
    # Total nodes visited in B&B can be large.
    
    # Alternative: Use the "Color Sort" heuristic from Tomita's paper.
    # This is complex to implement correctly in Python quickly.
    
    # Let's try a simpler B&B with bitsets for P.
    # P is an int.
    # R is a list (or set) of nodes in the current clique.
    
    # We need a function to get the next node from P.
    # If we use a static order, we can just iterate through the sorted list and check if the bit is set in P.
    
    sorted_nodes = sorted(range(n_nodes), key=lambda x: degrees[x], reverse=True)
    
    # To make "remove from P" and "iterate" fast, we'll stick to the bitset P.
    # But iterating a bitset in Python requires finding set bits.
    # We can precompute a list of nodes in order, and for each, check if it's in P.
    
    # Let's implement the recursive search.
    
    sys.setrecursionlimit(10000)
    
    def expand(R, P):
        nonlocal best_size, best_clique, deadline
        
        if time.time() > deadline:
            return
            
        # Heuristic: If |R| + |P| <= best_size, prune
        # Count bits in P
        p_size = P.bit_count() if hasattr(P, 'bit_count') else bin(P).count('1')
        if len(R) + p_size <= best_size:
            return
            
        # If P is empty, update best
        if p_size == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
            
        # Iterate through nodes in P in order of degree (or static order)
        # We use the static sorted_nodes list.
        # We need to iterate through nodes in P.
        # To do this efficiently, we can just loop through sorted_nodes and check if node is in P.
        # But we need to remove nodes from P as we go.
        
        # Optimization: Instead of checking all N nodes, we can extract set bits from P.
        # But we want them in order of degree.
        # Since sorted_nodes is static, we can just iterate it and filter.
        
        # However, P changes.
        # Let's iterate through sorted_nodes.
        # For each node v in sorted_nodes:
        #   if v is in P:
        #      ...
        #      P = P & ~(1 << v)
        #      if len(R) + bit_count(P) <= best_size: break
        
        # This is O(N) per level. Depth is small. N=8000.
        # 8000 * 30 = 240k operations. Very fast.
        
        # Wait, the loop inside expand iterates through ALL N nodes to find those in P.
        # If P is sparse, this is wasteful.
        # But N is only 8000.
        
        # Let's try it.
        
        # Copy P to local variable? No, we modify P.
        # We pass P by value (int is immutable).
        
        for v in sorted_nodes:
            bit_v = 1 << v
            if not (P & bit_v):
                continue
                
            # v is in P
            
            # Branch: include v
            R.append(v)
            P_new = P & adj[v]
            expand(R, P_new)
            R.pop()
            
            # Remove v from P
            P = P & ~bit_v
            
            # Prune check
            p_size = P.bit_count() if hasattr(P, 'bit_count') else bin(P).count('1')
            if len(R) + p_size <= best_size:
                break

    # Initial call
    # P = all nodes
    initial_P = (1 << n_nodes) - 1
    expand([], initial_P)
    
    # 4. Construct Result
    # best_clique contains node indices.
    # Convert back to [residue, modulus] pairs.
    result = []
    for idx in best_clique:
        m, r = candidates[idx]
        result.append([r, m])
        
    # Verify and Fallback
    if not result:
        # Fallback to simple greedy if B&B failed or found nothing
        # This shouldn't happen, but safety first.
        result = [[1, 2]]
        
    # Double check validity
    def is_valid(classes):
        n = len(classes)
        for i in range(n):
            r1, m1 = classes[i]
            for j in range(i+1, n):
                r2, m2 = classes[j]
                g = math.gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True
        
    if not is_valid(result):
        # If something went wrong, use a safe small set
        result = []
        # Greedy fallback
        selected = []
        for m in range(2, MAX_MOD + 1):
            for r in range(m):
                valid = True
                for r_p, m_p in selected:
                    g = math.gcd(m, m_p)
                    if (r - r_p) % g == 0:
                        valid = False
                        break
                if valid:
                    selected.append([r, m])
                    break
        result = selected
        if not result:
            result = [[1, 2]]

    return result

# EVOLVE-BLOCK-END
