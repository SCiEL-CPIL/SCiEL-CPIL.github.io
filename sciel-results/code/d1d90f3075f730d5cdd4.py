import time
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Candidates 2..256
    n = 256
    nodes = list(range(2, n + 1))
    num_nodes = len(nodes)
    
    # Map value to index for quick access
    val_to_idx = {v: i for i, v in enumerate(nodes)}
    
    # Weights: 1/v
    weights = [1.0 / v for v in nodes]
    
    # Build Conflict Graph (Adjacency List)
    # Conflict if lcm(a, b) <= 256
    # We store neighbors for each node.
    # Since we want Independent Set, we want to avoid conflicts.
    # In B&B for MWIS, we usually work on the graph G where edges are conflicts.
    # We want a set S such that no two nodes in S are connected in G.
    
    adj = [[] for _ in range(num_nodes)]
    # Bitmasks for fast intersection checks?
    # num_nodes = 255. Python ints are arbitrary precision, so bitmasks are fine.
    adj_mask = [0] * num_nodes
    
    for i in range(num_nodes):
        vi = nodes[i]
        for j in range(i + 1, num_nodes):
            vj = nodes[j]
            # Calculate LCM
            # lcm(vi, vj) = vi * vj // gcd(vi, vj)
            g = gcd(vi, vj)
            if (vi // g) * vj <= n:  # lcm <= 256
                # Conflict
                adj[i].append(j)
                adj[j].append(i)
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    # Sort nodes by weight descending?
    # Actually, for B&B, ordering matters.
    # High weight nodes should be considered early? Or low degree?
    # Let's stick to natural order (2..256) or sort by weight desc.
    # Weight desc means small numbers first. Small numbers have many conflicts.
    # This might prune quickly if we pick a small number?
    # Or maybe we should order by "degree" in conflict graph?
    # Let's try sorting by weight descending (small numbers first) as they are most valuable.
    
    # Reorder nodes for B&B
    # We need to remap adjacencies.
    # This is expensive. Let's just use the original indices and pass a list of candidates?
    # Standard B&B on MWIS:
    # State: (candidate_set, current_set)
    # Upper Bound: current_weight + sum(weights of candidates)
    # Better Bound: Greedy Coloring on the candidate subgraph?
    # Or simply: sum of weights of candidates.
    # Since weights are 1/x, the sum is harmonic-like.
    
    # Let's use a recursive B&B with bitmasks for candidates.
    # This is efficient for N=255.
    
    best_weight = 0.0
    best_set = []
    
    # Precompute weight sums for prefixes? No, candidates change dynamically.
    # We will maintain a bitmask of available candidates.
    
    # To speed up B&B, we can use a simple upper bound:
    # UB = current_weight + sum(1/x for x in candidates)
    # This is loose.
    # A tighter bound for MWIS:
    # If we color the conflict graph, the size of the largest independent set is bounded by the sum of the maximum weights of each color class? No.
    # Actually, for weighted IS, a common bound is the "greedy bound" or using LP relaxation.
    # Given the time limit, let's try a simple B&B with the sum of weights as UB first.
    # If it's too slow, we might need to add a coloring bound.
    
    # Optimization: Precompute the sum of weights for any subset? Impossible (2^255).
    # But we can compute sum(weights) for a bitmask quickly if we have a helper?
    # Iterating through set bits is O(k).
    
    def get_sum_weight(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    # This is O(N) per call. In B&B, we call it often.
    # We can maintain the current sum and the sum of remaining candidates?
    # Yes.
    
    # Let's refine the B&B logic.
    # State: (avail_mask, current_weight, current_set)
    # avail_mask: bitmask of nodes still available to be added.
    
    # Pruning:
    # 1. If current_weight + get_sum_weight(avail_mask) <= best_weight: return
    
    # Branching:
    # Pick a node from avail_mask.
    # Heuristic: Pick the node with highest weight? Or highest degree?
    # Let's pick the node with highest weight (smallest value) to maximize gain early.
    # Find index of max weight in avail_mask.
    
    def find_max_weight_idx(mask):
        max_w = -1.0
        max_idx = -1
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            if weights[idx] > max_w:
                max_w = weights[idx]
                max_idx = idx
            m ^= lsb
        return max_idx

    # This is O(N) per node selection.
    # Total nodes 255.
    # Depth can be up to 255.
    # 255 * 255 = 65k ops per path? No, it's O(N) per level, depth N. O(N^2) per path.
    # Number of paths? Exponential.
    
    # We need a better upper bound to prune.
    # The "sum of weights" bound is very loose because it assumes we can take ALL remaining nodes.
    # We can only take an independent set from the remaining nodes.
    # A better bound:
    # Partition the remaining nodes into cliques in the conflict graph?
    # If we can partition the remaining candidates into k cliques, the IS can contain at most one node from each clique.
    # So UB = current_weight + sum(max_weight(clique_i))
    # Finding a clique cover is hard.
    # However, the conflict graph is defined by LCM <= 256.
    # Numbers with a common prime factor p such that p*x <= 256?
    # If a and b share a factor p, and lcm(a,b) <= 256, they conflict.
    # Consider the set of multiples of p.
    # If we pick a node v, it conflicts with all its neighbors.
    
    # Let's try a simple improvement:
    # When we pick a node v, we remove v and all its neighbors from avail_mask.
    # The bound calculation `get_sum_weight(avail_mask)` is still loose.
    
    # Alternative: Use a greedy coloring bound on the *complement*? No.
    # Standard MWIS B&B uses a greedy coloring of the candidate subgraph to bound the *size* (for unweighted).
    # For weighted, we can use the sum of the largest weight in each color class?
    # If we color the conflict graph of the candidates, each color class is an independent set in the conflict graph?
    # No, a proper coloring of the conflict graph means adjacent nodes have different colors.
    # So a color class is an independent set in the conflict graph.
    # Wait, if C is a color class, it is an independent set.
    # We want to find the maximum weight independent set.
    # The maximum weight independent set is NOT bounded by the sum of max weights of color classes.
    # That bound works for the *size* of the IS (unweighted), where IS size <= number of colors.
    # For weighted, if we have a coloring c: V -> {1..k}, any independent set can contain at most one node of each color?
    # No! An independent set can contain multiple nodes of the same color if they are not adjacent.
    # In a proper coloring, nodes of the same color are NOT adjacent.
    # So a color class IS an independent set.
    # So we CAN take ALL nodes of a specific color class.
    # So the bound "sum of max weights of color classes" is WRONG.
    # The bound is: The IS is a subset of the union of color classes.
    # Actually, since each color class is an independent set, the entire color class can be included in the IS?
    # No, the IS must be independent in the WHOLE graph.
    # Nodes in different color classes might be adjacent.
    # So we can't just take all of them.
    
    # Let's stick to the simple bound but optimize the search order.
    # To make the simple bound work, we need to prune early.
    # The simple bound is only useful if the sum of remaining weights drops below best.
    # Since weights are 1/x, the sum drops slowly.
    
    # Let's try a different strategy:
    # The graph is sparse?
    # Let's check the density.
    # If the graph is sparse, the IS is large, and the sum of weights is a decent bound?
    # Actually, if the graph is sparse, we can take almost all nodes.
    # The bound `sum(weights)` is close to the optimal value.
    # Pruning will be effective if we find a good solution early.
    
    # Strategy:
    # 1. Greedy initialization to get a good `best_weight` quickly.
    # 2. B&B with `sum(weights)` bound.
    
    # Greedy Initialization
    def greedy_init():
        avail = (1 << num_nodes) - 1
        # Remove bit 0? No, nodes are 0..254 corresponding to 2..256.
        # Mask for all nodes: (1 << num_nodes) - 1
        curr_w = 0.0
        curr_set = []
        while avail:
            # Pick max weight node
            idx = find_max_weight_idx(avail)
            if idx == -1: break
            curr_w += weights[idx]
            curr_set.append(idx)
            # Remove idx and its neighbors
            avail &= ~(1 << idx)
            avail &= ~adj_mask[idx]
        return curr_w, curr_set

    best_weight, best_set_idx = greedy_init()
    best_set = [nodes[i] for i in best_set_idx]
    
    # B&B
    # We need to track the set of selected nodes to return it.
    # Passing a list is slow. We can reconstruct from the path?
    # Or store the set as a bitmask too?
    # If we store `selected_mask`, we can convert to list at the end.
    
    def bb(avail_mask, current_weight, selected_mask):
        nonlocal best_weight, best_set, best_set_idx
        
        if time.time() > deadline:
            return
            
        # Upper Bound
        # Sum of weights in avail_mask
        # Optimization: Precompute weight sums for chunks?
        # Or just iterate.
        # To speed up, we can maintain `remaining_sum`?
        # But `avail_mask` changes non-trivially (removing neighbors).
        # Let's just calculate it.
        
        # Quick check: if avail_mask is 0
        if avail_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set_idx = []
                m = selected_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    best_set_idx.append(idx)
                    m ^= lsb
                best_set = [nodes[i] for i in best_set_idx]
            return
            
        # Calculate UB
        ub = current_weight
        m = avail_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            ub += weights[idx]
            m ^= lsb
            
        if ub <= best_weight:
            return
            
        # Pick a node to branch on
        # Heuristic: Pick node with highest weight?
        # Or node with highest degree in avail subgraph?
        # High degree removes more neighbors, potentially pruning more.
        # Let's try highest weight first.
        idx = find_max_weight_idx(avail_mask)
        
        if idx == -1:
            return
            
        # Branch 1: Include idx
        # New avail: avail_mask without idx and its neighbors
        new_avail = avail_mask & ~(1 << idx)
        new_avail &= ~adj_mask[idx]
        
        bb(new_avail, current_weight + weights[idx], selected_mask | (1 << idx))
        
        # Branch 2: Exclude idx
        # New avail: avail_mask without idx
        new_avail2 = avail_mask & ~(1 << idx)
        
        bb(new_avail2, current_weight, selected_mask)

    # Start B&B
    # Initial avail: all nodes
    initial_avail = (1 << num_nodes) - 1
    bb(initial_avail, 0.0, 0)
    
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Result: {res}")
    print(f"Sum: {sum(1/x for x in res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if (res[i]*res[j])//gcd(res[i], res[j]) <= 256:
                valid = False
                print(f"Invalid pair: {res[i]}, {res[j]}")
    print(f"Valid: {valid}")
