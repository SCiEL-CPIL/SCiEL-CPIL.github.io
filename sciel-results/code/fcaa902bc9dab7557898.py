"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 256

    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Build conflict graph: edge between u and v if gcd(u,v) > 1
    # We want Maximum Weight Independent Set on conflict graph
    # = Maximum Weight Clique on complement graph (coprime graph)
    
    n_nodes = N  # Nodes 0..255 correspond to values 1..256
    weights = [i + 1 for i in range(n_nodes)]
    
    # Build coprime adjacency as bitmasks
    adj_mask = [0] * n_nodes
    for i in range(n_nodes):
        vi = i + 1
        for j in range(i + 1, n_nodes):
            vj = j + 1
            if math.gcd(vi, vj) == 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Precompute: for each node, store its weight
    # Use a faster weight-sum method: precompute cumulative sums isn't straightforward
    # for arbitrary bitmasks, so we use a lookup-based approach for small masks
    # Actually, let's just optimize the bit iteration
    
    def mask_weight_sum(mask):
        """Compute sum of weights for nodes in mask using bit tricks."""
        total = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            total += weights[idx]
            m ^= lsb
        return total

    # Greedy heuristic to get initial lower bound - try multiple orderings
    def greedy_clique(order=None):
        P = (1 << n_nodes) - 1
        R = 0
        w = 0
        while P:
            if order is None:
                # Pick max weight node in P
                best_node = -1
                best_w = -1
                pm = P
                while pm:
                    lsb = pm & -pm
                    idx = lsb.bit_length() - 1
                    if weights[idx] > best_w:
                        best_w = weights[idx]
                        best_node = idx
                    pm ^= lsb
            else:
                # Pick first node in P according to order
                best_node = -1
                for idx in order:
                    if (P >> idx) & 1:
                        best_node = idx
                        break
            
            if best_node == -1:
                break
            
            R |= (1 << best_node)
            w += weights[best_node]
            P = P & adj_mask[best_node] & ~(1 << best_node)
        
        return R, w

    # Try multiple greedy strategies to get a good initial bound
    init_R, init_w = greedy_clique()
    global_best_weight = init_w
    global_best_mask = init_R
    
    # Try greedy with different orderings (e.g., by weight descending, by degree)
    deg_order = sorted(range(n_nodes), key=lambda i: (weights[i], adj_mask[i].bit_count()), reverse=True)
    R2, w2 = greedy_clique(order=deg_order)
    if w2 > global_best_weight:
        global_best_weight = w2
        global_best_mask = R2
    
    # Try greedy picking lowest weight first (to maximize count, then refine)
    low_order = sorted(range(n_nodes), key=lambda i: weights[i])
    R3, w3 = greedy_clique(order=low_order)
    if w3 > global_best_weight:
        global_best_weight = w3
        global_best_mask = R3
    
    # Try greedy picking by weight/degree ratio
    ratio_order = sorted(range(n_nodes), key=lambda i: weights[i] / max(1, adj_mask[i].bit_count()), reverse=True)
    R4, w4 = greedy_clique(order=ratio_order)
    if w4 > global_best_weight:
        global_best_weight = w4
        global_best_mask = R4

    # Branch and bound with Bron-Kerbosch style pivot
    sys.setrecursionlimit(10000)
    
    # Time check counter to reduce syscall frequency
    time_check_counter = [0]
    
    def should_check_time():
        time_check_counter[0] += 1
        if time_check_counter[0] % 100 == 0:
            return time.time() > deadline
        return False
    
    def bk_search(R_mask, P_mask, X_mask, current_weight):
        nonlocal global_best_weight, global_best_mask
        
        if should_check_time():
            return
            
        if not P_mask:
            if current_weight > global_best_weight:
                global_best_weight = current_weight
                global_best_mask = R_mask
            return
            
        # Upper bound: current_weight + sum of all weights in P
        # Quick check: if even adding all of P can't beat best, prune
        bound = current_weight + mask_weight_sum(P_mask)
        if bound <= global_best_weight:
            return
            
        # Pivot selection: pick node in P with max degree in P
        pivot_idx = -1
        max_deg = -1
        pm = P_mask
        while pm:
            lsb = pm & -pm
            idx = lsb.bit_length() - 1
            deg = (P_mask & adj_mask[idx]).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot_idx = idx
            pm ^= lsb
        
        if pivot_idx == -1:
            lsb = P_mask & -P_mask
            pivot_idx = lsb.bit_length() - 1
        
        # Branch on vertices in P that are NOT adjacent to pivot
        verts_to_iter = P_mask & ~adj_mask[pivot_idx]
        
        # Collect and sort by weight descending for better pruning
        verts_list = []
        vm = verts_to_iter
        while vm:
            lsb = vm & -vm
            idx = lsb.bit_length() - 1
            verts_list.append(idx)
            vm ^= lsb
        
        verts_list.sort(key=lambda x: weights[x], reverse=True)
        
        for v in verts_list:
            if should_check_time():
                return
            
            new_R = R_mask | (1 << v)
            new_P = P_mask & adj_mask[v] & ~(1 << v)
            new_X = X_mask & adj_mask[v] & ~(1 << v)
            new_weight = current_weight + weights[v]
            
            bk_search(new_R, new_P, new_X, new_weight)
            
            # Update P and X for next iteration
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)
    
    # Initial call
    initial_P = (1 << n_nodes) - 1
    bk_search(0, initial_P, 0, 0)
    
    # Local search refinement: try to improve by swapping
    if time.time() < deadline - 5.0:
        result_mask = global_best_mask
        result_set = set()
        m = result_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            result_set.add(idx)
            m ^= lsb
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            # Try adding a node not in the set
            for add_idx in range(n_nodes):
                if add_idx in result_set:
                    continue
                if time.time() > deadline - 3.0:
                    break
                # Check if add_idx is coprime with all in result_set
                compatible = True
                for other in result_set:
                    if not (adj_mask[add_idx] >> other) & 1:
                        compatible = False
                        break
                if compatible:
                    new_weight = global_best_weight + weights[add_idx]
                    if new_weight > global_best_weight:
                        result_set.add(add_idx)
                        global_best_weight = new_weight
                        improved = True
                        break
            
            # Try removing a node and adding a heavier one
            if time.time() > deadline - 3.0:
                break
            for rem_idx in list(result_set):
                if time.time() > deadline - 3.0:
                    break
                for add_idx in range(n_nodes):
                    if add_idx in result_set or add_idx == rem_idx:
                        continue
                    if weights[add_idx] <= weights[rem_idx]:
                        continue
                    # Check if add_idx is coprime with all in (result_set - {rem_idx})
                    compatible = True
                    for other in result_set:
                        if other == rem_idx:
                            continue
                        if not (adj_mask[add_idx] >> other) & 1:
                            compatible = False
                            break
                    if compatible:
                        new_weight = global_best_weight - weights[rem_idx] + weights[add_idx]
                        if new_weight > global_best_weight:
                            result_set.remove(rem_idx)
                            result_set.add(add_idx)
                            global_best_weight = new_weight
                            improved = True
                            break
                if improved:
                    break
        
        # Update mask from result_set
        new_mask = 0
        for idx in result_set:
            new_mask |= (1 << idx)
        global_best_mask = new_mask

    # Extract result
    result = []
    mask = global_best_mask
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        mask ^= lsb
    
    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback: return 1 and all primes (valid but suboptimal)
                return [1] + primes
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
