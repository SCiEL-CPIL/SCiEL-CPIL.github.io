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
    
    # Precompute weight sum lookup table using 8-bit chunks
    # For a 256-bit mask, split into 32 chunks of 8 bits each
    # wsum_table[chunk_idx][byte_value] = sum of weights for set bits in that byte
    CHUNK_SIZE = 8
    N_CHUNKS = (n_nodes + CHUNK_SIZE - 1) // CHUNK_SIZE  # 32 chunks
    wsum_table = [[0] * 256 for _ in range(N_CHUNKS)]
    for c in range(N_CHUNKS):
        base = c * CHUNK_SIZE
        for byte_val in range(256):
            s = 0
            for b in range(CHUNK_SIZE):
                if byte_val & (1 << b):
                    idx = base + b
                    if idx < n_nodes:
                        s += weights[idx]
            wsum_table[c][byte_val] = s

    def mask_weight_sum(mask):
        """Compute sum of weights for nodes in mask using precomputed table."""
        total = 0
        for c in range(N_CHUNKS):
            byte_val = (mask >> (c * CHUNK_SIZE)) & 0xFF
            total += wsum_table[c][byte_val]
        return total

    # Greedy heuristic to get initial lower bound
    def greedy_clique():
        P = (1 << n_nodes) - 1
        R = 0
        w = 0
        while P:
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
            
            if best_node == -1:
                break
            
            R |= (1 << best_node)
            w += weights[best_node]
            P = P & adj_mask[best_node] & ~(1 << best_node)
        
        return R, w

    init_R, init_w = greedy_clique()
    global_best_weight = init_w
    global_best_mask = init_R

    # Branch and bound with Bron-Kerbosch style pivot
    sys.setrecursionlimit(10000)
    
    def bk_search(R_mask, P_mask, X_mask, current_weight):
        nonlocal global_best_weight, global_best_mask
        
        if time.time() > deadline:
            return
            
        if not P_mask:
            if current_weight > global_best_weight:
                global_best_weight = current_weight
                global_best_mask = R_mask
            return
            
        # Upper bound: current_weight + sum of all weights in P
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
            if time.time() > deadline:
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
