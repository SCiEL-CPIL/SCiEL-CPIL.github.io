"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    
    N = 256
    
    # Precompute primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    
    # For each number 1..N, compute its set of prime factors
    prime_factors = [set() for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            prime_factors[multiple].add(p)
    
    # Build conflict graph: edge between i and j if gcd(i,j) > 1
    # We want Maximum Weight Independent Set on this graph
    # Equivalently: Maximum Weight Clique on the complement graph
    
    try:
        import networkx as nx
        
        # Build the complement graph (where edges mean "coprime")
        # Node i (1-indexed) connects to node j if gcd(i,j) == 1
        G_complement = nx.Graph()
        G_complement.add_nodes_from(range(1, N + 1))
        
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                if gcd(i, j) == 1:
                    G_complement.add_edge(i, j)
        
        # Assign weights = node values
        for node in G_complement.nodes():
            G_complement.nodes[node]['weight'] = node
        
        # Find maximum weight clique in complement graph
        # This equals maximum weight independent set in original conflict graph
        if time.time() < deadline:
            clique = nx.algorithms.approximation.max_weight_clique(G_complement)
            if clique:
                result = [int(x) for x in clique]
                return result
    except Exception:
        pass
    
    # Fallback: greedy approach with local search
    selected = [1]
    used_primes = set()
    
    for i in range(N, 1, -1):
        if prime_factors[i] & used_primes:
            continue
        selected.append(i)
        used_primes |= prime_factors[i]
    
    # Local search: try 1-for-1 and 2-for-1 swaps
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        current_sum = sum(selected)
        selected_set = set(selected)
        
        # Try 1-for-1 swaps
        for i in range(N, 1, -1):
            if i in selected_set:
                continue
            if prime_factors[i] & used_primes:
                continue
            # Try removing each selected element (except 1) to see if we can add i
            for idx in range(len(selected) - 1, -1, -1):
                if selected[idx] == 1:
                    continue
                removed = selected[idx]
                # Check if i is compatible with all others except removed
                temp_used = used_primes - prime_factors[removed]
                if not (prime_factors[i] & temp_used):
                    # This swap is valid
                    new_sum = current_sum - removed + i
                    if new_sum > current_sum:
                        selected.remove(removed)
                        selected.append(i)
                        selected_set.discard(removed)
                        selected_set.add(i)
                        used_primes = used_primes - prime_factors[removed] | prime_factors[i]
                        current_sum = new_sum
                        improved = True
                        break
        
        # Try adding any compatible number
        for i in range(N, 1, -1):
            if i in selected_set:
                continue
            if prime_factors[i] & used_primes:
                continue
            selected.append(i)
            selected_set.add(i)
            used_primes |= prime_factors[i]
            current_sum += i
            improved = True
        
        # Try 2-for-1 swaps: remove two small, add one large
        if not improved:
            for i in range(N, 1, -1):
                if i in selected_set:
                    continue
                if prime_factors[i] & used_primes:
                    continue
                # Find two elements whose removal would allow adding i
                candidates_to_remove = []
                for idx in range(len(selected) - 1, -1, -1):
                    if selected[idx] == 1:
                        continue
                    if prime_factors[i] & prime_factors[selected[idx]]:
                        candidates_to_remove.append(idx)
                
                if len(candidates_to_remove) >= 2:
                    # Try all pairs
                    for a in range(len(candidates_to_remove)):
                        for b in range(a + 1, len(candidates_to_remove)):
                            idx_a = candidates_to_remove[a]
                            idx_b = candidates_to_remove[b]
                            rem_a = selected[idx_a]
                            rem_b = selected[idx_b]
                            
                            # Check if removing both allows adding i
                            temp_used = used_primes - prime_factors[rem_a] - prime_factors[rem_b]
                            if not (prime_factors[i] & temp_used):
                                new_sum = current_sum - rem_a - rem_b + i
                                if new_sum > current_sum:
                                    # Perform the swap
                                    # Remove both (be careful with indices)
                                    max_idx = max(idx_a, idx_b)
                                    min_idx = min(idx_a, idx_b)
                                    selected.pop(max_idx)
                                    selected.pop(min_idx)
                                    selected.append(i)
                                    selected_set.discard(rem_a)
                                    selected_set.discard(rem_b)
                                    selected_set.add(i)
                                    used_primes = temp_used | prime_factors[i]
                                    current_sum = new_sum
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
    
    return [int(x) for x in selected]

# EVOLVE-BLOCK-END
