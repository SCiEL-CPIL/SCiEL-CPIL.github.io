import time
import random
import math
from itertools import combinations

def _has_k222_fast(adj, n):
    """
    Fast check if the graph defined by adjacency list of bitmasks 'adj' on n vertices 
    contains K2,2,2 as a subgraph.
    adj[i] is a bitmask of neighbors of i.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            common_mask = adj[u] & adj[v]
            common_mask &= ~(1 << u)
            common_mask &= ~(1 << v)
            
            # Need at least 2 more common neighbors to form K2,2,2
            if common_mask.bit_count() < 2:
                continue
                
            cn_list = []
            m = common_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            
            L = len(cn_list)
            for i in range(L):
                b1 = cn_list[i]
                for j in range(i + 1, L):
                    b2 = cn_list[j]
                    common_b1_b2 = adj[b1] & adj[b2] & common_mask
                    if common_b1_b2.bit_count() >= 2:
                        return True
    return False

def _count_violations_fast(adj, n):
    """Count number of K2,2,2 subgraphs in the graph."""
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            common_mask = adj[u] & adj[v]
            common_mask &= ~(1 << u)
            common_mask &= ~(1 << v)
            
            if common_mask.bit_count() < 2:
                continue
                
            cn_list = []
            m = common_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            
            L = len(cn_list)
            for i in range(L):
                b1 = cn_list[i]
                for j in range(i + 1, L):
                    b2 = cn_list[j]
                    common_b1_b2 = adj[b1] & adj[b2] & common_mask
                    if common_b1_b2.bit_count() >= 2:
                        count += 1
    return count

def _find_violation_edges(adj, n, max_edges=10):
    """
    Find up to max_edges edges that are part of K2,2,2 subgraphs.
    Returns list of (u, v) tuples where edge (u,v) is in color 0 (adj[i]&adj[j]).
    If no violations in color 0, returns edges from color 1 (complement).
    """
    violations = []
    
    # Check color 0
    for u in range(n):
        if len(violations) >= max_edges:
            break
        for v in range(u + 1, n):
            if len(violations) >= max_edges:
                break
            if not (adj[u] >> v) & 1:
                continue
            common_mask = adj[u] & adj[v]
            common_mask &= ~(1 << u)
            common_mask &= ~(1 << v)
            
            if common_mask.bit_count() < 2:
                continue
                
            cn_list = []
            m = common_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            
            L = len(cn_list)
            found = False
            for i in range(L):
                b1 = cn_list[i]
                for j in range(i + 1, L):
                    b2 = cn_list[j]
                    common_b1_b2 = adj[b1] & adj[b2] & common_mask
                    if common_b1_b2.bit_count() >= 2:
                        violations.append((u, v))
                        found = True
                        break
                if found:
                    break
            if found:
                continue
                
    if violations:
        return violations
    
    # Check color 1 (complement)
    full = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
        
    for u in range(n):
        if len(violations) >= max_edges:
            break
        for v in range(u + 1, n):
            if len(violations) >= max_edges:
                break
            if not (comp[u] >> v) & 1:
                continue
            common_mask = comp[u] & comp[v]
            common_mask &= ~(1 << u)
            common_mask &= ~(1 << v)
            
            if common_mask.bit_count() < 2:
                continue
                
            cn_list = []
            m = common_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            
            L = len(cn_list)
            found = False
            for i in range(L):
                b1 = cn_list[i]
                for j in range(i + 1, L):
                    b2 = cn_list[j]
                    common_b1_b2 = comp[b1] & comp[b2] & common_mask
                    if common_b1_b2.bit_count() >= 2:
                        # Edge (u,v) is 1 in comp, so 0 in adj
                        violations.append((u, v))
                        found = True
                        break
                if found:
                    break
            if found:
                continue
                
    return violations

def _is_valid_fast(adj, n):
    """Check both color classes are K2,2,2-free using fast check."""
    if _has_k222_fast(adj, n):
        return False
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    if _has_k222_fast(comp, n):
        return False
    return True

def _total_violations(adj, n):
    """Count total K2,2,2 subgraphs in both color classes."""
    v0 = _count_violations_fast(adj, n)
    full = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    v1 = _count_violations_fast(comp, n)
    return v0 + v1

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n_target = 30
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    best_adj = None
    best_n = 29  # Fallback
    best_score = float('inf')
    
    # Multiple restarts with simulated annealing using violation count as objective
    num_restarts = 15
    iterations_per_restart = 6000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 8:
            break
            
        n = n_target
        adj = [0] * n
        
        # Initialize first 29 vertices with Paley graph
        for i in range(q):
            for j in range(i + 1, q):
                diff = (j - i) % q
                if diff in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                
        # Initialize vertex 29 (index 29) with different strategies
        if restart == 0:
            # Random
            for i in range(q):
                if random.random() < 0.5:
                    adj[i] |= (1 << 29)
                    adj[29] |= (1 << i)
        elif restart == 1:
            # All 0
            pass
        elif restart == 2:
            # All 1
            for i in range(q):
                adj[i] |= (1 << 29)
                adj[29] |= (1 << i)
        elif restart == 3:
            # Based on Paley pattern extended
            for i in range(q):
                diff = (29 - i) % q
                if diff in squares:
                    adj[i] |= (1 << 29)
                    adj[29] |= (1 << i)
        elif restart == 4:
            # Sparse random
            for i in range(q):
                if random.random() < 0.3:
                    adj[i] |= (1 << 29)
                    adj[29] |= (1 << i)
        elif restart == 5:
            # Dense random
            for i in range(q):
                if random.random() < 0.7:
                    adj[i] |= (1 << 29)
                    adj[29] |= (1 << i)
        else:
            # Random with varying bias
            bias = 0.2 + 0.6 * (restart - 6) / max(1, num_restarts - 6)
            for i in range(q):
                if random.random() < bias:
                    adj[i] |= (1 << 29)
                    adj[29] |= (1 << i)
        
        # Simulated annealing with violation count as objective
        temperature = 50.0
        cooling_rate = 0.9998
        best_local_adj = adj[:]
        best_local_violations = _total_violations(adj, n)
        current_violations = best_local_violations
        
        for iteration in range(iterations_per_restart):
            if time.time() > deadline - 3:
                break
            
            if current_violations == 0:
                # Valid coloring found!
                best_adj = adj[:]
                best_n = n
                best_score = 0
                break
            
            # Choose a random violating edge
            violations = _find_violation_edges(adj, n, max_edges=8)
            if not violations:
                # No violations found, should be valid
                best_adj = adj[:]
                best_n = n
                best_score = 0
                break
            
            u, v = random.choice(violations)
            
            # Compute delta: change in violation count
            old_adj = adj[:]
            
            # Flip edge (u, v)
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_violations = _total_violations(adj, n)
            delta = new_violations - current_violations
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta <= 0:
                current_violations = new_violations
                if new_violations < best_local_violations:
                    best_local_violations = new_violations
                    best_local_adj = adj[:]
            else:
                prob = math.exp(-delta / temperature) if temperature > 0 else 0
                if random.random() < prob:
                    current_violations = new_violations
                else:
                    # Reject: restore old state
                    adj = old_adj
            
            # Cool down
            temperature *= cooling_rate
            
            # Reheat if stuck
            if iteration % 500 == 499 and temperature < 1.0:
                temperature = 20.0
        
        # After each restart, update best if improved
        final_violations = _total_violations(best_local_adj, n)
        if final_violations < best_score:
            best_score = final_violations
            best_adj = best_local_adj[:]
            if best_score == 0:
                best_n = n
                break
    
    if best_n == 30:
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (best_adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
        
    # Fallback to n=29
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}
