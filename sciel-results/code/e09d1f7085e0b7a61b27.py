import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 330s limit
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def bits_to_list(mask):
        """Convert bitmask to list of set bit positions."""
        result = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            result.append(idx)
            temp ^= lsb
        return result

    def count_bits(mask):
        return bin(mask).count('1')

    def find_all_k4s(adj, n):
        """Find all K4s in the graph using bitmask operations."""
        k4s = []
        for i in range(n):
            ni = bits_to_list(adj[i])
            for j in ni:
                if j <= i:
                    continue
                common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                cl = bits_to_list(common)
                for a in cl:
                    for b in cl:
                        if a >= b:
                            continue
                        if adj[a] & (1 << b):
                            k4s.append((1 << i) | (1 << j) | (1 << a) | (1 << b))
        return k4s

    def find_all_i4s(adj, n):
        """Find all independent sets of size 4 in the graph."""
        all_mask = (1 << n) - 1
        i4s = []
        for i in range(n):
            non_ni = all_mask & ~adj[i] & ~(1 << i)
            ni_list = bits_to_list(non_ni)
            for j in ni_list:
                if j <= i:
                    continue
                common_non = non_ni & ~adj[j] & ~(1 << i) & ~(1 << j)
                cnl = bits_to_list(common_non)
                for a in cnl:
                    for b in cnl:
                        if a >= b:
                            continue
                        if not (adj[a] & (1 << b)):
                            i4s.append((1 << i) | (1 << j) | (1 << a) | (1 << b))
        return i4s

    def is_valid(adj, n):
        """Check if the graph has no K5 and no I5."""
        k4s = find_all_k4s(adj, n)
        i4s = find_all_i4s(adj, n)
        all_mask = (1 << n) - 1
        for u in range(n):
            # K5 check: is there a K4 in N(u)?
            for k4 in k4s:
                if (k4 & adj[u]) == k4:
                    return False
            # I5 check: is there an I4 in non-N(u)?
            non_neigh = all_mask & ~adj[u] & ~(1 << u)
            for i4 in i4s:
                if (i4 & non_neigh) == i4:
                    return False
        return True

    # Verify seed
    if not is_valid(adj, n):
        return n, []

    best_n = n
    best_adj = adj[:]

    # 2. Try to extend using precomputed conflict sets
    current_n = n
    current_adj = adj[:]
    
    max_target = 43  # We want to reach 42 if possible
    
    for target_n in range(current_n + 1, max_target + 1):
        if time.time() > deadline:
            break
            
        found = False
        
        # Precompute all K4s and I4s in current graph
        k4s = find_all_k4s(current_adj, current_n)
        i4s = find_all_i4s(current_adj, current_n)
        all_mask = (1 << current_n) - 1
        
        # Generate candidate neighbor set sizes
        mid = current_n // 2
        size_candidates = []
        for offset in range(0, current_n + 1):
            s1 = mid - offset
            s2 = mid + offset
            if 0 <= s1 <= current_n:
                size_candidates.append(s1)
            if 0 <= s2 <= current_n and s2 != s1:
                size_candidates.append(s2)
        size_candidates = list(set(size_candidates))
        random.shuffle(size_candidates)
        
        # Limit attempts per size to keep time reasonable
        max_attempts = 300
        
        for size in size_candidates:
            if time.time() > deadline:
                break
            if found:
                break
            if size < 0 or size > current_n:
                continue
                
            attempts = max_attempts
            while attempts > 0 and not found:
                if time.time() > deadline:
                    break
                attempts -= 1
                
                # Random neighbor set of given size
                neighbors = random.sample(range(current_n), size)
                neighbors_mask = 0
                for v in neighbors:
                    neighbors_mask |= (1 << v)
                non_neighbors_mask = all_mask & ~neighbors_mask
                
                # Check K5: any K4 contained in neighbors_mask?
                k5_found = False
                for k4 in k4s:
                    if (k4 & neighbors_mask) == k4:
                        k5_found = True
                        break
                if k5_found:
                    continue
                    
                # Check I5: any I4 contained in non_neighbors_mask?
                i5_found = False
                for i4 in i4s:
                    if (i4 & non_neighbors_mask) == i4:
                        i5_found = True
                        break
                if i5_found:
                    continue
                    
                # Valid! Add vertex
                found = True
                new_n = current_n + 1
                new_adj = current_adj + [0]
                for v in neighbors:
                    new_adj[v] |= (1 << new_n)
                    new_adj[new_n] |= (1 << v)
                    
                current_n = new_n
                current_adj = new_adj
                best_n = current_n
                best_adj = current_adj[:]
                break

        if not found:
            # Try a few restarts with different random seeds
            restarts = 5
            for r in range(restarts):
                if time.time() > deadline:
                    break
                # Reset to best state
                current_n = best_n
                current_adj = best_adj[:]
                
                # Recompute k4s and i4s for this state
                k4s = find_all_k4s(current_adj, current_n)
                i4s = find_all_i4s(current_adj, current_n)
                all_mask = (1 << current_n) - 1
                
                # Try with different random choices
                mid = current_n // 2
                balanced_sizes = [s for s in range(max(0, mid-5), min(current_n+1, mid+6))]
                random.shuffle(balanced_sizes)
                
                for size in balanced_sizes:
                    if time.time() > deadline:
                        break
                    if found:
                        break
                    if size < 0 or size > current_n:
                        continue
                        
                    attempts = 400
                    while attempts > 0 and not found:
                        if time.time() > deadline:
                            break
                        attempts -= 1
                        
                        neighbors = random.sample(range(current_n), size)
                        neighbors_mask = 0
                        for v in neighbors:
                            neighbors_mask |= (1 << v)
                        non_neighbors_mask = all_mask & ~neighbors_mask
                        
                        k5_found = False
                        for k4 in k4s:
                            if (k4 & neighbors_mask) == k4:
                                k5_found = True
                                break
                        if k5_found:
                            continue
                            
                        i5_found = False
                        for i4 in i4s:
                            if (i4 & non_neighbors_mask) == i4:
                                i5_found = True
                                break
                        if i5_found:
                            continue
                        
                        found = True
                        new_n = current_n + 1
                        new_adj = current_adj + [0]
                        for v in neighbors:
                            new_adj[v] |= (1 << new_n)
                            new_adj[new_n] |= (1 << v)
                            
                        current_n = new_n
                        current_adj = new_adj
                        best_n = current_n
                        best_adj = current_adj[:]
                        break
                    if found:
                        break
            
            if not found:
                break

    # Reconstruct edges from best_adj
    final_edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i] & (1 << j):
                final_edges.append((i, j))
                
    return best_n, final_edges
# EVOLVE-BLOCK-END
