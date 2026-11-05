"""Improved program: Greedy vertex addition with bitset-based (5,5)-freedom checking."""
import random
import time
from itertools import combinations

def _has_k5_or_ind5(adj_bits, n):
    """Check if graph has K5 or independent 5-set using bitmasks.
    adj_bits[i] is a bitmask of neighbors of vertex i.
    Returns True if either exists."""
    
    def has_clique_k5():
        # Backtracking with bitsets
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda v: bin(adj_bits[v]).count('1'), reverse=True)
        order_to_idx = {v: i for i, v in enumerate(order)}
        
        def backtrack(candidates_mask, current_size):
            if current_size == 5:
                return True
            if bin(candidates_mask).count('1') < 5 - current_size:
                return False
            # Try each candidate in order
            mask = candidates_mask
            while mask:
                # Get lowest set bit
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                mask ^= lsb
                # New candidates: intersection of current candidates (after v) with neighbors of v
                new_mask = candidates_mask & adj_bits[v] & ~(lsb - 1)
                # Actually, we need candidates that come after v in our ordering
                # Let's just use all remaining candidates
                new_mask = candidates_mask & adj_bits[v] & (mask | 0)  # remaining candidates
                # Better: new_mask = candidates_mask & adj_bits[v] & (candidates_mask ^ lsb)
                new_mask = candidates_mask & adj_bits[v] & (candidates_mask ^ lsb)
                if bin(new_mask).count('1') >= 5 - current_size - 1:
                    if backtrack(new_mask, current_size + 1):
                        return True
            return False
        
        return backtrack((1 << n) - 1, 0)
    
    if has_clique_k5():
        return True
    
    # Check for independent set of size 5: K5 in complement
    comp_adj_bits = [0] * n
    for i in range(n):
        # Complement neighbors: all except self and current neighbors
        all_mask = (1 << n) - 1
        comp_adj_bits[i] = all_mask & ~(adj_bits[i] | (1 << i))
    
    def has_ind5():
        def backtrack(candidates_mask, current_size):
            if current_size == 5:
                return True
            if bin(candidates_mask).count('1') < 5 - current_size:
                return False
            mask = candidates_mask
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                mask ^= lsb
                new_mask = candidates_mask & comp_adj_bits[v] & (candidates_mask ^ lsb)
                if bin(new_mask).count('1') >= 5 - current_size - 1:
                    if backtrack(new_mask, current_size + 1):
                        return True
            return False
        
        return backtrack((1 << n) - 1, 0)
    
    return has_ind5()


def _paley_graph(q):
    """Construct Paley graph on q vertices (q must be prime ≡ 1 mod 4)."""
    residues = {(x * x) % q for x in range(1, q)}
    adj_bits = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    return q, adj_bits


def construct():
    """Construct a (5,5)-free graph maximizing n."""
    start_time = time.time()
    time_limit = 280
    
    # Start with Paley graph on 37 vertices (known to be (5,5)-free)
    n, adj_bits = _paley_graph(37)
    
    # Verify
    if _has_k5_or_ind5(adj_bits, n):
        # Should not happen, but fallback to smaller
        n, adj_bits = _paley_graph(17)
    
    best_n = n
    best_adj_bits = adj_bits[:]
    
    random.seed(42)
    
    # Try to add vertices one at a time
    # For each new vertex, try random connection patterns
    max_iterations = 50000
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if n >= 42:
            break
        
        # Try to add a new vertex
        new_n = n + 1
        found = False
        
        # Try several random connection patterns
        for trial in range(50):
            # Generate random connection pattern for new vertex
            new_connections = 0
            for i in range(n):
                if random.random() < 0.5:
                    new_connections |= (1 << i)
            
            # Create new adj_bits
            new_adj_bits = adj_bits[:] + [0]
            for i in range(n):
                if (new_connections >> i) & 1:
                    new_adj_bits[i] |= (1 << new_n)
                    new_adj_bits[new_n] |= (1 << i)
            
            # Check if (5,5)-free
            if not _has_k5_or_ind5(new_adj_bits, new_n):
                adj_bits = new_adj_bits
                n = new_n
                found = True
                if n > best_n:
                    best_n = n
                    best_adj_bits = adj_bits[:]
                if n >= 42:
                    break
                break
        
        if n >= 42:
            break
        
        # If we couldn't add a vertex, try edge flips to create more flexibility
        if not found:
            # Try flipping a random edge
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a != b:
                old_a = adj_bits[a]
                old_b = adj_bits[b]
                # Flip edge (a, b)
                if (adj_bits[a] >> b) & 1:
                    adj_bits[a] &= ~(1 << b)
                    adj_bits[b] &= ~(1 << a)
                else:
                    adj_bits[a] |= (1 << b)
                    adj_bits[b] |= (1 << a)
                
                if not _has_k5_or_ind5(adj_bits, n):
                    # Keep the flip
                    pass
                else:
                    # Revert
                    adj_bits[a] = old_a
                    adj_bits[b] = old_b
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj_bits[i] >> j) & 1:
                edges.append((i, j))
    
    return best_n, edges
