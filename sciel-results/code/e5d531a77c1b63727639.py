import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Helper to get popcount
    def popcount(x):
        return bin(x).count('1')
    
    # For Python 3.10+, int.bit_count() is available and faster
    try:
        (1).bit_count()
        def pc(x):
            return x.bit_count()
    except AttributeError:
        def pc(x):
            return bin(x).count('1')

    def check_subgraph_regular(adj, mask):
        """
        Check if the induced subgraph on vertices in 'mask' is regular.
        adj: list of int bitmasks.
        mask: int bitmask of vertices.
        Returns True if regular, False otherwise.
        """
        degs = []
        m = mask
        while m:
            # Extract lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            # Degree of v within mask
            deg = pc(adj[v] & mask)
            degs.append(deg)
            # Early exit if not all equal
            if len(degs) > 1 and degs[0] != deg:
                return False
        return len(set(degs)) == 1

    def count_violations(adj, n, sizes):
        """
        Count the number of regular induced subgraphs of given sizes.
        Also returns a list of violating masks (limited) to speed up search.
        """
        total_violations = 0
        violating_masks = []
        
        for k in sizes:
            if k > n:
                continue
            # Iterate through all combinations of size k
            # Optimization: Use bit manipulation to generate masks? 
            # itertools.combinations is okay for k=6,7,8 on n=20.
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                if check_subgraph_regular(adj, mask):
                    total_violations += 1
                    if len(violating_masks) < 100: # Keep a small sample
                        violating_masks.append(mask)
                        
        return total_violations, violating_masks

    def full_check(adj, n):
        """
        Check ALL subsets of size >= 6.
        """
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if check_subgraph_regular(adj, mask):
                    return False
        return True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Target sizes for local search: 6, 7, 8 are most frequent and easiest to break
    # Sizes 9, 10 are also good to check during search to prevent large regular subgraphs forming
    search_sizes = [6, 7, 8]
    
    # Try n=21 first, then 20
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 15:
            break
            
        best_adj = None
        best_violations = float('inf')
        
        # Multiple random restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            # Slightly biased towards regular graphs? No, random 0.5 is fine.
            # Or start with a random regular graph of degree ~n/2?
            # Let's stick to random 0.5 for simplicity and diversity.
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Calculate initial violations
            curr_violations, curr_masks = count_violations(adj, n, search_sizes)
            
            if curr_violations == 0:
                # Check full validity
                if full_check(adj, n):
                    return {'n': n, 'edges': get_edges(adj, n)}
            
            # Local Search
            # Simulated Annealing parameters
            T = 10.0
            T_min = 0.01
            cooling_rate = 0.95
            
            for step in range(5000):
                if time.time() > deadline - 5:
                    break
                
                # Select a move: Flip an edge
                # Prefer edges involved in violations if possible
                if curr_masks and random.random() < 0.7:
                    mask = random.choice(curr_masks)
                    # Get vertices in mask
                    vs = []
                    m = mask
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        m ^= lsb
                        vs.append(v)
                    if len(vs) >= 2:
                        u, v = random.sample(vs, 2)
                    else:
                        u, v = random.sample(range(n), 2)
                else:
                    u, v = random.sample(range(n), 2)
                
                # Flip edge (u, v)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # Evaluate new state
                new_violations, new_masks = count_violations(adj, n, search_sizes)
                delta = new_violations - curr_violations
                
                # Accept if better, or with probability exp(-delta/T)
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    curr_violations = new_violations
                    curr_masks = new_masks
                    
                    if curr_violations == 0:
                        # Potential solution found, check full
                        if full_check(adj, n):
                            return {'n': n, 'edges': get_edges(adj, n)}
                else:
                    # Reject move, revert
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                
                T *= cooling_rate
                if T < T_min:
                    T = T_min

    # Fallback: Return a known valid small graph
    # n=10 random graph is almost certainly valid
    n = 10
    random.seed(42)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Verify fallback
    if full_check(adj, n):
        return {'n': n, 'edges': get_edges(adj, n)}
        
    return {'n': 5, 'edges': []}

if __name__ == "__main__":
    res = construct()
    print(f"Constructed graph with n={res['n']}, edges={len(res['edges'])}")
