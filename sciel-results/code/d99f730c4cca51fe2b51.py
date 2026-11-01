import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    # 150s limit. 10s safety margin.
    deadline = start_time + 140.0

    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # Lower bound on M: need total_diffs_needed distinct differences in {1..M}
    M_lower = total_diffs_needed

    # Upper bound from greedy
    greedy_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(greedy_rows)
    best_rows = greedy_rows

    # Strategy:
    # 1. Generate a pool of candidate rows for a given M.
    # 2. Build a conflict graph (bitset representation).
    # 3. Use Branch and Bound to find an independent set of size n_rows.
    # 4. Iterate M from lower bound up to current best, trying to find a valid set.
    
    # We want to find the smallest M.
    # Start from M_lower, go up.
    # If M is too small, no solution. If M is large, solution likely exists.
    # We know best_M from greedy. We try to improve it.
    
    # To speed up, we can start from best_M - 1 downwards? No, we want minimal M.
    # So we start from M_lower and go up.
    # However, if M_lower is far from best_M, this might take long if we fail many times.
    # Better: Start from best_M - 1 and go down? No, we want to *find* a solution.
    # If we find a solution for M, it's valid. We want the *smallest* M.
    # So we must check M_lower, M_lower+1, ... until we find one.
    # If we don't find one by best_M-1, we return best_M.
    
    # Optimization: The number of candidates grows with M.
    # For small M, few candidates -> fast check.
    # For large M, many candidates -> slow check.
    # So checking small M first is efficient in terms of "work per M" if it fails quickly.
    # But if it fails, we move to M+1.
    
    # Let's try M in range(M_lower, best_M):
    
    # Pre-compute nothing? The candidate generation depends on M.
    
    # To avoid generating duplicates or too many, we limit candidate pool size.
    # If pool is too small, we might miss the solution.
    # If pool is too large, B&B is slow.
    
    # Heuristic: For each M, generate up to MAX_CANDIDATES valid rows.
    # If we can't find n_rows disjoint ones in this pool, try larger M.
    
    MAX_CANDIDATES = 2000
    MAX_BNB_NODES = 500000  # Limit on B&B search space exploration
    
    # We will iterate M from M_lower to best_M - 1
    # If we find a solution, return it.
    # If we exhaust the range, return best_rows (greedy).
    
    # Note: If best_M is close to M_lower, this loop is short.
    # For (7,5), M_lower=105, best_M~112-115. Loop is small.
    
    for M in range(M_lower, best_M):
        if time.time() > deadline - 5:
            break
            
        # Generate candidates for this M
        candidates = _generate_candidates(k, M, MAX_CANDIDATES, deadline)
        
        if len(candidates) < n_rows:
            continue
            
        # Build conflict graph
        # Node i conflicts with Node j if they share any difference.
        # Represent each node's used diffs as a Python int (bitset).
        # Diff values are up to M. M is ~112. Fits in int.
        
        bitsets = []
        for row in candidates:
            mask = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    mask |= (1 << d)
            bitsets.append(mask)
        
        # Adjacency list or bitset of neighbors?
        # For B&B, we need to quickly find neighbors of a node.
        # Let's precompute a list of neighbors for each node.
        # N is at most 2000. N^2 is 4M. Bitwise AND is fast.
        # neighbors[i] = list of j such that (bitsets[i] & bitsets[j]) != 0
        
        neighbors = [[] for _ in range(len(candidates))]
        for i in range(len(candidates)):
            bi = bitsets[i]
            for j in range(i + 1, len(candidates)):
                bj = bitsets[j]
                if (bi & bj) != 0:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        
        # B&B to find Independent Set of size n_rows
        # State: (available_nodes_bitset, current_set_size, current_max_entry_in_set)
        # Wait, we want to minimize the global max entry.
        # Since all candidates have max entry <= M, any valid set has max entry <= M.
        # So we just need *any* independent set of size n_rows.
        # The "score" is just M.
        
        # B&B:
        # We pick a node, add to set, remove it and its neighbors from available.
        # Recurse.
        # Pruning:
        # 1. If current_set_size + max_possible_additions < n_rows: prune.
        #    Max possible additions: We can use a greedy coloring bound (Welsh-Powell) on the induced subgraph of available nodes to estimate clique number (which is upper bound on independent set size? No, independent set in G is clique in complement. Or use greedy bound on G directly: color G, number of colors is upper bound on Independent Set? No, number of colors is upper bound on Clique. For Independent Set, we can use a greedy packing bound: pick a node, remove it and neighbors, repeat. Count is lower bound. Upper bound is harder. Simple bound: size of available set).
        #    Actually, a simple upper bound on the size of an independent set in a graph is the number of vertices.
        #    A better upper bound: The independence number of a graph is at most the number of vertices.
        #    We can use a greedy heuristic to get a *lower* bound on the size of the independent set we *can* find, but for pruning we need an *upper* bound on what is *possible*.
        #    Upper bound: If the graph is dense, IS is small.
        #    Let's use a simple bound: if len(available) < n_rows - current_size: prune.
        #    This is weak.
        
        #    Better: Use the fact that we need n_rows nodes.
        #    If we have picked `picked` nodes, we need `rem` more.
        #    The available nodes must contain an independent set of size `rem`.
        #    A necessary condition: The available subgraph must not be "too connected".
        #    Simple prune: If the minimum degree in the available subgraph is high, it's hard to find large IS.
        #    Specifically, if for every node in available, its degree in available >= n_rows - current_size - 1, then we can't pick another node? No.
        
        #    Let's stick to a simple recursive search with bitsets for speed.
        #    Order nodes by degree (ascending) to find IS faster?
        #    Actually, for finding *any* IS of size K, picking low-degree nodes first is often good.
        
        #    Let's implement a recursive search.
        #    To avoid recursion depth issues, we can use a stack, but recursion is fine for K=7.
        
        found = False
        solution = []
        
        def bnb(available_mask, current_set, current_size):
            nonlocal found, solution
            
            if found:
                return
            
            if current_size == n_rows:
                found = True
                solution = list(current_set)
                return
            
            # Pruning:
            # Count available nodes
            avail_count = bin(available_mask).count('1')
            if avail_count < n_rows - current_size:
                return
            
            # Heuristic: Try nodes in order of increasing degree in the available subgraph?
            # Precomputing degrees dynamically is expensive.
            # Let's just iterate through available nodes.
            # To speed up, we can order them once.
            
            # Get list of available nodes
            # Since N <= 2000, we can iterate 0..N-1 and check bit.
            # But we can be smarter.
            
            # Let's try a few random orders or fixed order.
            # Fixed order 0..N-1 is fine if we prune well.
            
            # Optimization: If we fail to find a solution, we might want to try a different strategy.
            # But for now, simple DFS.
            
            # To improve performance, we can use a "forward checking" or similar.
            # Here, we just pick a node, remove it and neighbors.
            
            # Iterate through available nodes
            # We can stop early if we find one.
            
            # To avoid exploring symmetric branches, we can impose an ordering on the set we build?
            # E.g., indices in current_set are increasing.
            # This reduces branching factor.
            
            last_picked = -1
            for i in range(len(candidates)):
                if i <= last_picked:
                    continue
                if not (available_mask & (1 << i)):
                    continue
                
                # Try picking i
                new_available = available_mask & ~(1 << i)
                # Remove neighbors of i
                for nb in neighbors[i]:
                    new_available &= ~(1 << nb)
                
                current_set.append(i)
                bnb(new_available, current_set, current_size + 1)
                if found:
                    return
                current_set.pop()
                
                last_picked = i
                
                # Safety check for time
                if time.time() > deadline - 2:
                    return
        
        bnb((1 << len(candidates)) - 1, [], 0)
        
        if found:
            # Extract rows
            new_rows = [candidates[idx] for idx in solution]
            # Verify
            if _is_valid(new_rows, k):
                best_M = M
                best_rows = new_rows
                break # Found minimal M, stop
        
    return best_rows

def _generate_candidates(k, M, max_count, deadline):
    """
    Generate up to max_count valid rows of length k+1 starting at 0, max entry <= M.
    """
    candidates = []
    seen = set() # To avoid duplicate rows (as tuples)
    
    # To get diverse candidates, we can use random sampling.
    # But we also want "good" ones? For MIS, we just need a diverse set of nodes.
    # Random is good for coverage.
    
    attempts = 0
    max_attempts = max_count * 50 # Limit total work
    
    while len(candidates) < max_count and attempts < max_attempts and time.time() < deadline - 10:
        attempts += 1
        # Pick k distinct values from 1..M
        if M < k:
            break
        vals = random.sample(range(1, M + 1), k)
        vals.sort()
        row = [0] + vals
        
        # Check Sidon property (distinct diffs within row)
        d_set = set()
        valid = True
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    valid = False
                    break
                d_set.add(d)
        
        if not valid:
            continue
            
        # Check duplicate
        t = tuple(row)
        if t in seen:
            continue
        seen.add(t)
        candidates.append(row)
        
    return candidates

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False

        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True

# Fallback if B&B is too slow or fails to find solution in range
# We already have best_rows from greedy.

if __name__ == "__main__":
    # Test
    rows = construct(7, 5)
    print(rows)
    print("Max:", _max_entry(rows))
    print("Valid:", _is_valid(rows, 5))

