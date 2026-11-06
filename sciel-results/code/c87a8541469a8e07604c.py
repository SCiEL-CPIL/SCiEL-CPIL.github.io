import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    # predecessors is likely a list of lists, where predecessors[i] is list of task indices that must precede i
    # Or a dict? The prompt says "instance has times, predecessors, cycle_time".
    # Standard format in these tasks is often list of lists.
    
    n = len(times)
    
    # Build adjacency list for successors
    successors = defaultdict(list)
    in_degree = np.zeros(n, dtype=int)
    
    for i in range(n):
        if i in predecessors:
            for p in predecessors[i]:
                successors[p].append(i)
                in_degree[i] += 1
        else:
            # Ensure all nodes are in the structure
            pass
            
    # Topological Sort (Kahn's Algorithm)
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topological_order = []
    while queue:
        u = queue.popleft()
        topological_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If topological sort doesn't include all nodes, there's a cycle. 
    # But ALBP instances should be DAGs.
    # We'll process in reverse topological order to compute downstream properties.
    
    # down_path[i]: Longest path time starting from i (including t[i])
    down_path = np.zeros(n, dtype=float)
    # trans_count[i]: Number of transitive successors
    trans_count = np.zeros(n, dtype=int)
    
    # Process in reverse topological order
    # We need to be careful with the order. 
    # Reverse topological order means we process sinks first, then their predecessors.
    rev_topo = topological_order[::-1]
    
    for i in rev_topo:
        # Base case: t[i]
        down_path[i] = times[i]
        
        max_successor_path = 0.0
        successor_trans_set = set()
        
        for s in successors[i]:
            # Update max path
            if down_path[s] > max_successor_path:
                max_successor_path = down_path[s]
            
            # Accumulate transitive successors
            # We need the set of all successors of s, plus s itself.
            # To avoid O(N^2) set operations which might be slow for N=30? 
            # N=30 is small. Sets are fine.
            successor_trans_set.add(s)
            # We need to store the transitive successors for each node to compute for its predecessors.
            # Let's store a bitmask or a set for each node.
            # Since N=30, a bitmask (int) is very efficient.
            
        # Actually, let's use bitmasks for transitive successors to be efficient and clean.
        # Re-initialize if we change strategy. 
        # For N=30, a simple set is fine too, but bitmask is faster for union.
        
    # Let's restart the computation with bitmasks for robustness and speed.
    
    # successors_mask[i] is a bitmask of all immediate and transitive successors of i.
    successors_mask = np.zeros(n, dtype=np.int64)
    
    # We need to compute down_path and successors_mask in reverse topological order.
    # down_path[i] = times[i] + max(down_path[s] for s in immediate_successors(i))
    # successors_mask[i] = immediate_successors(i) OR (union of successors_mask[s] for s in immediate_successors(i))
    
    # We need the list of immediate successors again.
    # We have 'successors' dict.
    
    for i in rev_topo:
        max_path = 0.0
        mask = 0
        
        for s in successors[i]:
            if down_path[s] > max_path:
                max_path = down_path[s]
            mask |= (1 << s)  # Immediate successor s
            mask |= successors_mask[s]  # All transitive successors of s
            
        down_path[i] = times[i] + max_path
        successors_mask[i] = mask
        
    # Now compute score
    # trans_count[i] = popcount(successors_mask[i])
    # numpy doesn't have popcount directly for int64 arrays in older versions, 
    # but we can loop or use bit_count if Python 3.10+. 
    # Let's use a simple loop or bin().count('1') for safety, N=30 is tiny.
    
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        # Count bits in successors_mask[i]
        # int.bit_count() is Python 3.10+. 
        # Fallback: bin(x).count('1')
        t_count = bin(successors_mask[i]).count('1')
        
        # Composite score
        # Emphasize downstream criticality (down_path) and branching (t_count)
        # Base is times[i]
        score = times[i] + down_path[i] + 0.5 * t_count
        scores[i] = score
        
    return scores
