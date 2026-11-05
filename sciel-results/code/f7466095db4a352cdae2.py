"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    """
    Computes a priority score for each task based on the Longest Path (Height)
    in the precedence graph. Tasks with longer paths to the end of the line
    are prioritized, which tends to reduce the number of stations compared to
    simple Longest Processing Time (LPT).
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    # predecessors[i] is a list of indices j such that j must precede i
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Compute height (longest path to sink) using topological sort
    # in_degree[i] is the number of predecessors of i
    in_degree = [len(preds) for preds in predecessors]
    
    # Queue for topological sort (Kahn's algorithm)
    # We want to process nodes in reverse topological order to compute heights
    # Or simply compute out_degree and process sinks first.
    
    # Let's compute out_degree
    out_degree = [len(succs) for succs in successors]
    
    # Initialize heights
    heights = np.zeros(n, dtype=float)
    
    # Queue of nodes with no successors (sinks)
    from collections import deque
    queue = deque()
    for i in range(n):
        if out_degree[i] == 0:
            queue.append(i)
            heights[i] = times[i]
            
    # Process nodes in reverse topological order
    # We need to decrement out_degree of predecessors as we process a node
    # If a node's out_degree becomes 0, it means all its successors have been processed
    # Then we can compute its height
    
    processed = 0
    while queue:
        u = queue.popleft()
        processed += 1
        # For each predecessor of u, we have found a successor
        # We need to know which predecessors exist.
        # We have the 'predecessors' list.
        # Let's iterate through all nodes to find predecessors of u? 
        # That is O(N^2). For N=30, it's fine.
        # But we can be smarter.
        
        # Actually, standard Kahn's works on in-degrees.
        # For heights, we want to propagate from sinks to sources.
        # So we use out-degrees.
        
        # Let's re-verify the logic.
        # If u is processed, its height is final.
        # For every predecessor p of u:
        #   heights[p] = max(heights[p], times[p] + heights[u])
        #   out_degree[p] -= 1
        #   if out_degree[p] == 0: queue.append(p)
        
        # To find predecessors of u efficiently, we can pre-build a list of predecessors
        # We already have 'predecessors' list which is indexed by node.
        # But we need to iterate predecessors of u.
        # Let's build a reverse adjacency list if not already implicit.
        # 'predecessors' is given as a list of lists.
        # predecessors[i] contains j where j -> i.
        # So to find predecessors of u, we just look at predecessors[u].
        
        for p in predecessors[u]:
            # p is a predecessor of u
            # Update height of p
            candidate_height = times[p] + heights[u]
            if candidate_height > heights[p]:
                heights[p] = candidate_height
            
            out_degree[p] -= 1
            if out_degree[p] == 0:
                queue.append(p)
                
    # If the graph has cycles, some nodes won't be processed. 
    # Assembly line balancing graphs are DAGs, so this should be fine.
    # If processed < n, we might have a cycle. In that case, heights for unprocessed nodes are 0.
    # We should handle this by falling back to times or similar.
    if processed < n:
         # Fallback: use times for unprocessed nodes
         for i in range(n):
             if heights[i] == 0 and times[i] > 0:
                 heights[i] = times[i]
                 
    return heights
# EVOLVE-BLOCK-END
