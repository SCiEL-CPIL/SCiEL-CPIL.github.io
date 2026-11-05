import numpy as np
from collections import deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    
    # Build successor list
    # successors[i] = list of tasks that i precedes
    successors = [[] for _ in range(n)]
    in_degree = [0] * n # in_degree in the reverse graph (out_degree in original)
    
    for i in range(n):
        for pred in predecessors[i]:
            # pred precedes i
            # So i is a successor of pred
            successors[pred].append(i)
            # In the reverse graph (edges succ -> pred), 
            # wait, let's just use the original graph properties.
            # We want to compute dist[i] = times[i] + max(dist[s] for s in successors[i])
            # This requires dist[s] to be known.
            # So we need to process i after all its successors s are processed.
            # This is a reverse topological order of the original graph.
            
            # Let's count out-degrees of the original graph to manage the queue.
            # Actually, let's just count how many successors each node has? No.
            # Let's count in-degrees of the REVERSE graph.
            # Reverse graph edge: s -> i (if i is successor of s? No, if s is successor of i, edge i->s in orig, s->i in rev).
            # We want to process i when all s in successors[i] are done.
            # This is equivalent to processing nodes in reverse topological order of the original graph.
            
            # Let's just use a simple O(N^2) DP or Topological Sort.
            pass
            
    # Let's do Topological Sort on the Original Graph to get an order, then reverse it.
    # Or simpler: Calculate out-degree (number of successors) for each node.
    # Queue nodes with 0 successors (sinks).
    # When we process a node, we decrement the "successors remaining" count of its predecessors?
    # No, that's for in-degrees.
    
    # Let's stick to the standard:
    # 1. Build successors list.
    # 2. Compute "successor_count" for each node (out-degree in original graph).
    # 3. Queue all nodes with successor_count == 0.
    # 4. While queue not empty:
    #      Pop node u.
    #      For each predecessor p of u:
    #          decrement successor_count[p]
    #          if successor_count[p] == 0:
    #              queue p
    #      This processes nodes in reverse topological order (sinks first).
    
    # We need predecessors list again.
    # predecessors[i] is given.
    
    # Re-initialize
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            
    # Count successors for each node
    succ_count = [len(successors[i]) for i in range(n)]
    
    # Queue for nodes with no successors
    q = deque()
    for i in range(n):
        if succ_count[i] == 0:
            q.append(i)
            
    # dist[i] will be the longest path from i to a sink
    dist = np.zeros(n)
    processed = 0
    
    # We need to update dist[pred] when we process succ.
    # dist[pred] = times[pred] + max(dist[succ] for succ in successors[pred])
    # We can maintain a running max or just calculate when all successors are done.
    # Since we only push `pred` to queue when ALL its successors are done (succ_count becomes 0),
    # we can calculate dist[pred] at that moment.
    
    while q:
        u = q.popleft()
        # u's successors are all processed.
        # Calculate dist[u]
        if succ_count[u] == 0 and not hasattr(u, 'dist_calc'): # Wait, succ_count is 0 initially for sinks.
             # For sinks, dist[u] = times[u]
             # For others, we calculate when they enter the queue.
             pass
        
        # Actually, let's just calculate dist[u] right here.
        # If u has successors, they were processed before u entered the queue.
        # But we didn't store the max.
        
        # Let's refine:
        # When we process u (pop from queue), all its successors have been processed.
        # We need the max dist of its successors.
        # We can store a temporary `max_succ_dist` array.
        
    # Let's restart the logic cleanly.
    
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            
    succ_count = [len(successors[i]) for i in range(n)]
    q = deque()
    for i in range(n):
        if succ_count[i] == 0:
            q.append(i)
            
    dist = np.zeros(n)
    max_succ_dist = np.zeros(n) # Max dist among processed successors
    
    while q:
        u = q.popleft()
        # All successors of u are processed.
        if succ_count[u] > 0:
             # Wait, succ_count[u] was decremented as successors were processed.
             # It is 0 now.
             # So we can compute dist[u]
             dist[u] = times[u] + max_succ_dist[u]
        else:
             # It was a sink initially
             dist[u] = times[u]
             
        # Now, update predecessors of u
        for p in range(n):
            if u in successors[p]: # This is O(N^2) check, inefficient but N=30.
                # Better: store predecessors list
                pass
                
    # Let's use the predecessors list directly for the reverse step.
    # We have `predecessors` list from input.
    
    # Corrected Logic:
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            
    succ_count = [len(successors[i]) for i in range(n)]
    q = deque()
    for i in range(n):
        if succ_count[i] == 0:
            q.append(i)
            
    dist = np.zeros(n)
    max_succ_dist = np.zeros(n)
    
    while q:
        u = q.popleft()
        # Calculate dist for u
        if max_succ_dist[u] > 0: # Check if it had successors? No, max_succ_dist is 0 if no successors or all 0 times.
            # Actually, if succ_count[u] was > 0 initially, it had successors.
            # But we lost that info if we don't track it.
            # Let's just use: dist[u] = times[u] + max_succ_dist[u]
            # For sinks, max_succ_dist is 0, so dist = times. Correct.
            # For others, max_succ_dist holds the max. Correct.
            pass
        dist[u] = times[u] + max_succ_dist[u]
        
        # Propagate to predecessors
        for p in predecessors[u]: # p is a predecessor of u
            # u is a successor of p
            # Update max_succ_dist for p
            if dist[u] > max_succ_dist[p]:
                max_succ_dist[p] = dist[u]
            # Decrement remaining successor count for p
            succ_count[p] -= 1
            if succ_count[p] == 0:
                q.append(p)
                
    return dist
