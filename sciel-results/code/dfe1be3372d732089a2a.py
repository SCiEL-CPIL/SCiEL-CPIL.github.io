"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Score each assembly task for priority-based station assignment.
    Higher score = higher priority.
    
    Strategy:
    1. Compute transitive successor count (S_i): Tasks that unlock more work are prioritized.
    2. Compute longest downstream path (L_i): Tasks on critical paths are prioritized.
    3. Compute task time (T_i): Larger tasks are slightly prioritized to reduce fragmentation.
    4. Combine with weights favoring S_i heavily, as it is the strongest ALB heuristic.
    5. Apply a non-linear boost to high S_i tasks to sharpen bottleneck identification.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    try:
        times = np.asarray(instance["times"], dtype=float)
        predecessors = instance["predecessors"]
        cycle_time = instance["cycle_time"]
        n = len(times)
        
        if n == 0:
            return np.array([])
            
        # Build graph structures
        # successors[i] = list of tasks that depend on i
        successors = [[] for _ in range(n)]
        in_degree = [0] * n
        
        for i in range(n):
            for p in predecessors[i]:
                successors[p].append(i)
                in_degree[i] += 1
        
        # Topological sort (Kahn's algorithm) to process nodes in order
        # We need reverse topological order for downstream calculations
        q = deque()
        for i in range(n):
            if in_degree[i] == 0:
                q.append(i)
                
        topo_order = []
        temp_in_degree = in_degree[:]
        while q:
            u = q.popleft()
            topo_order.append(u)
            for v in successors[u]:
                temp_in_degree[v] -= 1
                if temp_in_degree[v] == 0:
                    q.append(v)
            
        # If there's a cycle, topo_order will be incomplete. 
        # The problem implies a valid DAG. If not, we fallback to simple time.
        if len(topo_order) != n:
            # Fallback: simple time-based priority
            return times / (np.max(times) + 1e-9)

        # 1. Compute Longest Downstream Path (including own time)
        # Process in reverse topological order
        longest_down = np.zeros(n)
        for u in reversed(topo_order):
            max_path = 0.0
            for v in successors[u]:
                if longest_down[v] > max_path:
                    max_path = longest_down[v]
            longest_down[u] = times[u] + max_path
            
        # 2. Compute Transitive Successor Count
        # count_desc[u] = number of all descendants of u
        count_desc = np.zeros(n)
        for u in reversed(topo_order):
            total = 0
            for v in successors[u]:
                total += 1 + count_desc[v]
            count_desc[u] = total

        # Normalization
        max_time = np.max(times)
        max_ld = np.max(longest_down)
        max_sc = np.max(count_desc)
        
        if max_time > 0:
            time_norm = times / max_time
        else:
            time_norm = np.zeros_like(times)
            
        if max_ld > 0:
            down_norm = longest_down / max_ld
        else:
            down_norm = np.zeros_like(longest_down)
            
        if max_sc > 0:
            succ_norm = count_desc / max_sc
        else:
            succ_norm = np.zeros_like(count_desc)

        # Composite Score
        # Emphasize Successor Count (Rank) heavily, as it indicates how much work is blocked.
        # Longest Path is secondary. Time is tertiary.
        # Weights: 0.1 Time, 0.2 Downstream Path, 0.7 Successor Count
        priority_scores = 0.1 * time_norm + 0.2 * down_norm + 0.7 * succ_norm
        
        # Non-linear boost for bottlenecks
        # Tasks with high successor counts get a multiplicative boost
        # This creates a sharper distinction between critical and non-critical tasks
        boost = 1.0 + 0.3 * succ_norm
        priority_scores = priority_scores * boost
        
        # Normalize final scores to [0, 1] for consistency, though relative order is what matters
        max_p = np.max(priority_scores)
        if max_p > 0:
            priority_scores = priority_scores / max_p
            
        return priority_scores
        
    except Exception:
        # Fallback to simple time-based priority if any error occurs
        times = np.asarray(instance["times"], dtype=float)
        if len(times) > 0:
            return times / (np.max(times) + 1e-9)
        return np.array([])
# EVOLVE-BLOCK-END
