"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import defaultdict, deque

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # safety margin before 120s limit


def _time_remaining():
    return max(0.0, _DEADLINE - time.time())


def _build_graph(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance.get("predecessors", [])
    n = len(times)
    # preds: list of lists, preds[i] = list of predecessor indices for task i
    # Build successor map and in-degree
    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            indeg[i] += 1
    return times, succ, indeg, preds


def _longest_path_to_end(times, succ, n):
    """Compute longest path from each node to any sink (reverse topological)."""
    # Use memoized DFS or topological sort
    # Since we have DAG, do reverse topological order
    # First get topological order
    indeg_tmp = np.zeros(n, dtype=int)
    for i in range(n):
        for s in succ[i]:
            indeg_tmp[s] += 1
    
    from collections import deque
    q = deque()
    for i in range(n):
        if indeg_tmp[i] == 0:
            q.append(i)
    
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg_tmp[v] -= 1
            if indeg_tmp[v] == 0:
                q.append(v)
    
    # longest_path[i] = max(times[j] + longest_path[j] for j in succ[i])
    longest_path = np.zeros(n, dtype=float)
    for i in reversed(topo):
        best = 0.0
        for s in succ[i]:
            if times[s] + longest_path[s] > best:
                best = times[s] + longest_path[s]
        longest_path[i] = times[i] + best
    return longest_path, topo


def _longest_path_from_start(times, preds, n):
    """Compute longest path from any source to each node."""
    indeg = np.zeros(n, dtype=int)
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            indeg[i] += 1
    
    from collections import deque
    q = deque()
    for i in range(n):
        if indeg[i] == 0:
            q.append(i)
    
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    
    longest_from_start = np.zeros(n, dtype=float)
    for i in topo:
        best = 0.0
        for p in preds[i]:
            if longest_from_start[p] + times[p] > best:
                best = longest_from_start[p] + times[p]
        longest_from_start[i] = best + times[i]
    return longest_from_start, topo


def _greedy_balance(instance, priority_scores):
    """
    Greedy assembly line balancing using the given priority scores.
    Returns number of stations.
    """
    times, succ, indeg, preds = _build_graph(instance)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Priority: higher score = higher priority
    # We'll use a max-heap (negate for min-heap)
    
    # Track which tasks are available (all predecessors done)
    # Initially, tasks with no predecessors are available
    # But we also need to track remaining successors count for each task
    # to know when a task is "done" (all its successors have been assigned)
    
    # Actually, for greedy BAL:
    # At each step, among all eligible tasks (all predecessors assigned to some station),
    # pick the one with highest priority that fits in current station.
    
    # Let's track:
    # - assigned: set of tasks already assigned
    # - pred_count[i]: number of unassigned predecessors of task i
    # - When pred_count[i] becomes 0, task i becomes eligible
    
    pred_count = np.array([len(p) for p in preds], dtype=int)
    eligible = []  # min-heap of (-priority, task_index)
    
    for i in range(n):
        if pred_count[i] == 0:
            eligible.append((-priority_scores[i], i))
    heapq.heapify(eligible)
    
    # For each task, track how many successors it has (to decrement when task is assigned)
    succ_count = np.array([len(s) for s in succ], dtype=int)
    
    # Station assignment
    station_load = 0.0
    num_stations = 1
    assigned_count = 0
    
    while assigned_count < n:
        # Try to fill current station
        placed_in_station = True
        while placed_in_station:
            # Find best eligible task that fits
            best_task = -1
            best_priority = -np.inf
            # We need to find the highest priority task that fits
            # Since heap gives us highest priority first, we can pop and check
            # But we need to put back those that don't fit
            candidates = []
            while eligible:
                neg_prio, task = heapq.heappop(eligible)
                if times[task] <= cycle_time - station_load:
                    # This task fits
                    if -neg_prio > best_priority:
                        best_priority = -neg_prio
                        best_task = task
                    candidates.append((neg_prio, task))
                else:
                    candidates.append((neg_prio, task))
            
            # Put all candidates back
            for item in candidates:
                heapq.heappush(eligible, item)
            
            if best_task == -1:
                # No eligible task fits in current station
                if station_load > 0:
                    # Move to next station
                    num_stations += 1
                    station_load = 0.0
                    # Continue trying to place in new station
                    placed_in_station = True
                    continue
                else:
                    # Station is empty and no task fits? Shouldn't happen if all tasks <= cycle_time
                    # But if a task is larger than cycle_time, it's infeasible
                    # For safety, break
                    break
            
            # Assign best_task to current station
            station_load += times[best_task]
            assigned_count += 1
            
            # Update successors
            for s in succ[best_task]:
                pred_count[s] -= 1
                if pred_count[s] == 0:
                    heapq.heappush(eligible, (-priority_scores[s], s))
            
            # Check if we can continue placing in this station
            # The while loop will continue and try to find another task
        # If we exited the inner while because no task fits and station was empty, break
        if station_load == 0 and assigned_count < n:
            # This means no task could be placed, which shouldn't happen
            # Break to avoid infinite loop
            break
    
    return num_stations


def _compute_priority(instance):
    """
    Compute priority scores for tasks.
    Uses a combination of:
    1. Longest path to end (critical path length)
    2. Number of successors (breadth)
    3. Task time
    """
    times, succ, indeg, preds = _build_graph(instance)
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Compute longest path to end
    lp_end, topo = _longest_path_to_end(times, succ, n)
    
    # Compute longest path from start
    lp_start, _ = _longest_path_from_start(times, preds, n)
    
    # Number of successors (direct)
    num_succ = np.array([len(s) for s in succ], dtype=float)
    
    # Number of predecessors (direct)
    num_pred = np.array([len(p) for p in preds], dtype=float)
    
    # Total work in subtree (all successors transitively)
    # Compute via topological order
    subtree_work = np.zeros(n, dtype=float)
    for i in reversed(topo):
        total = times[i]
        for s in succ[i]:
            total += subtree_work[s]
        subtree_work[i] = total
    
    # Combine priorities
    # Normalize each component
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-10:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_lp_end = normalize(lp_end)
    norm_lp_start = normalize(lp_start)
    norm_num_succ = normalize(num_succ)
    norm_subtree = normalize(subtree_work)
    norm_time = normalize(times)
    
    # Weighted combination
    priority = (
        0.4 * norm_lp_end +
        0.2 * norm_lp_start +
        0.15 * norm_num_succ +
        0.15 * norm_subtree +
        0.1 * norm_time
    )
    
    return priority


def priority(instance):
    """
    Score each assembly task. Higher score = higher priority.
    The greedy algorithm picks the highest-priority eligible task that fits.
    """
    if time.time() > _DEADLINE:
        # Return a simple fallback
        return np.asarray(instance["times"], dtype=float)
    
    try:
        return _compute_priority(instance)
    except Exception:
        return np.asarray(instance["times"], dtype=float)
# EVOLVE-BLOCK-END
