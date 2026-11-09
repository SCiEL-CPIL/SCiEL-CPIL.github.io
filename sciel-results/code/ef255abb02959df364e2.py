"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_successors_and_rank(instance):
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance.get("predecessors", None)
    cycle_time = float(instance.get("cycle_time", 40.0))

    # Build successor lists and in-degrees
    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    if preds is not None:
        for i in range(n):
            plist = preds[i]
            if plist is None:
                continue
            for p in plist:
                succ[p].append(i)
                indeg[i] += 1

    # Compute longest path from each node to any sink (backward pass)
    # Use topological order (Kahn)
    from collections import deque
    q = deque()
    for i in range(n):
        if indeg[i] == 0:
            q.append(i)

    topo = []
    indeg_work = indeg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                q.append(v)

    # If graph has cycles or is disconnected in a way that topo doesn't cover all,
    # append remaining nodes in original order
    if len(topo) < n:
        covered = set(topo)
        for i in range(n):
            if i not in covered:
                topo.append(i)

    # Backward pass: longest path from node i to any sink
    # lp[i] = times[i] + max(lp[j] for j in succ[i])
    lp = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for i in reversed(topo):
        max_succ = 0.0
        for j in succ[i]:
            if lp[j] > max_succ:
                max_succ = lp[j]
        lp[i] = times[i] + max_succ

    return times, succ, lp, cycle_time


def _simulate_priority_order(instance, priority_order):
    """Simulate the greedy station-filling algorithm.
    Returns number of stations used."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance.get("predecessors", None)
    cycle_time = float(instance.get("cycle_time", 40.0))

    # Build successor lists
    succ = [[] for _ in range(n)]
    if preds is not None:
        for i in range(n):
            plist = preds[i]
            if plist is None:
                continue
            for p in plist:
                succ[p].append(i)

    # Count remaining predecessors for each task
    rem_pred = np.zeros(n, dtype=int)
    if preds is not None:
        for i in range(n):
            plist = preds[i]
            if plist is not None:
                rem_pred[i] = len(plist)

    # Priority: higher value = higher priority
    # We need to pick the highest-priority eligible task that fits
    # Use a sorted list or heap. Since n is small (30), we can just scan.
    
    assigned = [False] * n
    station_load = 0.0
    num_stations = 1

    # Create a list of task indices sorted by priority (descending)
    # But we need dynamic eligibility, so we'll scan each time
    
    remaining = list(range(n))
    
    while remaining:
        # Find eligible tasks (all predecessors assigned)
        eligible = []
        for i in remaining:
            if rem_pred[i] == 0:
                eligible.append(i)
        
        if not eligible:
            # Should not happen if graph is valid
            break
        
        # Among eligible, pick the one with highest priority that fits
        # Sort eligible by priority descending
        eligible.sort(key=lambda i: priority_order[i], reverse=True)
        
        placed = False
        for i in eligible:
            if station_load + times[i] <= cycle_time:
                # Place this task
                assigned[i] = True
                station_load += times[i]
                remaining.remove(i)
                # Update rem_pred for successors
                for j in succ[i]:
                    rem_pred[j] -= 1
                placed = True
                break
        
        if not placed:
            # No eligible task fits, start new station
            num_stations += 1
            station_load = 0.0
            # If still can't place, something is wrong
            if num_stations > n + 1:
                break
            continue
    
    return num_stations


def priority(instance):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    times, succ, lp, cycle_time = _compute_successors_and_rank(instance)
    n = len(times)
    preds = instance.get("predecessors", None)
    
    # Base priority: longest path (backward rank)
    base_priority = lp.copy()
    
    # Add a small tie-breaker: longer tasks slightly preferred
    # priority = lp + alpha * times
    alpha = 0.1
    priority_order = base_priority + alpha * times
    
    # Simulate to get baseline
    baseline_stations = _simulate_priority_order(instance, priority_order)
    
    # Try a few variations if time permits
    best_priority = priority_order.copy()
    best_stations = baseline_stations
    
    # Try different alpha values
    alphas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    for a in alphas:
        if time.time() > deadline:
            break
        po = base_priority + a * times
        s = _simulate_priority_order(instance, po)
        if s < best_stations:
            best_stations = s
            best_priority = po
    
    # Try: priority = times only
    if time.time() < deadline:
        po = times.copy()
        s = _simulate_priority_order(instance, po)
        if s < best_stations:
            best_stations = s
            best_priority = po
    
    # Try: priority = lp only (no tie-breaker)
    if time.time() < deadline:
        po = base_priority.copy()
        s = _simulate_priority_order(instance, po)
        if s < best_stations:
            best_stations = s
            best_priority = po
    
    # Try: priority = lp * times (product)
    if time.time() < deadline:
        po = base_priority * times
        s = _simulate_priority_order(instance, po)
        if s < best_stations:
            best_stations = s
            best_priority = po
    
    # Try: priority = lp^2
    if time.time() < deadline:
        po = base_priority ** 2
        s = _simulate_priority_order(instance, po)
        if s < best_stations:
            best_stations = s
            best_priority = po
    
    # Try: priority = lp + times^2
    if time.time() < deadline:
        po = base_priority + times ** 2
        s = _simulate_priority_order(instance, po)
        if s < best_stations:
            best_stations = s
            best_priority = po
    
    # Try: priority = (lp + times) / (1 + times/cycle_time)
    if time.time() < deadline:
        po = (base_priority + times) / (1 + times / cycle_time)
        s = _simulate_priority_order(instance, po)
        if s < best_stations:
            best_stations = s
            best_priority = po
    
    # Try: priority = lp * (1 + 0.1 * times)
    if time.time() < deadline:
        po = base_priority * (1 + 0.1 * times)
        s = _simulate_priority_order(instance, po)
        if s < best_stations:
            best_stations = s
            best_priority = po
    
    return best_priority
# EVOLVE-BLOCK-END
