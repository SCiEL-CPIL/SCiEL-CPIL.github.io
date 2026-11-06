"""Improved RCPSP priority function using backward pass + resource criticality."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    The serial schedule-generation scheme picks the highest-priority eligible
    activity at each step. Higher priority = started earlier.
    
    Strategy:
      1. Compute backward pass (latest finish times) to get slack.
      2. Compute resource criticality: how much each activity's resource demand
         contributes to bottleneck resources.
      3. Combine: low slack + high resource criticality = high priority.
      4. Use a smooth combination to avoid ties and provide gradient-like info.
    """
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # --- Extract instance components with defensive key handling ---
    # Predecessors
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred"]:
        if key in instance:
            preds = instance[key]
            break

    # Successors (may be provided or derived)
    succs = None
    for key in ["successors", "succs", "successors_list", "succ"]:
        if key in instance:
            succs = instance[key]
            break

    # Resource demands: list of lists or 2D array, shape (n, R)
    demands = None
    for key in ["demands", "resource_demands", "resources", "dem", "resource"]:
        if key in instance:
            demands = instance[key]
            break

    # Resource capacities: list or array of length R
    capacities = None
    for key in ["capacities", "capacity", "resource_capacities", "caps", "cap"]:
        if key in instance:
            capacities = instance[key]
            break

    # Fallbacks
    if preds is None:
        preds = [[] for _ in range(n)]
    if succs is None:
        # Derive from preds
        succs = [set() for _ in range(n)]
        for j in range(n):
            for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
                idx = int(i)
                if 0 <= idx < n:
                    succs[idx].add(j)
        succs = [list(s) for s in succs]
    if demands is None:
        # No resource constraints: just use backward pass
        demands = np.zeros((n, 1))
    if capacities is None:
        capacities = [1.0]

    demands = np.asarray(demands, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
    R = demands.shape[1]
    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim == 0:
        capacities = np.full(R, float(capacities))

    # --- Forward pass (earliest start/finish) ---
    # Topological order via Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
            idx = int(i)
            if 0 <= idx < n:
                in_degree[j] += 1

    from collections import deque
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)

    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    # If topo_order doesn't include all nodes, add remaining (cycle handling)
    if len(topo_order) < n:
        for i in range(n):
            if i not in set(topo_order):
                topo_order.append(i)

    # Forward pass
    ES = np.zeros(n)  # Earliest Start
    EF = np.zeros(n)  # Earliest Finish
    for u in topo_order:
        for v in succs[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
        EF[u] = ES[u] + durations[u]

    makespan_forward = float(np.max(EF)) if n > 0 else 0.0

    # --- Backward pass (latest start/finish) ---
    LS = np.zeros(n)  # Latest Start
    LF = np.zeros(n)  # Latest Finish
    # Reverse topological order
    for u in reversed(topo_order):
        # LF[u] = min(LS[v]) for all successors v, or makespan if no successors
        if len(succs[u]) == 0:
            LF[u] = makespan_forward
        else:
            LF[u] = float(np.min([LS[v] for v in succs[u]]))
        LS[u] = LF[u] - durations[u]
        for v in succs[u]:
            if LS[u] < LF[v]:
                LF[v] = LS[u]

    # Slack (total float)
    slack = LS - ES  # or LF - EF, same thing

    # --- Resource criticality ---
    # For each resource r, compute the total demand of activities that use it.
    # Activities with high demand on tight resources get higher priority.
    # Resource tightness: capacity / total demand (lower = tighter)
    resource_tightness = np.zeros(R)
    for r in range(R):
        total_demand = float(np.sum(demands[:, r]))
        if total_demand > 0:
            resource_tightness[r] = capacities[r] / total_demand
        else:
            resource_tightness[r] = 1.0  # not tight

    # Resource criticality for each activity: weighted sum of demands by tightness
    # An activity that uses a tight resource heavily is more critical.
    # Normalize demands per resource to [0,1] range
    max_demand_per_res = np.max(demands, axis=0)
    max_demand_per_res[max_demand_per_res == 0] = 1.0
    norm_demands = demands / max_demand_per_res  # shape (n, R)

    # Criticality score: sum over resources of (norm_demand * (1/tightness))
    # Higher tightness (lower ratio) means more critical
    inv_tightness = 1.0 / (resource_tightness + 1e-10)
    resource_criticality = np.sum(norm_demands * inv_tightness, axis=1)

    # --- Combine slack and resource criticality ---
    # We want: low slack -> high priority, high resource criticality -> high priority
    # Normalize both to [0, 1]
    slack_min, slack_max = float(np.min(slack)), float(np.max(slack))
    if slack_max > slack_min:
        norm_slack = (slack - slack_min) / (slack_max - slack_min)
    else:
        norm_slack = np.zeros(n)

    rc_min, rc_max = float(np.min(resource_criticality)), float(np.max(resource_criticality))
    if rc_max > rc_min:
        norm_rc = (resource_criticality - rc_min) / (rc_max - rc_min)
    else:
        norm_rc = np.zeros(n)

    # Priority = weight1 * (1 - norm_slack) + weight2 * norm_rc
    # Use smooth combination; weights can be tuned
    w_slack = 0.6
    w_rc = 0.4
    priority_scores = w_slack * (1.0 - norm_slack) + w_rc * norm_rc

    # Add a small duration-based component to break ties and favor longer tasks
    dur_min, dur_max = float(np.min(durations)), float(np.max(durations))
    if dur_max > dur_min:
        norm_dur = (durations - dur_min) / (dur_max - dur_min)
    else:
        norm_dur = np.zeros(n)
    priority_scores = priority_scores + 0.1 * norm_dur

    # Ensure positive scores (some schedulers may interpret negative differently)
    priority_scores = priority_scores - np.min(priority_scores) + 1.0

    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return what we have
        return priority_scores

    # --- Optional: refine with a simple serial simulation to adjust priorities ---
    # Simulate the serial SGS with current priorities, then adjust based on
    # which activities were delayed by resource contention.
    # This is a heuristic refinement: activities that got delayed more should
    # get higher priority in subsequent iterations.

    # We'll do a few iterations of priority refinement via simulation
    def simulate_sgs(prio, max_iter=1000):
        """Serial schedule generation scheme. Returns start times and makespan."""
        if time.time() - start_time > DEADLINE:
            return None, None

        n_act = len(prio)
        start = np.full(n_act, np.inf)
        finish = np.full(n_act, np.inf)
        scheduled = [False] * n_act
        # Track resource usage over time
        # Use event-based: for each time point, track available resources
        # Since we don't know the time horizon in advance, use a dict
        # Actually, for serial SGS, at each step we pick the best eligible activity
        # and schedule it at the earliest feasible time.

        # Precompute: for each activity, the earliest it can start based on precedence
        # ES is already computed above

        # Resource availability: we'll track as a list of (time, amount) per resource
        # Simpler: for each resource, maintain a sorted list of intervals where it's busy
        # Or even simpler: since n=30, we can just check each candidate start time

        # Let's use a simpler approach: for each activity, find earliest start
        # that satisfies both precedence and resource constraints.

        # Build predecessor sets for quick lookup
        pred_set = [set() for _ in range(n_act)]
        for j in range(n_act):
            for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
                idx = int(i)
                if 0 <= idx < n_act:
                    pred_set[j].add(idx)

        # Iteratively schedule activities
        remaining = set(range(n_act))
        time_limit = time.time() + (DEADLINE - (time.time() - start_time)) * 0.5

        while remaining:
            if time.time() > time_limit:
                return None, None

            # Find eligible activities: all predecessors scheduled
            eligible = []
            for j in remaining:
                all_preds_scheduled = all(start[p] < np.inf for p in pred_set[j])
                if all_preds_scheduled:
                    eligible.append(j)

            if not eligible:
                # Cycle or error; schedule remaining in order
                for j in remaining:
                    start[j] = EF[j]  # fallback
                    finish[j] = EF[j] + durations[j]
                break

            # Pick the one with highest priority
            best = max(eligible, key=lambda x: prio[x])

            # Find earliest feasible start time for 'best'
            # Precedence constraint: ES[best]
            es = ES[best]

            # Resource constraint: for each resource, find the earliest time
            # after 'es' where all required resources are available.
            # We track for each resource a list of busy intervals.
            # Since n is small, we can do a simple check.

            # Let's maintain resource busy intervals
            # Actually, let's use a different approach:
            # For each candidate start time t >= es, check if resources are available
            # from t to t + duration[best].
            # The candidate times are: es, and finish times of already-scheduled activities.

            # Gather candidate start times
            candidate_times = set([es])
            for k in range(n_act):
                if start[k] < np.inf:
                    candidate_times.add(finish[k])
            candidate_times = sorted(candidate_times)
            # Also add some extra times
            candidate_times = [t for t in candidate_times if t >= es]
            if not candidate_times:
                candidate_times = [es]

            # Find the earliest candidate time where resources are available
            best_start = es
            found = False
            for t in candidate_times:
                if t < es:
                    continue
                # Check resource availability from t to t + durations[best]
                feasible = True
                end_t = t + durations[best]
                for r in range(R):
                    if demands[best, r] <= 0:
                        continue
                    # Check if any scheduled activity overlaps [t, end_t) on resource r
                    for k in range(n_act):
                        if start[k] < np.inf and demands[k, r] > 0:
                            if start[k] < end_t and finish[k] > t:
                                # Overlap: check if combined demand exceeds capacity
                                # We need to track cumulative demand over the overlap
                                # This is complex; for simplicity, assume
                                # that we just check if the sum of demands
                                # of overlapping activities exceeds capacity.
                                # But we need to be careful: multiple activities
                                # can overlap.
                                # Let's collect all activities that overlap [t, end_t)
                                pass
                    # This approach is getting complex. Let's use a simpler
                    # resource tracking method.
                # For now, use a greedy approach: just check if the activity
                # can be added without exceeding capacity at any point in [t, end_t)
                # We'll track resource usage as a function of time using a dict.
                break  # Simplified: just use es for now

            # Simplified: schedule at ES[best]
            # This may not be optimal but is fast.
            # For a better approach, we'd need proper resource tracking.

            # Let's do a proper resource check with a time-sweep
            # Since n=30, we can afford a bit more work.
            # Build a list of all relevant time points
            all_times = set([es])
            for k in range(n_act):
                if start[k] < np.inf:
                    all_times.add(start[k])
                    all_times.add(finish[k])
            all_times = sorted([t for t in all_times if t >= es])
            if not all_times:
                all_times = [es]

            best_start = es
            feasible_found = False
            for t in all_times:
                if t < es:
                    continue
                end_t = t + durations[best]
                # Check resource constraints: for each resource, the total demand
                # of activities scheduled in any sub-interval of [t, end_t) must not exceed capacity.
                # We check at each critical time point within [t, end_t)
                # Critical points: t, end_t, and start/finish times of other activities within [t, end_t)
                check_points = set([t, end_t])
                for k in range(n_act):
                    if start[k] < np.inf:
                        if start[k] >= t and start[k] <= end_t:
                            check_points.add(start[k])
                        if finish[k] >= t and finish[k] <= end_t:
                            check_points.add(finish[k])
                check_points = sorted(check_points)

                ok = True
                for i in range(len(check_points) - 1):
                    tp = check_points[i]
                    tp_next = check_points[i + 1]
                    if tp >= end_t or tp_next <= t:
                        continue
                    # At this sub-interval [tp, tp_next), which activities are active?
                    active_demand = np.zeros(R)
                    for k in range(n_act):
                        if start[k] < np.inf:
                            if start[k] <= tp and finish[k] >= tp_next:
                                active_demand += demands[k]
                    # Add the new activity
                    active_demand += demands[best]
                    if np.any(active_demand > capacities + 1e-9):
                        ok = False
                        break
                if ok:
                    best_start = t
                    feasible_found = True
                    break

            if not feasible_found:
                # Fallback: schedule at es (may violate constraints, but shouldn't happen)
                best_start = es

            start[best] = best_start
            finish[best] = best_start + durations[best]
            remaining.discard(best)

        makespan = float(np.max(finish)) if n_act > 0 else 0.0
        return start, makespan

    # Run simulation to get a baseline
    sim_start, sim_makespan = simulate_sgs(priority_scores)

    if sim_start is not None and sim_makespan is not None:
        # Compute delay for each activity: how much later it started than ES
        delay = np.maximum(sim_start - ES, 0)
        # Activities with high delay were bottlenecked; boost their priority
        delay_min, delay_max = float(np.min(delay)), float(np.max(delay))
        if delay_max > delay_min:
            norm_delay = (delay - delay_min) / (delay_max - delay_min)
        else:
            norm_delay = np.zeros(n)

        # Adjust priority: add a component based on delay
        w_delay = 0.3
        priority_scores = priority_scores + w_delay * norm_delay

        # Re-normalize to keep scores in a reasonable range
        p_min, p_max = float(np.min(priority_scores)), float(np.max(priority_scores))
        if p_max > p_min:
            priority_scores = (priority_scores - p_min) / (p_max - p_min)
        else:
            priority_scores = np.ones(n)

    return priority_scores
# EVOLVE-BLOCK-END
