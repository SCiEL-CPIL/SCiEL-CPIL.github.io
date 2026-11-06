"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def priority(instance):
    start_wall = time.time()
    deadline = 120.0
    safety_margin = 5.0

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance.get("predecessors", [])
    cycle_time = instance.get("cycle_time", 40.0)
    n = len(times)

    # Build successor list
    successors = [[] for _ in range(n)]
    pred_list = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
            pred_list[i].append(p)

    # Topological order
    in_degree = [len(pred_list[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                queue.append(v)

    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    # Transitive closure
    trans_succ = [set() for _ in range(n)]
    trans_pred = [set() for _ in range(n)]

    for u in reversed(topo_order):
        for v in successors[u]:
            trans_succ[u].add(v)
            trans_succ[u].update(trans_succ[v])

    for u in topo_order:
        for p in pred_list[u]:
            trans_pred[u].add(p)
            trans_pred[u].update(trans_pred[p])

    # Compute metrics
    pred_time_sum = np.zeros(n)
    succ_time_sum = np.zeros(n)
    succ_count = np.zeros(n)
    pred_count = np.zeros(n)

    for i in range(n):
        pred_time_sum[i] = sum(times[p] for p in trans_pred[i])
        succ_time_sum[i] = sum(times[s] for s in trans_succ[i])
        succ_count[i] = len(trans_succ[i])
        pred_count[i] = len(trans_pred[i])

    # Compute earliest start time (longest path from sources) for each task
    earliest_start = np.zeros(n)
    for u in topo_order:
        if pred_list[u]:
            earliest_start[u] = max(earliest_start[p] + times[p] for p in pred_list[u])
        else:
            earliest_start[u] = 0.0

    # Compute longest path duration from task to any sink
    # longest_path_downstream[i] = times[i] + max over successors s of (longest_path_downstream[s])
    # This represents the minimum remaining time after task i starts
    longest_path_downstream = np.zeros(n)
    for u in reversed(topo_order):
        if successors[u]:
            longest_path_downstream[u] = times[u] + max(longest_path_downstream[s] for s in successors[u])
        else:
            longest_path_downstream[u] = times[u]

    # Downstream slack deficit: cycle_time - longest_path_downstream[i]
    # Low deficit means the task's downstream chain nearly fills the cycle time,
    # so it has little temporal buffer and should be prioritized
    ct = max(cycle_time, 1.0)
    downstream_slack_deficit = ct - longest_path_downstream

    # Also compute total makespan lower bound for slack calculation
    latest_finish = longest_path_downstream  # same as longest path from task to sinks
    makespan_lb = np.max(latest_finish) if n > 0 else 1.0
    slack = makespan_lb - earliest_start - times

    # Normalize all metrics
    max_succ_time = np.max(succ_time_sum) if np.max(succ_time_sum) > 0 else 1.0
    max_pred_time = np.max(pred_time_sum) if np.max(pred_time_sum) > 0 else 1.0
    max_succ_count = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    max_pred_count = np.max(pred_count) if np.max(pred_count) > 0 else 1.0
    max_t = np.max(times) if np.max(times) > 0 else 1.0
    max_deficit = np.max(downstream_slack_deficit) if np.max(downstream_slack_deficit) > 0 else 1.0
    min_deficit = np.min(downstream_slack_deficit) if n > 0 else 0.0
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0

    # For downstream_slack_deficit, we want LOW deficit to have HIGH priority
    # Normalize so that min_deficit maps to 1.0 and max_deficit maps to 0.0
    if max_deficit > min_deficit:
        normalized_deficit = (max_deficit - downstream_slack_deficit) / (max_deficit - min_deficit)
    else:
        normalized_deficit = np.ones(n)

    # For slack, we want LOW slack to have HIGH priority, so we invert
    inverted_slack = (max_slack - slack) / max_slack if max_slack > 0 else np.zeros(n)

    # Weights: emphasize downstream slack deficit (new critical metric) + successor-based metrics
    w_deficit = 3.0
    w_succ_time = 1.5
    w_pred_time = 0.5
    w_succ_count = 1.0
    w_pred_count = 0.3
    w_time = 0.4
    w_low_slack = 0.8

    scores = (w_deficit * normalized_deficit
              + w_succ_time * (succ_time_sum / max_succ_time)
              + w_pred_time * (pred_time_sum / max_pred_time)
              + w_succ_count * (succ_count / max_succ_count)
              + w_pred_count * (pred_count / max_pred_count)
              + w_time * (times / max_t)
              + w_low_slack * inverted_slack)

    # Deadline guard
    if time.time() > deadline - safety_margin:
        pass  # We're already done computing, just return

    return scores
# EVOLVE-BLOCK-END
