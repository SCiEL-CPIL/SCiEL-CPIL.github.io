"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: kill after 145s to stay within 150s limit
_DEADLINE = 145.0
_start_time = time.time()


def _topological_order(predecessors, n):
    """Return a topological ordering of activities."""
    in_degree = [0] * n
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If not all nodes are in order (cycle), append remaining
    if len(order) < n:
        remaining = set(range(n)) - set(order)
        order.extend(sorted(remaining))
    
    return order


def _forward_pass(durations, predecessors, order, n):
    """Compute earliest start (ES) and earliest finish (EF)."""
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in order:
        if predecessors[u]:
            es[u] = max(ef[p] for p in predecessors[u])
        ef[u] = es[u] + durations[u]
    return es, ef


def _backward_pass(durations, predecessors, successors, order, project_end, n):
    """Compute latest start (LS) and latest finish (LF)."""
    ls = np.zeros(n)
    lf = np.zeros(n)
    # Process in reverse topological order
    for u in reversed(order):
        if successors[u]:
            lf[u] = min(ls[s] for s in successors[u])
        else:
            lf[u] = project_end
        ls[u] = lf[u] - durations[u]
    return ls, lf


def priority(instance):
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return durations as priority
        return np.asarray(instance["durations"], dtype=float)
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (n, n_res)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (n_res,)
    predecessors = instance["predecessors"]  # list of lists
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Get topological order
    order = _topological_order(predecessors, n)
    
    # Forward pass
    es, ef = _forward_pass(durations, predecessors, order, n)
    project_end = np.max(ef)
    
    # Backward pass
    ls, lf = _backward_pass(durations, predecessors, successors, order, project_end, n)
    
    # Slack
    slack = ls - es
    
    # Critical Path Method: activities with zero slack are on critical path
    # Priority: lower slack = higher priority (more urgent)
    # Also consider resource demand: activities that use more of scarce resources should be prioritized
    
    # Resource scarcity measure: for each resource, how tight is it?
    # Compute total demand vs capacity ratio
    total_demand = np.sum(demands, axis=0)  # (n_res,)
    scarcity = total_demand / (capacities + 1e-10)  # higher = more scarce
    
    # For each activity, compute how much it contributes to resource scarcity
    # weighted scarcity contribution
    scarcity_contribution = np.sum(demands * scarcity, axis=1)  # (n,)
    
    # Combine: 
    # 1. Low slack (high urgency) - primary
    # 2. High scarcity contribution - secondary
    # 3. Long duration - tertiary (long tasks block resources longer)
    
    # Normalize components
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0
    norm_slack = slack / max_slack  # 0 = critical, 1 = max slack
    
    max_scarc = np.max(scarcity_contribution) if np.max(scarcity_contribution) > 0 else 1.0
    norm_scarc = scarcity_contribution / max_scarc
    
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    norm_dur = durations / max_dur
    
    # Priority score: we want HIGH priority for activities that should be scheduled first
    # In the serial SGS, the highest-priority eligible activity is started first
    # So we want high scores for: low slack, high scarcity contribution, long duration
    
    # Weighted combination
    w_slack = 0.6
    w_scarc = 0.25
    w_dur = 0.15
    
    # Invert slack: low slack -> high priority
    priority_score = w_slack * (1.0 - norm_slack) + w_scarc * norm_scarc + w_dur * norm_dur
    
    # Add small random tie-breaker to avoid deterministic behavior issues
    # But keep it small so it doesn't override the main signal
    rng = np.random.RandomState(42)
    priority_score += rng.uniform(0, 0.01, size=n)
    
    return priority_score
# EVOLVE-BLOCK-END
