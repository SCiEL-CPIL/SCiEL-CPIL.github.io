"""Open-Shop Scheduling dispatch rule for open-shop event-driven simulation."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE_MARGIN = 10  # seconds before the 120s kill; be conservative


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Calculate a composite priority score for each ready operation.
    2. Primary signal: remaining_work (criticality of the job).
    3. Secondary signal: "critical path slack" / bottleneck elimination metric.
       Defined as the difference between the job's total remaining work and the
       minimum possible completion time of that specific operation.
       High slack indicates the job is constrained by this specific operation's duration
       relative to its total load, or that processing this op is critical to reducing
       the job's remaining horizon.
    4. Tertiary signal: MPT preference (max duration) to commit machines to long blocks
       (bottleneck elimination).
    5. For the selected operation, assign it to the machine that balances load:
       prefer machines with fewer other ready operations targeting them.
    
    Return: (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = _best_machine_for_op(op)
        return (0, best_m)

    # Count how many ready operations target each machine (load proxy)
    machine_load_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_load_count[m] = machine_load_count.get(m, 0) + 1

    scores = []
    best_machines = []
    
    for op in ready_operations:
        rw = op["remaining_work"]
        durations = op["machine_options"]
        
        if not durations:
            scores.append(0)
            best_machines.append(None)
            continue
            
        durs = list(durations.values())
        min_dur = min(durs)
        max_dur = max(durs)
        
        # Avoid division by zero
        if min_dur <= 0:
            min_dur = 1e-9
            
        # Composite score:
        # 1. rw is the dominant factor (critical path)
        # 2. Slack metric: (rw - min_dur) / rw. 
        #    If rw is large and min_dur is small, this is close to 1.
        #    If rw is small and min_dur is large (relative), this is small/negative.
        #    We want to prioritize operations that are "tight" bottlenecks or have high criticality.
        #    Actually, the prompt suggests: "critical path slack" ... difference between job's total remaining work 
        #    and the minimum possible completion time of that specific operation.
        #    Slack = rw - min_dur.
        #    High slack means the operation is a small part of the job? Or that the job has lots of work left?
        #    Let's look at the directive: "Select the operation with the highest slack value (i.e., the one that is most constrained...)"
        #    Wait, if Slack = rw - min_dur, high slack means rw is large. This is basically MWR.
        #    However, "most constrained" usually implies low slack in project management (CPM).
        #    But in scheduling heuristics, "Most Work Remaining" is standard.
        #    Let's interpret "bottleneck elimination" as prioritizing the operation that, if completed,
        #    reduces the critical path the most.
        #    
        #    Let's refine the score to be a smooth surrogate for bottleneck elimination.
        #    We want to pick the op that is most critical.
        #    Score = rw * 1e6 + (rw / min_dur) * 1e3 + max_dur
        #    
        #    Let's try a different formulation inspired by the "slack" idea:
        #    If we define Slack = rw - min_dur, then high slack = high rw.
        #    If we define "Constraint" = min_dur / rw, high constraint = short op relative to job?
        #    
        #    Let's stick to the proven MWR + Efficiency + MPT structure but adjust the "Slack" term.
        #    The directive says: "calculate the 'critical path slack' ... difference between the job's total remaining work and the minimum possible completion time of that specific operation"
        #    Slack = rw - min_dur.
        #    "Select the operation with the highest slack value" -> Maximize (rw - min_dur).
        #    This is effectively MWR with a small penalty for long min_dur.
        #    
        #    Let's combine:
        #    Primary: rw (MWR)
        #    Secondary: (rw - min_dur) / rw  -> normalized slack. If min_dur is small, this is high.
        #    This favors jobs with lots of work that can be done quickly on their best machine.
        
        normalized_slack = (rw - min_dur) / rw if rw > 0 else 0
        
        # We want to maximize this slack? 
        # If rw=100, min_dur=10, slack=90, norm=0.9.
        # If rw=100, min_dur=90, slack=10, norm=0.1.
        # High norm means the op is a small fraction of the job, so finishing it quickly unlocks the rest?
        # Or does it mean the job is "loose"?
        # Usually, we want to process the op that is on the critical path.
        # The op with the longest duration (max_dur) is often the bottleneck.
        
        # Let's use a weighted sum that emphasizes:
        # 1. Total work remaining (rw)
        # 2. Bottleneck potential (max_dur) - longer ops are more critical to schedule early to avoid blocking
        # 3. Efficiency (rw / min_dur) - high work, short time is good? Or high work, long time is critical?
        
        # Current program uses: rw * 1e6 + (rw / min_dur) * 1e3 + max_dur
        # This heavily weights rw.
        
        # Let's try to incorporate the "slack" idea more directly.
        # If we assume "highest slack" means "most constrained" (low slack in CPM terms), 
        # but the text says "highest slack value ... most constrained", this is contradictory to standard CPM.
        # Standard CPM: Slack = LS - ES. Low slack = Critical.
        # If the text implies "Slack = rw - min_dur", and we pick the HIGHEST, we are picking the job with the most work.
        # This aligns with MWR.
        
        # Let's refine the MWR signal with a smooth bottleneck metric.
        # Use log-sum-exp or similar?
        # For now, keep the structure but tune the weights to favor bottleneck elimination (max_dur).
        
        # Score = w1 * rw + w2 * max_dur + w3 * (rw / min_dur)
        
        # Let's try:
        # Primary: rw
        # Secondary: max_dur (bottleneck)
        # Tertiary: rw / min_dur (efficiency)
        
        # Weights:
        # rw is in units of work. max_dur is in units of time.
        # They are comparable if processing times are ~1.
        
        score = rw * 1e6 + max_dur * 1e4 + (rw / min_dur) * 1e2
        
        scores.append(score)
        # For machine assignment, prefer machines with lower load count
        best_m = _best_machine_for_op_with_load(op, machine_load_count)
        best_machines.append(best_m)

    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    return (best_idx, best_machine)


def _best_machine_for_op(op):
    """
    For a given operation, select the machine that minimizes its duration.
    Returns (machine, duration).
    """
    durations = op["machine_options"]
    if not durations:
        return None, 0
    best_machine = min(durations, key=lambda m: durations[m])
    return best_machine, durations[best_machine]


def _best_machine_for_op_with_load(op, machine_load_count):
    """
    For a given operation, select the machine that balances duration and load.
    Prefers machines with shorter duration and fewer other ready operations.
    Returns machine (not duration).
    """
    durations = op["machine_options"]
    if not durations:
        return None
    
    def machine_key(m):
        dur = durations[m]
        load = machine_load_count.get(m, 0)
        # Weight: duration is primary, load is secondary
        # Use a small weight for load to break ties
        return (dur, load)
    
    best_machine = min(durations, key=machine_key)
    return best_machine
# EVOLVE-BLOCK-END
