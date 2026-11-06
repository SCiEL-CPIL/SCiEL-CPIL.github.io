"""Job-Shop Scheduling: Critical Path SPT Dispatch Rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s wall-clock.
# Keep a safety margin; if we exceed it, fall back to a fast heuristic.
_DEADLINE = 145.0
_START_TIME = _time.monotonic()


def _check_deadline():
    return _time.monotonic() - _START_TIME > _DEADLINE


def _fast_dispatch(ready_operations, time):
    """Fast fallback: shortest processing time first (SPT) on min machine option."""
    if not ready_operations:
        return 0
    durations = []
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            durations.append(min(mo.values()))
        else:
            durations.append(0.0)
    best_idx = int(np.argmin(durations))
    # Return index only for fast path to avoid tuple overhead if simulator is strict, 
    # but the prompt says optionally (index, machine). 
    # The current program returns index. 
    # If I return (index, machine), I need to ensure the simulator handles it.
    # The prompt says "optionally (index, machine)".
    # I will return (index, machine) in the main dispatch to be optimal.
    # For fast dispatch, I'll just return index to be safe and fast.
    return best_idx


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    if _check_deadline():
        return _fast_dispatch(ready_operations, time)

    n = len(ready_operations)
    
    # 1. Calculate the current global makespan lower bound (max remaining work)
    #    and track which jobs contribute to this bound.
    current_max_work = 0.0
    for op in ready_operations:
        rw = op.get("remaining_work", 0.0)
        if rw > current_max_work:
            current_max_work = rw
            
    # If no remaining work is defined (or all zero), fall back to fast dispatch
    if current_max_work <= 0:
        return _fast_dispatch(ready_operations, time)

    # 2. Look-ahead: For each ready operation, simulate its completion.
    #    Calculate the new makespan lower bound if this op were started (and completed) now.
    #    The "new" remaining work for the job of this op is:
    #    current_remaining_work - duration_of_this_op_on_chosen_machine.
    #    We want to minimize the resulting max remaining work across all jobs.
    #    If multiple ops yield the same new max, we prefer the one that minimizes
    #    the secondary metric: the sum of remaining works of jobs that are *still*
    #    at the max, or simply the remaining work of the job itself if it's still critical.
    
    best_idx = -1
    best_machine = None
    best_new_max_work = float('inf')
    best_tie_breaker = float('inf')
    
    # Pre-compute remaining work per job to allow efficient update?
    # Since n is small (ready ops), we can just iterate.
    # However, we need to know the remaining work of *other* jobs to determine the new max.
    # The new max will be max(
    #     max(remaining_work[j] for j != job_i),
    #     remaining_work[i] - duration_i
    # )
    
    # To do this efficiently, let's group by job.
    job_remaining_work = {}
    for op in ready_operations:
        job_id = op["job"]
        rw = op.get("remaining_work", 0.0)
        if job_id not in job_remaining_work or rw > job_remaining_work[job_id]:
            job_remaining_work[job_id] = rw
            
    # Get the top two max remaining works to handle the case where the critical job is reduced below the second-highest.
    # Actually, simpler: 
    # new_max = max( max(remaining_work[j] for j != job_i), remaining_work[i] - duration )
    
    # Since we don't have the full list of all jobs (only ready ones), but "remaining_work" in the dict 
    # usually reflects the total remaining work for that job in the entire schedule, 
    # the max over *ready* operations might not be the true global max if non-ready jobs have more work.
    # HOWEVER, the problem states `ready_operations` is the set of ops that *can* start.
    # The `remaining_work` field in the op dict typically represents the remaining work for that *job*.
    # If a job is not ready, it's not in this list. But its remaining work is still part of the makespan.
    # We only have access to `ready_operations`. 
    # The previous heuristic used `max(op.get("remaining_work") for op in ready_operations)`.
    # This implies the simulator provides `remaining_work` for the job, and we assume the max among ready ops 
    # is a good proxy for the critical path, OR that non-ready jobs are not on the critical path 
    # (which is false, but a common heuristic approximation).
    
    # Let's stick to the approximation: The global LB is the max remaining work among *all* jobs.
    # We only see ready jobs. We assume the max among ready jobs is the current bottleneck.
    # If we reduce a job that is NOT the max, the global max doesn't change.
    # If we reduce a job that IS the max, the global max might drop to the second-highest.
    
    # So, for each candidate op i in job j:
    # new_work_j = remaining_work[j] - duration
    # other_max = max(remaining_work[k] for all other ready jobs k != j)
    # Note: We don't know the remaining work of non-ready jobs. We ignore them (assume they are <= current max).
    # new_global_max = max(new_work_j, other_max)
    
    # We want to minimize new_global_max.
    
    # Optimization: 
    # 1. Find the current max (M1) and the second max (M2) among ready jobs.
    # 2. For any job j where remaining_work[j] < M1:
    #    new_work_j = rw_j - dur < M1 - dur < M1.
    #    other_max = M1 (since M1 belongs to some other job k).
    #    new_global_max = max(<M1, M1) = M1.
    #    So reducing a non-max job does not change the LB (it stays M1).
    # 3. For a job j where remaining_work[j] == M1:
    #    new_work_j = M1 - dur.
    #    other_max = M2 (if there are other jobs with M1, M2=M1; else M2 is the next highest).
    #    new_global_max = max(M1 - dur, M2).
    
    # Therefore, we should ONLY consider operations from jobs that have the MAXIMUM remaining work.
    # Among those, we want to minimize max(M1 - dur, M2).
    # Since M1 and M2 are fixed for the current state, minimizing max(M1 - dur, M2) is equivalent to:
    # - If M1 - dur <= M2, the value is M2 (constant for all such ops).
    # - If M1 - dur > M2, the value is M1 - dur.
    # So we want to make (M1 - dur) as small as possible, i.e., maximize `dur`? 
    # Wait. We want to MINIMIZE the makespan.
    # If we pick an op with large `dur`, M1 - dur becomes smaller.
    # However, `dur` is the time the machine is busy. In a makespan minimization, 
    # reducing the critical path length is key.
    # But wait, if I pick a LONG operation, I am committing the machine for a long time.
    # Does that help the makespan?
    # The LB is the max remaining work. Reducing the work on the critical job reduces the LB.
    # A longer operation reduces the remaining work more.
    # So, theoretically, to drop the LB fastest, I should pick the LONGEST operation on the critical job?
    # Let's check:
    # Job A: 100 work. Job B: 90 work.
    # Op on A: 10 units. New A: 90. New Max: max(90, 90) = 90.
    # Op on A: 50 units. New A: 50. New Max: max(50, 90) = 90.
    # In this case, both result in LB 90.
    # But if Op on A is 100 units. New A: 0. New Max: max(0, 90) = 90.
    # So the LB drops to 90 regardless (as long as we clear enough work or if B is the bottleneck).
    # However, if B was 10 work.
    # Op on A: 10. New A: 90. Max: 90.
    # Op on A: 100. New A: 0. Max: 10.
    # Here, picking the longer op drops the LB from 100 to 10, while picking the short op drops it to 90.
    # So YES, picking the LONGER operation on the critical job is better for the LB.
    
    # BUT, there is a counter-argument: Machine contention.
    # If I pick a long op, the machine is blocked.
    # If I pick a short op, the machine is freed sooner, allowing other jobs to progress.
    # The LB heuristic ignores machine contention.
    # The previous heuristic (SPT) picked the SHORTEST.
    # Let's try a hybrid: 
    # 1. Filter to jobs with max remaining work.
    # 2. Among those, pick the op that minimizes `max(M1 - dur, M2)`.
    #    This effectively prefers longer ops if they drop the max below M2, or if M2 is low.
    #    If M2 is high (close to M1), it doesn't matter much.
    #    If M2 is low, we want to minimize M1 - dur -> maximize dur.
    
    # Let's calculate M1 and M2.
    works = [op.get("remaining_work", 0.0) for op in ready_operations]
    if not works:
        return _fast_dispatch(ready_operations, time)
        
    # Sort unique works to find top 2
    # Since n is small, just sort the list of works (with duplicates removed for M2 logic? No, M2 is the max of the *other* jobs).
    # Actually, if two jobs have M1, then for one of them, the "other max" is M1.
    
    # Let's just iterate candidates.
    best_score = float('inf')
    best_idx = -1
    best_machine = None
    
    # Pre-calculate M1 and M2 (second highest distinct job work? Or just second highest value?)
    # If Job A=100, Job B=100.
    # Pick op from A. Other max is 100 (from B).
    # New Max = max(100-dur, 100) = 100.
    # So if multiple jobs are at the max, reducing one doesn't help the LB until the others are reduced.
    
    # So, we should only consider ops from jobs that are UNIQUELY at the max?
    # If there are multiple jobs at max, the LB is stuck at that max.
    # We should prioritize reducing the number of jobs at the max?
    # Or just pick any?
    # If we pick from Job A (max=100), and Job B (max=100) exists.
    # New LB = 100.
    # If we pick from Job C (max=90).
    # New LB = 100.
    # So it doesn't matter for the LB.
    # In that case, fall back to SPT (shortest processing time) to free machines?
    # Or LPT (longest) to clear work?
    
    # Let's refine:
    # 1. Find M1 (max work) and count of jobs at M1.
    # 2. If count == 1:
    #    Only that job can reduce the LB.
    #    We want to minimize max(M1 - dur, M2).
    #    This implies we want to maximize `dur` (to push M1-dur down).
    #    BUT, we must also consider that a very long op blocks the machine.
    #    Let's try: Among ops on the unique max job, pick the one with the LARGEST duration?
    #    Or maybe a balance: `score = (M1 - dur) + 0.1 * dur`?
    #    Let's stick to the LB reduction: Minimize `max(M1 - dur, M2)`.
    #    This is equivalent to: 
    #       If M1 - dur <= M2, score = M2.
    #       Else score = M1 - dur.
    #    To minimize this, we want M1 - dur to be small. So maximize dur.
    #    So: Pick the LONGEST op on the unique critical job.
    
    # 3. If count > 1:
    #    LB cannot be reduced by any single op (since other critical jobs remain at M1).
    #    We should pick an op that helps clear the critical set.
    #    SPT (shortest) is good to free machines and allow parallel progress on other critical jobs?
    #    Or LPT to finish a critical job completely?
    #    If we finish a critical job, the count of critical jobs drops by 1.
    #    If we just reduce work, it stays critical.
    #    So, we should prefer ops that COMPLET the job (i.e., remaining_ops == 1 or duration == remaining_work)?
    #    `remaining_ops` tells us how many ops are left.
    #    If `remaining_ops` is 1, doing this op finishes the job.
    #    So, prioritize ops with `remaining_ops == 1`.
    #    Among those, maybe pick the shortest to finish quickly?
    #    If no op finishes a job, pick the one that reduces work most? (Longest)
    #    Or shortest to free machine?
    #    Let's try: 
    #    Priority 1: Ops that finish a job (remaining_ops == 1).
    #    Priority 2: Among those, shortest duration (SPT) to free machine.
    #    Priority 3: If none finish a job, pick the one with shortest duration (SPT) to free machine?
    #    Or longest to reduce LB? But LB doesn't drop.
    #    Let's try SPT for the non-unique max case.
    
    # Let's implement this logic.
    
    # Step 1: Identify jobs and their max work
    job_work = {}
    for i, op in enumerate(ready_operations):
        j = op["job"]
        w = op.get("remaining_work", 0.0)
        if j not in job_work or w > job_work[j]:
            job_work[j] = w
            
    if not job_work:
        return _fast_dispatch(ready_operations, time)
        
    max_work = max(job_work.values())
    
    # Find jobs at max work
    critical_jobs = [j for j, w in job_work.items() if w == max_work]
    num_critical = len(critical_jobs)
    
    # Find M2 (second highest work among jobs NOT in critical_jobs? Or just second highest overall?)
    # If num_critical == 1, M2 is the max of the other jobs.
    # If num_critical > 1, M2 is effectively max_work (since another critical job remains).
    
    if num_critical == 1:
        crit_job = critical_jobs[0]
        # M2 is max work of jobs != crit_job
        m2 = 0.0
        for j, w in job_work.items():
            if j != crit_job and w > m2:
                m2 = w
        
        best_idx = -1
        best_machine = None
        best_score = float('inf')
        
        for i, op in enumerate(ready_operations):
            if op["job"] != crit_job:
                continue
            mo = op.get("machine_options", {})
            if not mo:
                continue
            
            # We want to minimize max(max_work - dur, m2)
            # This is minimized by maximizing dur (up to a point).
            # Let's find the op that minimizes the resulting LB.
            # If multiple ops yield the same LB, we might prefer shorter duration to free the machine?
            # Or longer to clear work?
            # Let's stick to minimizing LB. If tie, break by shorter duration (SPT) to free machine sooner?
            # Actually, if LB is same, the "time to reach that LB" is different.
            # If I pick a long op, I'm busy longer.
            # If I pick a short op, I'm busy shorter, but LB is same (if M2 is the bottleneck).
            # If M2 is the bottleneck, it doesn't matter which op I pick from the critical job, 
            # as long as I pick one. The LB stays M2.
            # So I should pick the SHORTEST op to free the machine for other jobs!
            # Wait. If I pick the shortest op, I free the machine sooner.
            # If I pick the longest op, I block the machine.
            # So if M2 is the bottleneck (M1 - dur <= M2 for all dur), pick SPT.
            # If M1 - dur > M2 for some dur, I want to make M1 - dur small.
            # So I want dur to be large.
            
            # Let's calculate the score for each op.
            # Score = max(max_work - dur, m2)
            # If scores are equal, prefer smaller dur (SPT) to minimize machine blockage.
            
            for m, d in mo.items():
                new_lb = max(max_work - d, m2)
                # Tie-breaker: smaller d is better if new_lb is same.
                # If new_lb is smaller, it's better regardless of d.
                
                # We need to compare (new_lb, d).
                # Lower new_lb is better. If new_lb is same, lower d is better.
                
                if new_lb < best_score - 1e-9:
                    best_score = new_lb
                    best_idx = i
                    best_machine = m
                    # We need to store the duration for tie-breaking? 
                    # Or just re-evaluate?
                    # Let's store the best duration seen for the best score.
                    best_dur = d
                elif abs(new_lb - best_score) < 1e-9:
                    if d < best_dur - 1e-9:
                        best_dur = d
                        best_idx = i
                        best_machine = m
        
        if best_idx == -1:
            return _fast_dispatch(ready_operations, time)
        return (best_idx, best_machine)
        
    else:
        # Multiple critical jobs.
        # LB is stuck at max_work.
        # We want to reduce the number of critical jobs.
        # Prefer ops that FINISH a job (remaining_ops == 1).
        # Among those, pick SPT (shortest) to finish quickly.
        # If no op finishes a job, pick SPT to free machines.
        
        best_idx = -1
        best_machine = None
        best_priority = float('inf')
        best_dur = float('inf')
        
        for i, op in enumerate(ready_operations):
            if op["job"] not in critical_jobs:
                continue
            mo = op.get("machine_options", {})
            if not mo:
                continue
            
            remaining_ops = op.get("remaining_ops", 0)
            # Priority 0: Finishes job
            # Priority 1: Reduces work (all do this)
            # Let's assign priority:
            # if remaining_ops == 1: priority = 0
            # else: priority = 1
            
            prio = 0 if remaining_ops == 1 else 1
            
            # Find min duration for this op
            min_d = min(mo.values())
            # Find machine for min_d
            m_for_min_d = None
            for m, d in mo.items():
                if d == min_d:
                    m_for_min_d = m
                    break
            
            if prio < best_priority - 1e-9:
                best_priority = prio
                best_dur = min_d
                best_idx = i
                best_machine = m_for_min_d
            elif abs(prio - best_priority) < 1e-9:
                if min_d < best_dur - 1e-9:
                    best_dur = min_d
                    best_idx = i
                    best_machine = m_for_min_d
        
        if best_idx == -1:
            return _fast_dispatch(ready_operations, time)
        return (best_idx, best_machine)

# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
