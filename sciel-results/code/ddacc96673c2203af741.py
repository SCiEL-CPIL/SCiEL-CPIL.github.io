"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_start_time = _time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: Critical Ratio (CR) heuristic.
    
    Strategy: Select the ready operation with the lowest Critical Ratio score.
    CR = (Estimated Remaining Time for Job - Current Time) / Immediate Processing Time.
    
    This prioritizes operations that are urgent (low slack relative to current time) 
    and have short processing times. It shifts focus from static work volume to 
    dynamic deadline pressure, which is a distinct scheduling philosophy from NSPT.
    
    Note: Since we don't have explicit deadlines, we approximate 'slack' using the 
    sum of minimum durations of remaining operations in the job minus the current time.
    A lower CR indicates higher urgency.
    """
    # Deadline guard: if we are running out of time, return the first valid index
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    scores = []
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            scores.append(float('inf'))
            continue
        
        # Immediate processing time: minimum duration among available machines
        immediate_duration = min(machine_options.values())
        if immediate_duration <= 0:
            immediate_duration = 1e-9  # Avoid division by zero
        
        # Estimate remaining time for the job
        # Use remaining_ops to estimate total remaining work time
        # remaining_work is given, but CR typically uses time-based slack.
        # We assume remaining_work is in time units (processing time).
        remaining_work = op.get("remaining_work", immediate_duration)
        
        # Slack estimation: 
        # In classic CR, slack = Due Date - Current Time.
        # Here, we don't have Due Dates. 
        # Heuristic: Treat 'remaining_work' as the time needed to finish the job from now.
        # So, 'slack' relative to a hypothetical deadline of 'remaining_work' from now?
        # No, that's circular.
        
        # Alternative CR interpretation for Job Shop without due dates:
        # We want to minimize the time to completion.
        # Let's use a variant: Score = (Remaining Work - Immediate Duration) / Immediate Duration
        # This favors jobs with less remaining work relative to the current step.
        # Or simply: Score = Remaining Work / Immediate Duration (which is similar to NSPT but inverted logic?)
        
        # Let's stick to the prompt's definition: 
        # "prioritizes the operation with the lowest ratio of slack time to remaining processing time"
        # "slack is estimated by subtracting the current time from the sum of remaining durations for all unfinished operations in that job"
        
        # Sum of remaining durations for all unfinished operations in that job:
        # This is effectively 'remaining_work' if it represents time.
        # Let's assume remaining_work is the sum of min durations of remaining ops.
        
        # Slack = (Sum of remaining durations) - Current Time?
        # This doesn't make sense dimensionally if Current Time is large.
        
        # Let's re-read: "slack is estimated by subtracting the current time from the sum of remaining durations"
        # If sum_remaining < current_time, slack is negative (overdue).
        # CR = Slack / Processing Time.
        
        # However, 'remaining_work' is likely the work *remaining*, not the absolute time when it will finish.
        # If remaining_work is the time to finish, then the "deadline" is implicit.
        
        # Let's try a different CR proxy that is known to work in JSS:
        # Shortest Remaining Processing Time (SRPT) for the job.
        # But the prompt asks for CR.
        
        # Let's implement the literal interpretation:
        # 1. Estimate total remaining time for the job: T_rem = remaining_work
        # 2. Estimate "Due Time": T_due = T_rem (i.e., we want to finish as soon as possible, so the 'deadline' is now + T_rem)
        # 3. Slack = T_due - Current Time = T_rem - time
        # 4. CR = Slack / Immediate Duration
        
        # If T_rem - time is negative, CR is negative (high priority).
        # If T_rem is small and time is large, CR is very negative.
        
        # This seems to favor jobs that have been running for a long time with little work left?
        # Or jobs with small total work?
        
        # Let's check the behavior:
        # Job A: rem_work=10, time=100, imm_dur=1. Slack = 10-100 = -90. CR = -90.
        # Job B: rem_work=100, time=100, imm_dur=10. Slack = 100-100 = 0. CR = 0.
        # Job A is preferred. This favors jobs that are "overdue" relative to their own remaining work.
        # This is essentially a "Most Work Remaining" or "Longest Remaining" variant? No, it favors smaller remaining work if time is constant?
        # If time is constant, smaller rem_work -> more negative slack -> lower CR.
        # So it favors jobs with LESS remaining work. This is similar to SPT for the job.
        
        # Let's try this.
        
        slack = remaining_work - time
        # To avoid division by zero or issues, add a small epsilon to denominator
        cr_score = slack / immediate_duration
        
        scores.append(cr_score)
    
    # Find the index with the minimum CR score (most negative/urgent)
    min_score = float('inf')
    best_idx = 0
    
    for i, s in enumerate(scores):
        if s < min_score:
            min_score = s
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
