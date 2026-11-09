import time
import numpy as np

# Global state to track machine loads dynamically
# Since the simulation is event-driven and we only see 'ready_operations' at each step,
# we can infer current machine availability by tracking which machines are busy.
# However, the interface `dispatch(ready_operations, time)` does not provide a global state object.
# We must rely on the provided `ready_operations` and `time`.
# 
# The key insight for a "dynamic bottleneck identification" without global state is to look at the 
# composition of the ready set. If many jobs are waiting for the same machine, that machine is a bottleneck.
# 
# Strategy: 
# 1. For each ready operation, identify the "best" machine (min duration).
# 2. Count how many other ready operations *also* prefer that same machine (or have it as a very close second).
# 3. Assign a "congestion penalty" to operations that want to use already-congested machines.
# 4. Prefer operations that can be placed on less-congested machines, or operations that are critical (high remaining work) 
#    but can be placed on a machine that isn't currently a bottleneck for the *other* ready ops.
#
# To make it "maximize minimum slack", we approximate slack by looking at the distribution of preferred machines.
# If Machine A is preferred by 5 jobs and Machine B by 1 job, Machine A is likely to be the bottleneck.
# We should prioritize the job on Machine B to keep it free, or prioritize a job that can use Machine B to relieve pressure?
# Actually, in JSS, we usually want to keep critical jobs moving. 
#
# Let's refine the "Slack" concept:
# For each ready op i, let m_i be its min-duration machine.
# Let Load(m) be the count of ready ops whose min-duration machine is m.
# If we assign op i to m_i, we effectively "commit" that machine.
# We want to avoid creating a situation where one machine has a huge queue of "preferred" jobs.
#
# New Heuristic:
# Score(op) = (Remaining Work / Duration) * (1 / (1 + ConflictCount(machine)))
# Where ConflictCount is how many other ready ops also have this machine as their min-duration choice.
# This penalizes operations that are joining a crowd on a specific machine, effectively spreading the load
# and trying to keep the "minimum slack" (least loaded machine) available for future critical jobs.

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n_ops = len(ready_operations)
    
    # 1. Identify the preferred machine (min duration) for each operation
    # and calculate the base priority (WSPT-like: work / duration)
    preferred_machines = [None] * n_ops
    base_scores = [0.0] * n_ops
    min_durations = [0.0] * n_ops
    
    for i, op in enumerate(ready_operations):
        options = op['machine_options']
        if not options:
            base_scores[i] = -1
            continue
        
        min_dur = min(options.values())
        # Pick the first machine with min duration (deterministic tie-breaking)
        best_machine = next(m for m, d in options.items() if d == min_dur)
        
        preferred_machines[i] = best_machine
        min_durations[i] = min_dur
        
        remaining_work = op['remaining_work']
        # WSPT: Higher work-to-duration ratio is higher priority
        # Add a small epsilon to avoid div by zero, though durations should be > 0
        base_scores[i] = remaining_work / (min_dur + 1e-9)

    # 2. Count congestion on each preferred machine
    # This represents the "current demand" for each machine among ready ops
    machine_load = {}
    for m in preferred_machines:
        if m is not None:
            machine_load[m] = machine_load.get(m, 0) + 1

    # 3. Calculate final score incorporating congestion penalty
    # We want to select the operation that maximizes its "effective" priority
    # while considering that choosing a crowded machine might worsen the global bottleneck.
    # 
    # Strategy: Maximize (Base Score / (1 + Load))
    # This effectively says: "I have high priority, but if the machine I need is already crowded 
    # by other high-priority jobs, my effective urgency to *start now* on that specific machine 
    # is diluted because the queue is forming. Better to start a job on a less crowded machine 
    # to keep that machine's slack high."
    
    best_idx = 0
    best_score = -1
    best_machine = None

    for i, op in enumerate(ready_operations):
        if preferred_machines[i] is None:
            continue
            
        m = preferred_machines[i]
        load = machine_load.get(m, 1)
        
        # The denominator (1 + load) penalizes joining a crowded queue.
        # If load is 1 (only this job wants it), penalty is 2.
        # If load is 5 (5 jobs want it), penalty is 6.
        # This encourages dispatching jobs that can use less congested resources first,
        # thereby preserving the "slack" on those machines for future critical operations.
        
        final_score = base_scores[i] / (1 + load)
        
        if final_score > best_score:
            best_score = final_score
            best_idx = i
            best_machine = m

    return (best_idx, best_machine)
