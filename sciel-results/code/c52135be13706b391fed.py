"""Improved dispatch rule for Flexible Job-Shop Scheduling."""

def dispatch(ready_operations, time):
    """
    Picks the ready operation to start next and the best machine for it.
    
    Strategy:
    1. For each ready operation, determine the 'best' machine (minimum duration).
    2. Calculate a priority score based on the ratio of remaining work (urgency)
       to the minimum processing time (efficiency).
       Score = Remaining_Work / (Min_Duration + epsilon)
    3. Select the operation with the highest score.
    4. Return the index of the operation and the ID of the best machine.
    """
    if not ready_operations:
        return 0 # Fallback, though input should be non-empty

    best_index = 0
    best_score = -1.0
    best_machine = None

    for i, op in enumerate(ready_operations):
        remaining_work = op.get('remaining_work', 0)
        machine_options = op.get('machine_options', {})
        
        if not machine_options:
            continue

        # Find the machine with the minimum processing time
        min_duration = float('inf')
        min_machine = None
        for machine_id, duration in machine_options.items():
            if duration < min_duration:
                min_duration = duration
                min_machine = machine_id

        # If no valid machine found, skip or handle gracefully
        if min_machine is None:
            continue

        # Score: High remaining work and low duration is preferred.
        # Using a small epsilon to avoid division by zero.
        epsilon = 1e-6
        score = remaining_work / (min_duration + epsilon)
        
        # Tie-breaking: If scores are very close, prefer shorter duration (SPT)
        # to reduce blocking.
        if score > best_score or (abs(score - best_score) < 1e-9 and min_duration < (op.get('_last_min_dur', float('inf')))):
             # Note: We can't easily store state in the dict across calls if it's immutable,
             # so simple tie break on min_duration if scores are equal is hard without extra logic.
             # Let's just stick to the primary score. If tied, pick the one with smaller min_duration.
             pass

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = min_machine
        elif abs(score - best_score) < 1e-9:
            # Tie-break: Prefer shorter min_duration
            # We need to compare with the current best's min_duration.
            # We didn't store it. Let's re-calculate or just pick the first one found in case of tie?
            # Better: If score is tied, we should compare min_duration.
            # Let's modify the loop to track best_min_duration.
            pass

    # Refining the loop to handle tie-breaking properly
    best_index = 0
    best_score = -1.0
    best_machine = None
    best_min_dur = float('inf')

    for i, op in enumerate(ready_operations):
        remaining_work = op.get('remaining_work', 0)
        machine_options = op.get('machine_options', {})
        
        if not machine_options:
            continue

        min_duration = float('inf')
        min_machine = None
        for machine_id, duration in machine_options.items():
            if duration < min_duration:
                min_duration = duration
                min_machine = machine_id

        if min_machine is None:
            continue

        epsilon = 1e-6
        score = remaining_work / (min_duration + epsilon)
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = min_machine
            best_min_dur = min_duration
        elif abs(score - best_score) < 1e-9:
            # Tie-break: Prefer shorter duration
            if min_duration < best_min_dur:
                best_index = i
                best_machine = min_machine
                best_min_dur = min_duration
                # best_score remains effectively the same

    return (best_index, best_machine)
