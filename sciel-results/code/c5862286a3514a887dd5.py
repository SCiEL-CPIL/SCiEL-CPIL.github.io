"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Calculate scores:
    # Primary: remaining_work (Tail)
    # Secondary: Shortest processing time for the current op (to break ties and minimize immediate delay)
    
    # We want to maximize remaining_work.
    # If tied, we prefer the one with the smallest min_duration?
    # Let's combine them into a single score.
    # score = remaining_work * 1e6 - min_duration
    # This prioritizes remaining_work heavily, and breaks ties by preferring shorter immediate tasks.
    
    scores = []
    best_machines = []
    
    for op in ready_operations:
        work = op["remaining_work"]
        options = op["machine_options"]
        
        # Find the minimum duration among eligible machines
        if options:
            min_dur = min(options.values())
            best_machine = min(options, key=options.get)
        else:
            min_dur = 0
            best_machine = 0
            
        # Composite score
        # We want to maximize work.
        # If work is equal, we want to minimize min_dur.
        # So: score = work * 10000 - min_dur
        # Since durations are likely small (integers), 10000 is a safe multiplier.
        score = work * 10000.0 - min_dur
        scores.append(score)
        best_machines.append(best_machine)

    scores = np.array(scores)
    idx = int(np.argmax(scores))
    
    # Return tuple (index, machine) to explicitly select the best machine
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
