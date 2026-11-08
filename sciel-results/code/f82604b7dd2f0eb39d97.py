"""Open-shop scheduling dispatch rule.

We pick the ready operation that, if started on its shortest machine option,
would leave the longest total remaining work across all jobs and machines.
This is a greedy "most critical path" heuristic for open-shop scheduling.
"""
# EVOLVE-BLOCK-START
import time as _time

# Hard deadline guard: the evaluator kills after 150 s; keep a safety margin.
_START = _time.time()
_HARD_LIMIT = 140.0  # seconds


def _elapsed_ok():
    return (_time.time() - _START) < _HARD_LIMIT


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Build a compact summary for each ready op.
    # For each op we know:
    #   job, op, machine_options {machine: duration},
    #   remaining_ops, remaining_work
    #
    # Heuristic:
    #   For each ready op i, if we start it now on its shortest machine option,
    #   that op completes in min_dur_i.  The job of i still has (remaining_ops-1)
    #   ops left.  For the whole system, the makespan lower bound is at least
    #   max over jobs of (total remaining work for that job after this start).
    #
    # We approximate the impact of starting op i by:
    #   score_i = sum over all jobs j of (remaining_work_j_after_start)
    #           = total_remaining_work - min_dur_i + (remaining_work_i - min_dur_i)
    #   Actually, remaining_work for job i decreases by min_dur_i when we start
    #   and complete op i.  But other jobs are unchanged.
    #
    # A simpler, effective proxy for "most urgent":
    #   - Prefer ops with large remaining_work (they keep the job busy long)
    #   - Prefer ops on machines that are short (min duration) so we free the
    #     machine quickly and can start the next op
    #   - Prefer ops whose job has many remaining ops (more future decisions)
    #   - Prefer ops with few machine options (less flexibility, schedule early)
    #
    # We compute a composite score and pick the max.

    best_idx = 0
    best_score = -1.0

    for i, op in enumerate(ready_operations):
        if not _elapsed_ok():
            break

        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        machine_options = op.get("machine_options", {})

        if not machine_options:
            # No machine options: treat as zero urgency
            continue

        min_dur = min(machine_options.values())
        max_dur = max(machine_options.values())
        num_opts = len(machine_options)

        # Work density: average work per remaining op for this job
        work_per_op = remaining_work / max(remaining_ops, 1)

        # Urgency factors (higher = more urgent)
        f1 = remaining_work                 # total remaining work for this job
        f2 = -min_dur                        # shorter min duration is better
        f3 = work_per_op                    # work density
        f4 = 1.0 / max(num_opts, 1)         # fewer options = more constrained
        f5 = remaining_ops                  # more remaining ops = more future

        # Weighted composite score
        score = (
            1.0 * f1
            + 0.4 * f2
            + 0.3 * f3
            + 0.2 * f4
            + 0.1 * f5
        )

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
