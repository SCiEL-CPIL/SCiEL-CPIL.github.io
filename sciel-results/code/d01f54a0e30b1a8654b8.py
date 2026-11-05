# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

_dispatch_start_time = _time.time()
_rng = np.random.default_rng(42)

_instances = []
_eval_cache = {}
_best_params = None


def _elapsed():
    return _time.time() - _dispatch_start_time


def _make_instances(njobs=10, nmachines=5, ninst=5, seed=7):
    rng = np.random.default_rng(seed)
    ops = 10  # ~ 50 ops total, matching "10 jobs, 5 machines" scale
    insts = []
    for _ in range(ninst):
        jobs = []
        for j in range(njobs):
            seq = []
            for _ in range(ops):
                k = int(rng.integers(2, 5))
                m = rng.integers(0, nmachines, size=k)
                m = np.unique(m)
                dur = np.round(rng.uniform(1.0, 9.0, size=len(m)), 3)
                seq.append({int(m[i]): float(dur[i]) for i in range(len(m))})
            jobs.append(seq)
        insts.append(jobs)
    return insts


def _run_sim(jobs, params):
    w_work, w_rem, w_remop, w_dur, w_done = params[:5]
    machine_mode = params[5]
    njobs = len(jobs)
    nmachines = max(max(max(j) for j in job) for job in jobs) + 1
    job_next = [0] * njobs
    job_done = [0.0] * njobs
    mach_avail = [0.0] * nmachines
    total = sum(len(j) for j in jobs)
    done = 0
    makespan = 0.0
    ready = []
    for j in range(njobs):
        if jobs[j]:
            ready.append(j)
    while ready:
        # compute features
        best_score = -1e18
        best_job = ready[0]
        best_machine = None
        best_time = 0.0
        for j in ready:
            seq = jobs[j]
            op = seq[job_next[j]]
            remaining_work = sum(min(v.values()) for v in seq[job_next[j]:])
            remaining_ops = len(seq) - job_next[j]
            # Completeness: 1 - (remaining_ops / total_ops_in_job), higher means closer to done
            total_ops_in_job = len(seq)
            completeness = 1.0 - (remaining_ops / total_ops_in_job) if total_ops_in_job > 0 else 1.0
            vals = list(op.values())
            longest = max(vals)
            shortest = min(vals)
            score = (w_work * longest + w_rem * remaining_work +
                     w_remop * remaining_ops - w_dur * shortest +
                     w_done * completeness)
            if score > best_score:
                best_score = score
                best_job = j
                if machine_mode == 1:
                    best_machine = max(op, key=op.get)
                else:
                    best_machine = min(op, key=op.get)
                best_time = op[best_machine]
        ready.remove(best_job)
        start = mach_avail[best_machine]
        end = start + best_time
        mach_avail[best_machine] = end
        job_done[best_job] = end
        done += 1
        if done == total:
            makespan = end
        job_next[best_job] += 1
        if job_next[best_job] < len(jobs[best_job]):
            ready.append(best_job)
    return makespan


def _build_instances():
    global _instances
    t0 = _time.time()
    if _instances:
        return
    _instances = _make_instances()
    # warm up cache a little if we have room
    if _time.time() - t0 < 40:
        for p in [
            (1.0, 1.0, 0.0, 0.0, 0.0, 0), (0.5, 1.0, 0.0, 0.0, 0.0, 0),
            (1.0, 0.5, 0.0, 0.0, 0.0, 0), (1.0, 1.0, 0.0, 0.5, 0.0, 0),
            (1.0, 1.0, 0.0, 0.0, 0.5, 0), (1.0, 0.5, 0.0, 0.0, 0.5, 0),
        ]:
            _cache_params(p)


def _cache_params(params):
    key = (params[0], params[1], params[2], params[3], params[4], params[5])
    if key in _eval_cache:
        return _eval_cache[key]
    s = 0.0
    for j in _instances:
        s += _run_sim(j, params)
    val = s / len(_instances)
    _eval_cache[key] = val
    return val


def _optimize():
    global _best_params
    # grid search over weights and machine policy
    w_work = [0.0, 0.5, 1.0, 1.5]
    w_rem = [0.0, 0.5, 1.0, 1.5]
    w_remop = [0.0, 0.5, 1.0]
    w_dur = [0.0, 0.25, 0.5]
    w_done = [0.0, 0.25, 0.5, 1.0]
    modes = [0, 1]
    best_val = float('inf')
    best_p = (1.0, 1.0, 0.0, 0.0, 0.0, 0)
    it = 0
    for ww in w_work:
        for wr in w_rem:
            for wro in w_remop:
                for wd in w_dur:
                    for wd_done in w_done:
                        for md in modes:
                            p = (ww, wr, wro, wd, wd_done, md)
                            v = _cache_params(p)
                            if v < best_val:
                                best_val = v
                                best_p = p
                            it += 1
                            if _time.time() - _dispatch_start_time > _DEADLINE - _SAFETY_MARGIN:
                                _best_params = best_p
                                return
    _best_params = best_p


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    if _elapsed() > _DEADLINE - _SAFETY_MARGIN:
        # safe fallback: longest remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    _build_instances()
    if _best_params is None:
        _optimize()
    if _best_params is None:
        # if optimization timed out before finishing, use a sensible default
        w_work, w_rem, w_remop, w_dur, w_done, machine_mode = 1.0, 1.0, 0.0, 0.0, 0.5, 0
    else:
        w_work, w_rem, w_remop, w_dur, w_done, machine_mode = _best_params

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        mo = op["machine_options"]
        if mo:
            best_machine = max(mo, key=mo.get) if machine_mode == 1 else min(mo, key=mo.get)
            return (0, best_machine)
        return 0

    # feature extraction
    longest_arr = np.array([max(op["machine_options"].values()) if op["machine_options"] else 0.0 for op in ready_operations])
    shortest_arr = np.array([min(op["machine_options"].values()) if op["machine_options"] else 0.0 for op in ready_operations])
    rem_work_arr = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    rem_op_arr = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    # Completeness: 1 - remaining_ops / total_ops_in_job
    # We don't have total_ops_in_job directly, but we can infer it from the job structure
    # Since remaining_ops decreases as we progress, and we don't know total, use a proxy:
    # Higher remaining_ops means less complete. Use 1 - normalized remaining_ops as completeness proxy.
    max_remop = np.max(rem_op_arr) if np.max(rem_op_arr) > 0 else 1.0
    completeness_arr = 1.0 - (rem_op_arr / max_remop)

    maxL = np.max(longest_arr) if np.max(longest_arr) > 0 else 1.0
    maxR = np.max(rem_work_arr) if np.max(rem_work_arr) > 0 else 1.0
    maxO = np.max(rem_op_arr) if np.max(rem_op_arr) > 0 else 1.0
    maxS = np.max(shortest_arr) if np.max(shortest_arr) > 0 else 1.0

    normL = longest_arr / maxL
    normR = rem_work_arr / maxR
    normO = rem_op_arr / maxO
    normS = shortest_arr / maxS

    scores = (w_work * normL + w_rem * normR + w_remop * normO - w_dur * normS +
              w_done * completeness_arr)
    best_idx = int(np.argmax(scores))

    mo = ready_operations[best_idx]["machine_options"]
    if mo:
        best_machine = max(mo, key=mo.get) if machine_mode == 1 else min(mo, key=mo.get)
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
