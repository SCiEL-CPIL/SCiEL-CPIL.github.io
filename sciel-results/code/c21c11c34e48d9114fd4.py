"""Job-shop dispatch rule: SEARCH approach.
At import time we generate many candidate dispatch heuristics (varied weights/rules),
evaluate each on small synthetic job-shop instances via an event-driven simulation,
and keep the best one within a wall-clock budget. dispatch() then just applies the
chosen rule with a cheap deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
import random
import heapq

_DEADLINE = 130.0        # Safety margin before 150s kill
_START_TIME = _time.time()
_BUDGET_FOR_SEARCH = 115.0  # leave margin for evaluation overhead

# ---------------------------------------------------------------------------
# Candidate rule definitions.
# Each rule is a function of (op, time, n_ops) -> score (higher = more urgent).
# op has keys: job, op, machine_options {machine: duration}, remaining_ops, remaining_work
# ---------------------------------------------------------------------------
def _min_dur(op):
    return min(op["machine_options"].values())

def _soft_min_dur(op, temp=1.0):
    """Soft-min surrogate via log-sum-exp for smoothness."""
    vals = list(op["machine_options"].values())
    if len(vals) == 1:
        return vals[0]
    # soft-min = -soft_max(-vals)
    neg_vals = [-v for v in vals]
    m = max(neg_vals)
    s = sum(np.exp((v - m) / temp) for v in neg_vals)
    return -(m + temp * np.log(s))

def _mean_dur(op):
    v = op["machine_options"].values()
    return sum(v) / len(v)

def _max_dur(op):
    return max(op["machine_options"].values())

def _num_machines(op):
    return len(op["machine_options"])

def _remaining_work(op):
    return op["remaining_work"]

def _remaining_ops(op):
    return op["remaining_ops"]

def _avg_remaining(op):
    ro = op["remaining_ops"]
    return op["remaining_work"] / ro if ro > 0 else op["remaining_work"]

def _machine_load(op):
    """Estimate machine load: sum of durations across all machine options."""
    return sum(op["machine_options"].values())

def _bottleneck_factor(op):
    """Higher for ops with fewer machine options (more constrained)."""
    nm = _num_machines(op)
    return 1.0 / (nm + 1.0) if nm > 0 else 1.0

# Rule families with tunable weights (a, b, c, d, e):
# score = a*eff + b*remaining_work - c*remaining_ops + d*mean_duration + e*num_machines
# where eff = remaining_work / min_duration  (efficiency)

def make_rule(a, b, c, d, e):
    def rule(op, time, n_ops):
        md = _min_dur(op)
        eff = _remaining_work(op) / md if md > 0 else _remaining_work(op) + 1000.0
        return (a * eff + b * _remaining_work(op) - c * _remaining_ops(op)
                + d * _mean_dur(op) + e * _num_machines(op))
    return rule

# SPT rule: minimize min duration -> score = -min_duration
def rule_spt(op, time, n_ops):
    return -_min_dur(op)

# SPT with soft-min surrogate
def rule_spt_soft(op, time, n_ops):
    return -_soft_min_dur(op, temp=0.5)

# Critical path slack: remaining_work / min_duration (urgency metric)
def rule_critical_slack(op, time, n_ops):
    rw = op["remaining_work"]
    md = _min_dur(op)
    if md <= 0:
        return rw * 100.0
    return rw / md

# Critical path slack with soft-min
def rule_critical_slack_soft(op, time, n_ops):
    rw = op["remaining_work"]
    md = _soft_min_dur(op, temp=0.5)
    if md <= 0:
        return rw * 100.0
    return rw / md

# Longest remaining work
def rule_lrw(op, time, n_ops):
    return _remaining_work(op)

# Most remaining ops
def rule_mro(op, time, n_ops):
    return _remaining_ops(op)

# Fewest remaining ops
def rule_fro(op, time, n_ops):
    return -_remaining_ops(op)

# Min remaining work
def rule_mrw(op, time, n_ops):
    return -_remaining_work(op)

# Mean duration (SPT on mean)
def rule_spt_mean(op, time, n_ops):
    return -_mean_dur(op)

# Ratio: remaining_work / num_machines
def rule_ratio(op, time, n_ops):
    nm = _num_machines(op)
    return _remaining_work(op) / nm if nm > 0 else _remaining_work(op)

# ATC: Apparent Tardiness Cost - prioritize jobs with more remaining work relative to time
def rule_atc(op, time, n_ops):
    rw = _remaining_work(op)
    md = _min_dur(op)
    if md <= 0:
        return rw
    # Higher score for ops with more remaining work that can be done quickly
    return rw / (md + 1.0)

# Machine contention aware: prefer ops that go to less loaded machines
def rule_machine_contention(op, time, n_ops):
    md = _min_dur(op)
    rw = _remaining_work(op)
    # Penalize if the op goes to a machine that's likely busy (long duration)
    # and reward if it can finish quickly
    return -md * 0.3 + rw / (md + 1.0) * 0.7

# Critical ratio with machine awareness
def rule_cr_machine(op, time, n_ops):
    rw = _remaining_work(op)
    md = _min_dur(op)
    ro = _remaining_ops(op)
    if ro <= 0:
        return rw
    # Critical ratio: remaining work / time to complete
    # Adjust for machine contention
    cr = rw / (md * ro + 1.0)
    return cr

# Critical path slack divided by avg remaining op duration
def rule_slack_avg(op, time, n_ops):
    rw = _remaining_work(op)
    ro = _remaining_ops(op)
    md = _min_dur(op)
    if md <= 0 or ro <= 0:
        return rw * 100.0
    avg_op_dur = rw / ro
    # Slack: how much work remains relative to what can be done quickly
    return (rw - avg_op_dur) / (md + 1.0) + rw / (md * ro + 1.0)

# CR: Critical Ratio - remaining_work / (time_limit - current_time)
# We don't have a time limit, so use a normalized version
def rule_cr(op, time, n_ops):
    rw = _remaining_work(op)
    ro = _remaining_ops(op)
    if ro <= 0:
        return rw
    avg_remaining = rw / ro
    # Prioritize jobs where remaining work is large relative to average
    return rw - avg_remaining * 0.5

# EDD-like: prioritize jobs with fewer remaining ops (closer to completion)
def rule_edd(op, time, n_ops):
    return -_remaining_ops(op) * 0.1 - _min_dur(op) * 0.1

# Composite: combine SPT with remaining work
def rule_spt_rw(op, time, n_ops):
    return -_min_dur(op) * 0.5 + _remaining_work(op) * 0.5

# Machine utilization aware: prefer ops on machines with fewer options (bottleneck)
def rule_bottleneck(op, time, n_ops):
    nm = _num_machines(op)
    md = _min_dur(op)
    # Fewer machine options = more constrained = higher priority
    return (1.0 / (nm + 1.0)) * _remaining_work(op) - md * 0.1

# Priority based on job completion urgency
def rule_completion_urgency(op, time, n_ops):
    ro = _remaining_ops(op)
    rw = _remaining_work(op)
    md = _min_dur(op)
    # Jobs closer to completion get higher priority
    urgency = 1.0 / (ro + 1.0)
    # But also consider how much work remains relative to what can be done now
    efficiency = rw / (md + 1.0) if md > 0 else rw
    return urgency * 0.6 + efficiency * 0.4

# Load balancing: spread work across machines
def rule_load_balance(op, time, n_ops):
    md = _min_dur(op)
    rw = _remaining_work(op)
    nm = _num_machines(op)
    # Prefer ops that can go to any machine (flexible) but also consider load
    flexibility = 1.0 / (nm + 1.0)
    load = rw / nm if nm > 0 else rw
    return flexibility * 0.5 - load * 0.1

# ---------------------------------------------------------------------------
# Event-driven job-shop simulator.
# ready_operations: list of dicts for the ready ops at each event.
# The simulator maintains, for each job, its operation sequence and machine
# availability, and uses the dispatch rule to pick which ready op to start.
# ---------------------------------------------------------------------------
class JobShopInstance:
    def __init__(self, jobs_ops):
        # jobs_ops: list of jobs, each a list of (machine, duration)
        self.jobs = jobs_ops
        self.n_jobs = len(jobs_ops)
        self.machines = set()
        for job in jobs_ops:
            for m, d in job:
                self.machines.add(m)
        self.machines = sorted(self.machines)
        self.n_machines = len(self.machines)

    def total_work(self):
        return sum(d for job in self.jobs for _, d in job)

def simulate(instance, rule, max_steps=100000):
    """Run event-driven simulation. Returns makespan."""
    n_jobs = instance.n_jobs
    n_machines = instance.n_machines

    # job[i] = (current_op_index, [ops])
    job_cur_op = [0] * n_jobs
    job_remaining_ops = [len(instance.jobs[i]) for i in range(n_jobs)]
    job_remaining_work = [sum(d for _, d in instance.jobs[i]) for i in range(n_jobs)]

    machine_avail = [0] * n_machines
    machine_to_idx = {m: idx for idx, m in enumerate(instance.machines)}

    # Track completed ops per job to build ready_operations dynamically
    completed = [0] * n_jobs

    # We'll simulate by repeatedly: find all ready ops, dispatch picks one,
    # schedule it on its (single) machine, update availability, repeat.
    # In a 1-machine-per-op job-shop, each op has exactly one machine option.
    # But the interface allows machine_options dict; for generated instances we
    # use single machine per op (classic job-shop). The dispatch receives
    # machine_options as {machine: duration}.

    t = 0.0
    finished = 0
    steps = 0
    max_t = 0.0

    while finished < n_jobs and steps < max_steps:
        steps += 1
        # Build ready operations list
        ready = []
        for i in range(n_jobs):
            if job_cur_op[i] < len(instance.jobs[i]):
                m, d = instance.jobs[i][job_cur_op[i]]
                ready.append({
                    "job": i,
                    "op": job_cur_op[i],
                    "machine_options": {m: d},
                    "remaining_ops": job_remaining_ops[i],
                    "remaining_work": job_remaining_work[i],
                })
        if not ready:
            break

        # Dispatch
        if len(ready) == 1:
            idx = 0
        else:
            scores = [rule(op, t, len(ready)) for op in ready]
            idx = int(np.argmax(scores))

        op = ready[idx]
        i = op["job"]
        m, d = instance.jobs[i][job_cur_op[i]]
        m_idx = machine_to_idx[m]

        start = max(t, machine_avail[m_idx])
        end = start + d
        machine_avail[m_idx] = end

        # Update job state
        job_cur_op[i] += 1
        job_remaining_ops[i] -= 1
        job_remaining_work[i] -= d
        completed[i] += 1

        t = end
        if t > max_t:
            max_t = t
        if job_cur_op[i] >= len(instance.jobs[i]):
            finished += 1

    return max_t

# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------
def generate_instance(n_jobs=6, n_machines=4, seed=0):
    rng = random.Random(seed)
    jobs = []
    for j in range(n_jobs):
        ops = []
        # Each job visits a random permutation of machines (classic job-shop)
        ms = list(range(n_machines))
        rng.shuffle(ms)
        for m in ms:
            d = rng.randint(1, 10)
            ops.append((m, d))
        jobs.append(ops)
    return JobShopInstance(jobs)

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def generate_candidates():
    candidates = []
    # Base rules
    candidates.append(("spt", rule_spt))
    candidates.append(("spt_soft", rule_spt_soft))
    candidates.append(("lrw", rule_lrw))
    candidates.append(("mro", rule_mro))
    candidates.append(("fro", rule_fro))
    candidates.append(("mrw", rule_mrw))
    candidates.append(("spt_mean", rule_spt_mean))
    candidates.append(("ratio", rule_ratio))
    candidates.append(("atc", rule_atc))
    candidates.append(("cr", rule_cr))
    candidates.append(("edd", rule_edd))
    candidates.append(("spt_rw", rule_spt_rw))
    candidates.append(("bottleneck", rule_bottleneck))
    candidates.append(("machine_contention", rule_machine_contention))
    candidates.append(("cr_machine", rule_cr_machine))
    candidates.append(("completion_urgency", rule_completion_urgency))
    candidates.append(("load_balance", rule_load_balance))
    candidates.append(("critical_slack", rule_critical_slack))
    candidates.append(("critical_slack_soft", rule_critical_slack_soft))
    candidates.append(("slack_avg", rule_slack_avg))

    # Weighted hybrid rules
    # a: efficiency weight, b: remaining_work weight, c: remaining_ops weight,
    # d: mean_duration weight, e: num_machines weight
    grid_a = [0.0, 0.5, 1.0, 2.0, 3.0]
    grid_b = [0.0, 0.1, 0.5]
    grid_c = [0.0, 0.05, 0.1]
    grid_d = [-0.1, 0.0, 0.1]
    grid_e = [-0.05, 0.0, 0.05]

    for a in grid_a:
        for b in grid_b:
            for c in grid_c:
                for d in grid_d:
                    for e in grid_e:
                        rule = make_rule(a, b, c, d, e)
                        name = f"hyb_{a}_{b}_{c}_{d}_{e}"
                        candidates.append((name, rule))

    # Critical slack hybrid: weight between critical_slack and spt
    for w in [0.2, 0.4, 0.5, 0.6, 0.8]:
        def make_slack_hyb(w):
            def rule(op, time, n_ops):
                s1 = rule_critical_slack(op, time, n_ops)
                s2 = rule_spt(op, time, n_ops)
                return w * s1 + (1 - w) * s2
            return rule
        candidates.append((f"slack_spt_{w}", make_slack_hyb(w)))

    # Additional targeted combinations
    # Combine SPT with completion urgency
    for w1 in [0.3, 0.5, 0.7]:
        for w2 in [0.3, 0.5, 0.7]:
            if abs(w1 + w2 - 1.0) < 0.01:
                def make_comb(w1, w2):
                    def rule(op, time, n_ops):
                        s1 = rule_spt(op, time, n_ops)
                        s2 = rule_completion_urgency(op, time, n_ops)
                        return w1 * s1 + w2 * s2
                    return rule
                candidates.append((f"spt_urgency_{w1}_{w2}", make_comb(w1, w2)))

    # Combine ATC with machine contention
    for w1 in [0.3, 0.5, 0.7]:
        for w2 in [0.3, 0.5, 0.7]:
            if abs(w1 + w2 - 1.0) < 0.01:
                def make_comb2(w1, w2):
                    def rule(op, time, n_ops):
                        s1 = rule_atc(op, time, n_ops)
                        s2 = rule_machine_contention(op, time, n_ops)
                        return w1 * s1 + w2 * s2
                    return rule
                candidates.append((f"atc_contention_{w1}_{w2}", make_comb2(w1, w2)))

    # Combine critical_slack with completion_urgency
    for w1 in [0.3, 0.5, 0.7]:
        for w2 in [0.3, 0.5, 0.7]:
            if abs(w1 + w2 - 1.0) < 0.01:
                def make_comb3(w1, w2):
                    def rule(op, time, n_ops):
                        s1 = rule_critical_slack(op, time, n_ops)
                        s2 = rule_completion_urgency(op, time, n_ops)
                        return w1 * s1 + w2 * s2
                    return rule
                candidates.append((f"slack_urgency_{w1}_{w2}", make_comb3(w1, w2)))
    return candidates

# ---------------------------------------------------------------------------
# Tournament: evaluate each candidate on a set of instances, pick best
# ---------------------------------------------------------------------------
def run_tournament():
    candidates = generate_candidates()
    rng = random.Random(42)
    # Generate test instances closer to actual evaluation size (10 jobs, 5 machines)
    # Use a mix of sizes for robustness
    test_instances = []
    # Primary: 10 jobs, 5 machines (matches evaluation) - more instances
    for s in range(10):
        test_instances.append(generate_instance(n_jobs=10, n_machines=5, seed=s))
    # Secondary: smaller instances for diversity
    for s in range(3):
        test_instances.append(generate_instance(n_jobs=8, n_machines=5, seed=s + 100))
    for s in range(2):
        test_instances.append(generate_instance(n_jobs=6, n_machines=4, seed=s + 200))
    # Tertiary: larger instances to stress-test
    for s in range(2):
        test_instances.append(generate_instance(n_jobs=12, n_machines=5, seed=s + 300))

    results = []
    for name, rule in candidates:
        if _time.time() - _START_TIME > _BUDGET_FOR_SEARCH:
            break
        total = 0.0
        for inst in test_instances:
            ms = simulate(inst, rule)
            total += ms
        mean_ms = total / len(test_instances)
        results.append((mean_ms, name, rule))

    if not results:
        return rule_spt  # fallback
    results.sort(key=lambda x: x[0])
    best_rule = results[0][2]
    return best_rule

# ---------------------------------------------------------------------------
# Run search at import time
# ---------------------------------------------------------------------------
_best_rule = None
if _time.time() - _START_TIME < _BUDGET_FOR_SEARCH:
    try:
        _best_rule = run_tournament()
    except Exception:
        _best_rule = None
if _best_rule is None:
    _best_rule = rule_spt  # safe fallback

# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------
def dispatch(ready_operations, time):
    """Dispatch rule for job-shop scheduling.
    Uses the best rule found by the import-time tournament search."""
    if not ready_operations:
        return 0
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Compute scores using the chosen rule
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = _best_rule(ready_operations[i], time, n)

    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
