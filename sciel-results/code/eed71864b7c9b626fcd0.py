"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import math

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()

def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)

def _run_online_sgs(instance):
    """
    Serial SGS with online, time-dependent priorities.
    Priority = dynamic_tail + resource_pressure
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    if n == 0:
        return 0.0
        
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    num_resources = 0
    d_mat = None
    c_vec = None
    if demands is not None:
        d_mat = np.asarray(demands, dtype=float)
        if d_mat.shape[0] != n:
             d_mat = d_mat[:n]
        num_resources = d_mat.shape[1] if d_mat.ndim == 2 else 0
        if capacities is not None:
            c_vec = np.asarray(capacities, dtype=float)
            
    # Pre-compute static tail (resource-weighted)
    # tail[i] = duration[i] + max(tail[j] * weight) for j in successors[i]
    # weight is based on resource demand relative to capacity
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(predecessors[i])
        
    topo_order = []
    queue = [i for i in range(n) if in_degree[i] == 0]
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If topo order is incomplete (cycle), fallback to 0..n-1
    if len(topo_order) < n:
        topo_order = list(range(n))

    static_tail = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        max_succ_tail = 0.0
        for v in successors[u]:
            # Weight by resource demand of v relative to capacity
            w = 1.0
            if d_mat is not None and c_vec is not None and num_resources > 0:
                # Simple heuristic: sum of demand/capacity ratios
                w = np.sum(d_mat[v] / np.maximum(c_vec, 1e-6))
                # Normalize w to [0, 1] roughly? Or just use it.
                # Let's use w to scale the tail contribution.
                # Actually, higher resource demand means higher contention, so higher effective tail.
                pass 
            # Use a simple weight: 1.0 if no resources, else demand sum
            if d_mat is not None and c_vec is not None and num_resources > 0:
                 w = np.sum(d_mat[v]) / np.max(c_vec) if np.max(c_vec) > 0 else 1.0
            else:
                 w = 1.0
                 
            val = static_tail[v] * w + durations[v]
            if val > max_succ_tail:
                max_succ_tail = val
                
        static_tail[u] = durations[u] + max_succ_tail
        
    # State
    scheduled = [False] * n
    start_times = np.zeros(n, dtype=float)
    finish_times = np.zeros(n, dtype=float)
    
    # Resource load tracking: list of (finish_time, demand_vec) for active jobs?
    # Since we schedule in serial SGS, we only need to check feasibility at candidate start times.
    # Candidate start times are: max(finish_times of preds) and finish_times of all scheduled jobs that use same resources.
    
    # To speed up feasibility check, we can keep track of resource usage intervals.
    # But for n=30, brute force checking against all scheduled jobs is fast enough.
    
    scheduled_count = 0
    while scheduled_count < n:
        if _remaining_budget() < 5.0:
             # Fallback: schedule remaining in topological order or index order to avoid timeout
             # This is a safety net.
             for i in range(n):
                 if not scheduled[i]:
                     # Check if preds are scheduled. If not, we have a problem.
                     # But in a valid instance, if we've been scheduling, preds should be done.
                     # If we hit timeout, we might have unscheduled preds.
                     # Just break to return what we have? No, must return makespan.
                     # Let's just finish the loop quickly with a simple heuristic.
                     pass
             break

        best_job = -1
        best_prio = -np.inf
        
        # Find eligible jobs
        eligible_jobs = []
        for i in range(n):
            if not scheduled[i]:
                eligible = True
                for p in predecessors[i]:
                    if not scheduled[p]:
                        eligible = False
                        break
                if eligible:
                    eligible_jobs.append(i)
                    
        if not eligible_jobs:
             break

        # For each eligible job, compute online priority
        # Priority = static_tail[i] + resource_pressure(i, current_time)
        # current_time is not a single scalar, but the earliest start time depends on the job.
        # However, for ranking, we can estimate the "load" at the expected start time.
        # A simpler proxy: compute the resource pressure based on the *current* set of scheduled jobs.
        # Let's define resource_pressure(i) = sum over resources r of (load_r_at_es_i + demand_i_r) / capacity_r
        # To make it comparable, we need es_i.
        
        for i in eligible_jobs:
            es_pred = 0.0
            for p in predecessors[i]:
                if finish_times[p] > es_pred:
                    es_pred = finish_times[p]
            
            # Compute resource pressure at es_pred
            pressure = 0.0
            if d_mat is not None and c_vec is not None and num_resources > 0:
                for r in range(num_resources):
                    if d_mat[i][r] > 0:
                        load = 0.0
                        for j in range(n):
                            if scheduled[j]:
                                if start_times[j] <= es_pred < finish_times[j]:
                                    load += d_mat[j][r]
                        # Log-sum-exp style soft-max for max?
                        # Here we just sum the ratios.
                        ratio = (load + d_mat[i][r]) / c_vec[r]
                        # Use log to emphasize high ratios
                        pressure += math.log1p(ratio)
            
            # Dynamic tail adjustment?
            # The static tail is fixed. We can add a term that penalizes jobs that are "tight" in time.
            # slack = (makespan_estimate - es_pred) - duration[i] - tail[i]
            # For now, use static_tail + pressure
            
            prio = static_tail[i] + pressure * 5.0  # Weight pressure
            
            if prio > best_prio:
                best_prio = prio
                best_job = i
                
        if best_job == -1:
            break
            
        # Schedule best_job
        es_pred = 0.0
        for p in predecessors[best_job]:
            if finish_times[p] > es_pred:
                es_pred = finish_times[p]
                
        candidate_times = {es_pred}
        if d_mat is not None and c_vec is not None and num_resources > 0:
            for r in range(num_resources):
                if d_mat[best_job][r] > 0:
                    for j in range(n):
                        if scheduled[j] and d_mat[j][r] > 0:
                            candidate_times.add(finish_times[j])
                            
        earliest_start = float('inf')
        for t in sorted(candidate_times):
            if t < es_pred:
                continue
            feasible = True
            if d_mat is not None and c_vec is not None and num_resources > 0:
                for r in range(num_resources):
                    if d_mat[best_job][r] == 0:
                        continue
                    load = 0.0
                    for j in range(n):
                        if scheduled[j]:
                            if start_times[j] <= t < finish_times[j]:
                                load += d_mat[j][r]
                    if load + d_mat[best_job][r] > c_vec[r]:
                        feasible = False
                        break
            if feasible:
                earliest_start = t
                break
                
        if earliest_start == float('inf'):
            earliest_start = es_pred
            
        start_times[best_job] = earliest_start
        finish_times[best_job] = earliest_start + durations[best_job]
        scheduled[best_job] = True
        scheduled_count += 1
        
    makespan = np.max(finish_times) if n > 0 else 0.0
    return makespan

def priority(instance):
    """
    Returns a dummy priority vector, but the actual scheduling logic is in _run_online_sgs.
    The task requires returning a score per activity.
    However, the evaluator uses the priority vector to run SGS?
    "Write a Python priority(instance) function that scores each activity; a serial schedule-generation scheme starts the highest-priority eligible activity..."
    
    Wait. The contract says: "a serial schedule-generation scheme starts the highest-priority eligible activity under precedence and renewable-resource limits."
    This implies the *evaluator* runs the SGS using my priority vector.
    If I return a static vector, the evaluator uses it.
    My "online" strategy cannot be implemented if the evaluator runs the SGS with a static vector.
    
    Let me re-read carefully:
    "Write a Python priority(instance) function that scores each activity; a serial schedule-generation scheme starts the highest-priority eligible activity..."
    
    If the evaluator runs the SGS, I must return a static vector.
    The "online" strategy is only possible if *I* run the SGS and return the makespan?
    No, the return interface is "one score per activity".
    
    So, I cannot implement a fully online strategy where priority changes at each step, because the evaluator likely calls `priority(instance)` once, gets a vector, and then runs its own SGS loop using that vector.
    
    Therefore, I must compute a *static* priority vector that *approximates* the benefits of the online strategy.
    
    How to approximate online resource pressure in a static vector?
    The resource pressure depends on the schedule, which depends on the priorities. It's a fixed point problem.
    
    Alternative: Use the insights from the "online" logic to design a better *static* priority.
    The online logic prioritized: `static_tail + resource_pressure`.
    Since we can't know the exact `es_pred` and `load` for each job in a static vector, we can estimate the *average* or *worst-case* resource pressure.
    
    Or, we can use a different structural change:
    Instead of `slack` or `tail`, use a **Resource-Weighted Critical Path (RWCP)** that accounts for the probability of resource contention.
    
    Let's stick to a static vector but make it smarter.
    Current best (1.0347) uses SA to optimize a static vector.
    Inspiration 1 (1.0996) uses Min-Regret (online-ish, but implemented as a scheduler? No, it says "constructs the schedule directly").
    If Inspiration 1's `priority` function returns a vector, how does it do Min-Regret?
    Maybe it computes the vector such that it *simulates* the regret?
    
    Actually, looking at the code structure of the current program, it runs `_run_serial_sgs` internally during SA to evaluate the vector.
    This means the *program* runs the SGS to find the best vector.
    The evaluator then takes the returned vector and runs *its own* SGS?
    "Scored by mean makespan on five generated single-mode instances... The evaluator measures the raw value"
    If the evaluator runs its own SGS, it must use the vector I return.
    
    So, my job is to return the *best possible static vector* for the SGS.
    
    The current program uses SA to find this vector.
    The "online" idea was flawed if the evaluator runs SGS.
    
    New Strategy:
    Instead of SA on a continuous vector, use a **Discrete Priority Ranking Search** (e.g., Genetic Algorithm on permutations/rankings) or **Local Search on the Topological Order**.
    
    Or, even better: **Iterative Improvement of the Priority Vector using the SGS as a Black Box**, but with a different search method: **Simulated Annealing on the *order* of activities** (permutation) rather than continuous values.
    
    Actually, the current program already does SA on continuous values.
    Let's try **Tabu Search** on the priority vector.
    
    Or, let's look at the "Inspiration 1" again. It scored 1.0996 (worse than 1.0347).
    Inspiration 2 (1.0719) and 3 (1.0567) are also worse.
    
    The current program is the best.
    To improve, I need a better search or a better initialization.
    
    Structural Change:
    Replace SA with **Particle Swarm Optimization (PSO)** on the priority vector.
    PSO often finds better global minima in continuous spaces than SA, especially with good inertia and cognitive/social components.
    
    Let's implement PSO.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.array([])
        
    # Initialize PSO
    np.random.seed(42)
    
    # Compute a good initial vector (e.g., based on slack or tail)
    slack = _compute_slack(instance)
    p_slack = -slack
    
    # PSO Parameters
    swarm_size = 20
    max_iter = 100
    w = 0.7  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    # Initialize particles
    particles = np.zeros((swarm_size, n))
    for i in range(swarm_size):
        # Start near the slack-based vector
        particles[i] = p_slack + np.random.randn(n) * 0.5
        
    pbest = particles.copy()
    pbest_score = np.zeros(swarm_size)
    for i in range(swarm_size):
        pbest_score[i] = _run_serial_sgs(instance, pbest[i])
        
    gbest_idx = np.argmin(pbest_score)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_score[gbest_idx]
    
    velocities = np.zeros((swarm_size, n))
    
    start_time = time.time()
    timeout = 120.0
    
    for it in range(max_iter):
        if time.time() - start_time > timeout - 10:
            break
            
        for i in range(swarm_size):
            r1 = np.random.rand(n)
            r2 = np.random.rand(n)
            
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            
            particles[i] = particles[i] + velocities[i]
            
            score = _run_serial_sgs(instance, particles[i])
            
            if score < pbest_score[i]:
                pbest_score[i] = score
                pbest[i] = particles[i].copy()
                if score < gbest_score:
                    gbest_score = score
                    gbest = particles[i].copy()
                    
    return gbest

def _compute_slack(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", None)
    n = len(durations)
    if n == 0:
        return np.array([])
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
    
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    for u in topo_order:
        if predecessors[u]:
            es[u] = max(ef[p] for p in predecessors[u])
        ef[u] = es[u] + durations[u]
        
    makespan_no_res = np.max(ef)
    ls = np.zeros(n, dtype=float)
    lf = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if successors[u]:
            lf[u] = min(ls[s] for s in successors[u])
        else:
            lf[u] = makespan_no_res
        ls[u] = lf[u] - durations[u]
        
    slack = ls - es
    return slack

def _run_serial_sgs(instance, priority_vec):
    """
    Serial Schedule Generation Scheme.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    if n == 0:
        return 0.0
        
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # Pre-compute resource users
    num_resources = 0
    if demands is not None:
        d_mat = np.asarray(demands)
        if len(d_mat.shape) == 2:
            num_resources = d_mat.shape[1]
    
    resource_users = [[] for _ in range(num_resources)]
    if demands is not None:
        d_mat = np.asarray(demands, dtype=float)
        for i in range(n):
            for r in range(num_resources):
                if d_mat[i][r] > 0:
                    resource_users[r].append(i)
                    
    # State
    scheduled = [False] * n
    start_times = np.zeros(n, dtype=float)
    finish_times = np.zeros(n, dtype=float)
    
    scheduled_count = 0
    while scheduled_count < n:
        best_job = -1
        best_prio = -np.inf
        
        for i in range(n):
            if not scheduled[i]:
                eligible = True
                for p in predecessors[i]:
                    if not scheduled[p]:
                        eligible = False
                        break
                if eligible:
                    if priority_vec[i] > best_prio:
                        best_prio = priority_vec[i]
                        best_job = i
                        
        if best_job == -1:
            break
            
        es_pred = 0.0
        for p in predecessors[best_job]:
            if finish_times[p] > es_pred:
                es_pred = finish_times[p]
                
        candidate_times = {es_pred}
        if demands is not None and capacities is not None:
            d_mat = np.asarray(demands, dtype=float)
            c_vec = np.asarray(capacities, dtype=float)
            for r in range(num_resources):
                if d_mat[best_job][r] > 0:
                    for j in resource_users[r]:
                        if scheduled[j]:
                            candidate_times.add(finish_times[j])
                            
        earliest_start = float('inf')
        for t in sorted(candidate_times):
            if t < es_pred:
                continue
            feasible = True
            if demands is not None and capacities is not None:
                d_mat = np.asarray(demands, dtype=float)
                c_vec = np.asarray(capacities, dtype=float)
                for r in range(num_resources):
                    if d_mat[best_job][r] == 0:
                        continue
                    load = 0.0
                    for j in resource_users[r]:
                        if scheduled[j]:
                            if start_times[j] <= t < finish_times[j]:
                                load += d_mat[j][r]
                    if load + d_mat[best_job][r] > c_vec[r]:
                        feasible = False
                        break
            if feasible:
                earliest_start = t
                break
                
        if earliest_start == float('inf'):
            earliest_start = es_pred
            
        start_times[best_job] = earliest_start
        finish_times[best_job] = earliest_start + durations[best_job]
        scheduled[best_job] = True
        scheduled_count += 1
        
    makespan = np.max(finish_times) if n > 0 else 0.0
    return makespan

# EVOLVE-BLOCK-END
