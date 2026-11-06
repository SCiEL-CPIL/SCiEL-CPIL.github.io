"""Optimized dispatch rule for open-shop scheduling with load-balancing water-filling."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

# Global state for tracking simulation progress
_sim_state = {}
_DEADLINE = 110.0
_start_time = None

def _reset_sim_state():
    global _sim_state, _start_time
    _sim_state = {
        'in_progress': {},          # (job, op) -> (machine, completion_time)
        'machine_free_at': defaultdict(float),  # machine -> time when machine becomes free
        'machine_load': defaultdict(float),     # machine -> total work assigned
        'call_count': 0,
        'last_time': 0.0,
        'total_machines_seen': set(),
        'machine_queue_len': defaultdict(int),  # machine -> number of ops queued
    }
    _start_time = time.time()

def _check_time():
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE

def _soft_max(values, temp=1.0):
    """Smooth maximum via log-sum-exp trick."""
    if not values:
        return 0.0
    v = max(values)
    if temp <= 0:
        return v
    exp_sum = sum(math.exp((x - v) / temp) for x in values)
    return v + temp * math.log(exp_sum)

def _soft_min(values, temp=1.0):
    """Smooth minimum via log-sum-exp trick."""
    if not values:
        return 0.0
    v = min(values)
    if temp <= 0:
        return v
    exp_sum = sum(math.exp((v - x) / temp) for x in values)
    return v - temp * math.log(exp_sum)

def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Load-Balancing Water-Filling with SPT machine selection
    - Track in-progress operations and machine availability
    - For job selection: prioritize jobs with most remaining work (MWR)
    - For machine selection: pick machine that minimizes makespan impact
      using soft-max of resulting machine loads
    - Prioritize operations that minimally skew the global workload distribution
    
    Returns: (index, machine) tuple
    """
    # Initialize on first call
    if _start_time is None:
        _reset_sim_state()
    
    if not ready_operations:
        return (0, None)
    
    # Time guard
    if _check_time():
        best_idx = max(range(len(ready_operations)), 
                      key=lambda i: ready_operations[i]["remaining_work"])
        op = ready_operations[best_idx]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        else:
            best_machine = None
        return (best_idx, best_machine)
    
    state = _sim_state
    state['call_count'] += 1
    
    # Update state: check if any in-progress operations have completed
    completed_keys = []
    for key in list(state['in_progress'].keys()):
        machine, comp_time = state['in_progress'][key]
        if time >= comp_time:
            completed_keys.append(key)
    
    for key in completed_keys:
        machine, comp_time = state['in_progress'][key]
        del state['in_progress'][key]
        # Machine is now free at comp_time (which is <= current time)
        # Update machine_free_at to be at most current time
        if state['machine_free_at'][machine] > time:
            state['machine_free_at'][machine] = time
        # Decrement queue length
        if state['machine_queue_len'][machine] > 0:
            state['machine_queue_len'][machine] -= 1
    
    # Update last_time
    state['last_time'] = time
    
    # Get max remaining work for normalization
    max_work = max(op["remaining_work"] for op in ready_operations)
    if max_work == 0:
        max_work = 1
    
    # Get max remaining ops for normalization
    max_ops = max(op["remaining_ops"] for op in ready_operations)
    if max_ops == 0:
        max_ops = 1
    
    # Get current machine loads
    all_machines = state['total_machines_seen']
    current_loads = {m: state['machine_load'].get(m, 0.0) for m in all_machines}
    current_max_load = max(current_loads.values()) if current_loads else 0.0
    if current_max_load == 0:
        current_max_load = 1
    
    # Compute global load statistics for water-filling
    if current_loads:
        load_values = list(current_loads.values())
        mean_load = sum(load_values) / len(load_values)
        load_var = sum((x - mean_load) ** 2 for x in load_values) / len(load_values)
        load_std = math.sqrt(load_var) if load_var > 0 else 1.0
    else:
        mean_load = 0.0
        load_std = 1.0
    
    scores = []
    best_machines = []
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        job = op.get('job')
        
        machine_options = op.get("machine_options", {})
        if not machine_options:
            scores.append(0)
            best_machines.append(None)
            continue
        
        num_machines = len(machine_options)
        
        # Component 1: Most Work Remaining (primary) - normalized
        work_score = remaining_work / max_work
        
        # Component 2: Remaining operations (secondary) - normalized  
        ops_score = remaining_ops / max_ops
        
        # Component 3: Urgency - if few ops remain, prioritize to avoid idle time
        urgency_score = 1.0 / (remaining_ops + 1.0) if remaining_ops > 0 else 1.0
        
        # Component 4: Load-Balancing Water-Filling Score
        # For each candidate machine, compute the resulting load if this op is assigned there
        # Use soft-max of resulting machine loads to measure makespan impact
        machine_results = []
        for m, d in machine_options.items():
            # Current load of machine m
            current_load = state['machine_load'].get(m, 0.0)
            # New load if we assign this op to machine m
            new_load = current_load + d
            
            # Simulate the resulting machine loads
            simulated_loads = dict(current_loads)
            simulated_loads[m] = new_load
            
            # Use soft-max to get a smooth measure of makespan
            soft_max_load = _soft_max(list(simulated_loads.values()), temp=0.5)
            
            # Also consider when the machine becomes free
            free_at = state['machine_free_at'].get(m, 0.0)
            # Effective start time is max(current_time, free_at)
            # But for load balancing, we care about total load
            # Add a small penalty if the machine is busy (will cause delay)
            delay_penalty = max(0.0, free_at - time) * 0.1
            
            # Queue length penalty: machines with more queued work are more congested
            queue_penalty = state['machine_queue_len'].get(m, 0) * 0.05
            
            # Water-filling smoothness: penalize deviation from mean load
            # This encourages balanced distribution
            deviation = abs(new_load - mean_load) / load_std
            waterfill_penalty = deviation * 0.03
            
            cost = soft_max_load + delay_penalty + queue_penalty + waterfill_penalty
            machine_results.append((cost, m, d, new_load))
        
        # Find the best machine (minimum cost)
        best_machine_result = min(machine_results, key=lambda x: x[0])
        best_machine = best_machine_result[1]
        best_machine_cost = best_machine_result[0]
        best_new_load = best_machine_result[3]
        
        # Normalize machine score: lower cost is better
        max_cost = max(r[0] for r in machine_results)
        min_cost = min(r[0] for r in machine_results)
        if max_cost > min_cost:
            machine_score = 1.0 - (best_machine_cost - min_cost) / (max_cost - min_cost)
        else:
            machine_score = 0.5
        machine_score = max(0.0, min(1.0, machine_score))
        
        # Component 5: Flexibility bonus - more options is slightly better
        flex_score = min(num_machines / 5.0, 1.0)
        
        # Component 6: Blocking/criticality detection
        criticality_bonus = 0.0
        if num_machines <= 1:
            criticality_bonus = 0.25
        elif num_machines == 2:
            criticality_bonus = 0.10
        
        # Near completion bonus
        if remaining_ops <= 1:
            criticality_bonus += 0.20
        
        # Component 7: Water-filling smoothness
        smoothness_score = machine_score  # Already computed above
        
        # Weighted combination
        w_work = 0.45
        w_ops = 0.10
        w_urgency = 0.10
        w_machine = 0.20
        w_flex = 0.05
        w_critical = 0.10
        
        score = (w_work * work_score + 
                 w_ops * ops_score + 
                 w_urgency * urgency_score + 
                 w_machine * machine_score + 
                 w_flex * flex_score +
                 w_critical * criticality_bonus)
        
        scores.append(score)
        best_machines.append(best_machine)
    
    best_idx = max(range(len(ready_operations)), key=lambda i: scores[i])
    best_machine = best_machines[best_idx]
    
    # Update state: mark the selected operation as in-progress
    selected_op = ready_operations[best_idx]
    job = selected_op.get('job')
    op_id = selected_op.get('op', best_idx)
    machine_options = selected_op.get("machine_options", {})
    
    if best_machine is not None and best_machine in machine_options:
        duration = machine_options[best_machine]
        free_at = state['machine_free_at'].get(best_machine, 0.0)
        start_time = max(time, free_at)
        completion_time = start_time + duration
        
        state['in_progress'][(job, op_id)] = (best_machine, completion_time)
        state['machine_free_at'][best_machine] = completion_time
        state['machine_load'][best_machine] += duration
        state['total_machines_seen'].add(best_machine)
        state['machine_queue_len'][best_machine] += 1
    
    return (best_idx, best_machine)

# EVOLVE-BLOCK-END
