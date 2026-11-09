import numpy as np
import time as _time
from collections import defaultdict

_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = _time.time()
    return _time.time() - _start_time

def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick best machine for the single op
        best_m, best_d = min(op["machine_options"].items(), key=lambda x: x[1])
        return (0, best_m)
    
    # Bipartite graph approach: for each operation-machine pair, compute
    # the marginal benefit of assigning that operation to that machine.
    # Edge weight = reduction in critical path per unit of machine time.
    
    # First, gather all machines and their current loads (approximated by remaining work)
    machine_load = defaultdict(float)
    op_machines = []
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        op_machines.append(machine_options)
        if machine_options:
            # Approximate machine load contribution from this operation
            min_dur = min(machine_options.values())
            for m, d in machine_options.items():
                # Weight by how much this operation would contribute to machine m's load
                machine_load[m] += d / max(min_dur, 1e-9)
    
    # For each operation, compute the best machine assignment based on
    # marginal benefit: how much makespan reduction per unit time on that machine
    scores = np.zeros(n)
    best_machines = [None] * n
    
    # Normalize machine loads for comparison
    if machine_load:
        max_load = max(machine_load.values())
        min_load = min(machine_load.values())
        load_range = max_load - min_load if max_load > min_load else 1.0
    else:
        load_range = 1.0
    
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        if not machine_options:
            scores[i] = rem_work
            best_machines[i] = None
            continue
        
        best_score = float('-inf')
        best_m = None
        
        for m, d in machine_options.items():
            # Compute marginal benefit: 
            # 1. Work reduction: completing this op reduces remaining work by rem_work
            # 2. Machine efficiency: lower duration on this machine is better
            # 3. Load balancing: prefer machines with lower current load
            # 4. Critical path: ops with more remaining ops are more critical
            
            # Efficiency: work done per unit time on this machine
            efficiency = rem_work / max(d, 1e-9)
            
            # Load balancing: prefer less loaded machines
            m_load = machine_load.get(m, 0.0)
            load_factor = 1.0 - (m_load / max_load) if max_load > 0 else 1.0
            
            # Criticality: ops that block more other ops are more important
            criticality = rem_ops
            
            # Combined edge weight: marginal makespan reduction per unit time
            # Higher efficiency and lower load are better
            edge_weight = efficiency * 0.4 + load_factor * 0.3 + criticality * 0.2 + (1.0 / max(d, 1e-9)) * 0.1
            
            if edge_weight > best_score:
                best_score = edge_weight
                best_m = m
        
        scores[i] = best_score
        best_machines[i] = best_m
    
    best_idx = int(np.argmax(scores))
    return (best_idx, best_machines[best_idx])
