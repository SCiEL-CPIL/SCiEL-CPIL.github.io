import numpy as np
import time
import random

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    Strategy: Greedy Critical Path Construction.
    We iteratively select the 'most critical' unscheduled activity based on 
    remaining path length (CPM-like) and assign it the next highest priority.
    This creates a discrete ranking that respects the network structure.
    """
    start_time = time.time()
    # Safety margin: 140s out of 150s limit
    deadline = start_time + 140.0
    
    # 1. Extract and process instance data
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)  # Shape (N, R)
    capacities = np.array(instance["capacities"], dtype=float) # Shape (R,)
    
    # Handle predecessors
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
             
    # Pre-compute base critical path (ignoring resources) for initialization
    # This helps in breaking ties or providing a strong initial structure
    in_degree = [len(p) for p in preds]
    q = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)
                
    # Calculate base remaining path length (backward pass)
    # remaining_path[i] = max duration from i to end
    remaining_path = np.zeros(N)
    for u in reversed(topo_order):
        max_succ_path = 0
        for v in succs[u]:
            if remaining_path[v] > max_succ_path:
                max_succ_path = remaining_path[v]
        remaining_path[u] = durations[u] + max_succ_path

    # 2. Greedy Criticality Ranking
    # We want to assign higher priority (larger number) to more critical tasks.
    # Criticality is dynamic. We simulate a "perfect" schedule step-by-step 
    # but instead of scheduling, we just rank who *should* go first at each step.
    
    # However, a pure greedy ranking might be myopic. 
    # Let's use a hybrid: 
    # 1. Start with base critical path length as a strong prior.
    # 2. Use a lightweight simulation to adjust ranks for resource conflicts.
    
    # For robustness and speed, let's try a simple iterative improvement on the ranking itself.
    # But the directive asks for a DIFFERENT family. 
    # "Greedy Critical Path Construction" implies:
    # Step k: Among all unscheduled activities whose predecessors are scheduled, 
    # pick the one with the longest remaining path to the end. Assign it rank N-k.
    
    # This is effectively the "Critical Path Method" priority rule, but we can enhance it.
    # Standard CPM priority: Longest path to end.
    # Let's implement a slightly smarter version that accounts for resource scarcity.
    
    # Actually, the most robust "different" approach is often **Random Restarts with Simulated Annealing on Permutations**.
    # But let's stick to the "Criticality" theme as it's structurally sound.
    
    # Let's implement a **Lookahead Greedy** ranking.
    # We maintain a set of "available" activities.
    # At each step, we look at all available activities.
    # We calculate a score: remaining_path[i] + resource_pressure[i].
    # resource_pressure is high if the activity uses scarce resources.
    
    # To keep it fast and deterministic enough for a 120s budget on 5 instances:
    # We will perform multiple passes of this greedy construction with different random tie-breakers or slight perturbations in the criticality calculation, and keep the best resulting makespan.
    
    def simulate_sgs(priorities):
        """
        Fast Serial SGS simulation.
        Returns makespan.
        """
        scheduled = np.zeros(N, dtype=bool)
        active = []  # List of (finish_time, activity_index)
        remaining_res = capacities.copy()
        time_current = 0.0
        makespan = 0.0
        
        # Pre-compute eligible check logic for speed
        unsched_preds = np.array([len(p) for p in preds])
        
        # Initial eligible
        eligible_set = set(i for i in range(N) if unsched_preds[i] == 0)
        
        while not np.all(scheduled):
            # If no eligible, wait
            if not eligible_set:
                if not active:
                    break # Deadlock/Error
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                still_active = []
                for f, idx in active:
                    if f > time_current:
                        still_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                        # Update eligibility of successors
                        for v in succs[idx]:
                            if not scheduled[v]:
                                unsched_preds[v] -= 1
                                if unsched_preds[v] == 0:
                                    eligible_set.add(v)
                active = still_active
                continue
            
            # Sort eligible by priority (descending)
            elig_list = list(eligible_set)
            elig_list.sort(key=lambda x: priorities[x], reverse=True)
            
            any_scheduled = False
            to_remove = []
            for i in elig_list:
                # Check resources
                if np.all(remaining_res >= demands[i]):
                    # Schedule
                    scheduled[i] = True
                    remaining_res -= demands[i]
                    finish_t = time_current + durations[i]
                    active.append((finish_t, i))
                    makespan = max(makespan, finish_t)
                    any_scheduled = True
                    to_remove.append(i)
                    
            # Update eligible set
            if to_remove:
                for i in to_remove:
                    eligible_set.remove(i)
            
            if not any_scheduled:
                # Couldn't schedule anyone, wait
                if not active:
                    break
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                still_active = []
                for f, idx in active:
                    if f > time_current:
                        still_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                        for v in succs[idx]:
                            if not scheduled[v]:
                                unsched_preds[v] -= 1
                                if unsched_preds[v] == 0:
                                    eligible_set.add(v)
                active = still_active
        
        return makespan

    def generate_criticality_ranking(noise_seed):
        """
        Generates a priority vector based on greedy criticality.
        We simulate the scheduling process, but instead of committing, we rank
        the "best" candidate at each step. 
        """
        random.seed(noise_seed)
        
        # State for the greedy construction
        scheduled = np.zeros(N, dtype=bool)
        unsched_preds = np.array([len(p) for p in preds])
        eligible_set = set(i for i in range(N) if unsched_preds[i] == 0)
        
        priorities = np.zeros(N)
        current_rank = N
        
        # We need to track the "virtual" time and resources to make the criticality estimate better?
        # No, standard criticality is just path length. 
        # To make it "resource-aware", we can penalize activities that demand scarce resources.
        
        # Let's use a dynamic criticality:
        # crit[i] = remaining_path[i]
        # If we want to be smarter, we can look ahead 1 step.
        
        # Simple version: Rank by remaining_path[i] + random_noise
        # But let's try to incorporate resource scarcity.
        # Scarcity factor: sum(demands[i] / capacities)
        scarcity = np.sum(demands / capacities, axis=1)
        
        # Combine path length and scarcity
        # Weights: 0.8 * path + 0.2 * scarcity
        base_score = 0.8 * remaining_path + 0.2 * scarcity
        
        # Now, we perform a greedy selection to assign ranks.
        # This ensures that if two tasks have similar scores, the one that unblocks more successors 
        # (or just the one picked first in a valid sequence) gets higher rank.
        
        # Actually, just sorting by base_score is a static rule. 
        # The "Greedy Construction" implies we build the sequence.
        
        # Let's simulate a "perfect" scheduler that always picks the highest base_score eligible task.
        # The order in which they are picked becomes the priority rank.
        
        # This is equivalent to sorting by base_score, BUT it respects precedence.
        # If A and B are eligible, and A has higher score, A gets higher rank.
        # If B is not eligible until A finishes, B's rank is effectively lower.
        
        # So, we can just sort all nodes by base_score? No, that violates precedence in SGS.
        # SGS picks from *eligible*. 
        # If we assign priority = base_score, SGS will pick the highest base_score among eligible.
        # This is exactly the "Critical Path" priority rule.
        
        # To add variation and search, we perturb the base_score.
        
        perturbed_scores = base_score + np.random.normal(0, 1.0, N)
        
        # The priority vector is just these scores.
        # SGS will handle the precedence.
        
        return perturbed_scores

    # Main Loop: Try multiple random perturbations of the criticality scores.
    # This is effectively a Random Search over the "Criticality Weight" space.
    
    best_priorities = None
    best_score = float('inf')
    
    # Initial try with pure critical path
    init_prio = remaining_path.copy()
    score = simulate_sgs(init_prio)
    if score < best_score:
        best_score = score
        best_priorities = init_prio
    
    # Random Restarts
    # We have ~140s. 
    # Each simulation is O(N^2) roughly? N=30. Very fast.
    # We can do thousands of trials.
    
    num_trials = 2000
    
    for t in range(num_trials):
        if time.time() > deadline:
            break
            
        # Generate a new priority vector
        # Strategy: Mix of Critical Path, Scarcity, and Random Noise
        # Weight of CP: 0.5 to 1.0
        w_cp = random.uniform(0.5, 1.0)
        w_sc = random.uniform(0.0, 0.5)
        noise_scale = random.uniform(0.1, 2.0)
        
        scores = w_cp * remaining_path + w_sc * scarcity + np.random.normal(0, noise_scale, N)
        
        # Optional: Add a small bonus for activities with many successors (unblocking power)
        # succ_count = np.array([len(s) for s in succs])
        # scores += 0.1 * succ_count
        
        sim_score = simulate_sgs(scores)
        
        if sim_score < best_score:
            best_score = sim_score
            best_priorities = scores
            
            # If we found a very good one, we might want to explore around it?
            # For now, just keep the best.
            
    return best_priorities
