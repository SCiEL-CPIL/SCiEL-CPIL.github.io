import math
import time
import random
import numpy as np
from scipy.spatial.distance import cdist

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin within 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = cdist(coords, coords)
    
    customers = list(range(1, n_cust + 1))
    
    # Helper to evaluate a route: returns (valid, distance, time)
    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0, 0.0
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        for node in route_nodes:
            d = dist[prev, node]
            t += d / speed
            b -= d
            if b < -_EPS:
                return False, 0.0, 0.0
            
            # Charging station logic
            if node in stations:
                # If we are at a station, we can charge.
                # The problem states: "a station adds (battery - energy) * recharge_rate time and refills"
                # Usually this implies we charge to full.
                # "refills" -> battery becomes battery_cap.
                # Time added: (battery_cap - b) * recharge_rate? 
                # Or is it (battery_cap - current_b) / recharge_rate?
                # Contract: "recharge_rate 0.5". "station adds (battery - energy) * recharge_rate time"
                # This phrasing is tricky. "battery - energy" usually means capacity - current.
                # Let's assume it charges to full.
                # Time cost = (battery_cap - b) * recharge_rate? 
                # If recharge_rate is 0.5, and we need 100 units, time = 50? That's fast.
                # Or is it 1/recharge_rate? 
                # Let's look at the phrasing: "adds (battery - energy) * recharge_rate time".
                # If battery=300, energy=100 (current), diff=200. Time = 200 * 0.5 = 100.
                # This seems like a reasonable interpretation.
                time_to_charge = (battery_cap - b) * recharge_rate
                t += time_to_charge
                b = battery_cap
            else:
                # Customer logic
                t = max(t, ready[node])
                if t > due[node] + _EPS:
                    return False, 0.0, 0.0
                t += service[node]
                load += demands[node]
                if load > capacity + _EPS:
                    return False, 0.0, 0.0
            
            d_total += d
            prev = node
        
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0, 0.0
        if t > horizon + _EPS:
            return False, 0.0, 0.0
        d_total += d
        return True, d_total, t

    def evaluate_routes(routes):
        """Returns (valid, total_distance) for a list of routes."""
        total_d = 0.0
        for r in routes:
            if not r: continue
            valid, d, _ = check_route(r)
            if not valid:
                return False, 0.0
            total_d += d
        return True, total_d

    # --- Strategy: Sweep-Based Construction + LNS with SA ---

    best_overall_routes = None
    best_overall_dist = float('inf')
    
    # Initial simple valid solution: one customer per route
    fallback_routes = [[c] for c in customers]
    valid, dist = evaluate_routes(fallback_routes)
    if valid:
        best_overall_routes = [list(r) for r in fallback_routes]
        best_overall_dist = dist

    # 1. Constructive Heuristic: Sweep + Nearest Neighbor Insertion
    # Calculate angles from depot
    depot_x, depot_y = coords[0]
    angles = np.arctan2(coords[1:n_cust+1, 1] - depot_y, coords[1:n_cust+1, 0] - depot_x)
    
    # Try different sweep angles to start the partition
    best_construction_dist = float('inf')
    best_construction_routes = None
    
    for sweep_start in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        if time.time() > DEADLINE - 30: break
        
        # Rotate angles
        rotated_angles = (angles - math.radians(sweep_start)) % (2 * math.pi)
        
        # Sort customers by angle
        sorted_indices = np.argsort(rotated_angles)
        
        # Partition into routes based on capacity and battery constraints
        # We don't know the optimal number of routes, so we greedily fill them.
        routes = []
        current_route = []
        current_load = 0
        current_dist = 0
        current_battery = battery_cap
        current_time = 0
        
        # Simple greedy fill: add customer if it fits capacity, battery, and time window
        # This is a simplification. A better way is to cluster by angle into fixed size groups
        # and then optimize order.
        
        # Let's try grouping by angle into chunks that likely fit.
        # Avg demand 15, Cap 100 -> ~6 customers.
        # Let's try to form routes of ~5 customers based on angle proximity.
        
        # Alternative: Just sort and split into K routes based on angle, then optimize order.
        # How many routes? Estimate based on total demand / capacity.
        total_demand = np.sum(demands[1:n_cust+1])
        est_routes = max(4, int(np.ceil(total_demand / capacity)) + 1)
        
        # Split sorted customers into est_routes groups
        group_size = n_cust // est_routes
        remainder = n_cust % est_routes
        
        temp_routes = []
        idx = 0
        for i in range(est_routes):
            size = group_size + (1 if i < remainder else 0)
            if size == 0: continue
            group_customers = [sorted_indices[idx + j] + 1 for j in range(size)]
            idx += size
            
            # Order customers in this group using Nearest Neighbor
            route = []
            unvisited = list(group_customers)
            curr = 0
            while unvisited:
                # Find nearest unvisited
                d_to_unvisited = dist[curr, unvisited]
                min_idx = np.argmin(d_to_unvisited)
                next_node = unvisited[min_idx]
                route.append(next_node)
                unvisited.pop(min_idx)
                curr = next_node
            
            temp_routes.append(route)
        
        # Evaluate
        valid, d = evaluate_routes(temp_routes)
        if valid and d < best_construction_dist:
            best_construction_dist = d
            best_construction_routes = [list(r) for r in temp_routes]

    if best_construction_routes:
        best_overall_routes = best_construction_routes
        best_overall_dist = best_construction_dist

    # 2. Large Neighborhood Search (LNS) with Simulated Annealing
    # We will iteratively destroy a part of the solution and repair it.
    
    current_routes = [list(r) for r in best_overall_routes]
    current_dist = best_overall_dist
    
    # SA Parameters
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.995
    
    rng = random.Random(42)
    
    while time.time() < DEADLINE - 5:
        if not current_routes:
            break
            
        # --- Destroy ---
        # Randomly select a route and remove some customers
        # Or select a random set of customers to remove
        n_routes = len(current_routes)
        if n_routes == 0: break
        
        # Choose a strategy: 
        # 1. Remove all customers from one random route
        # 2. Remove a random subset of customers from a random route
        # 3. Remove customers from multiple routes
        
        strategy = rng.random()
        
        removed_customers = []
        new_routes = [list(r) for r in current_routes]
        
        if strategy < 0.5:
            # Remove all from one route
            r_idx = rng.randrange(n_routes)
            if new_routes[r_idx]:
                removed_customers = new_routes[r_idx]
                new_routes[r_idx] = []
        else:
            # Remove random subset from one route
            r_idx = rng.randrange(n_routes)
            r = new_routes[r_idx]
            if len(r) > 1:
                k = rng.randint(1, len(r) - 1) # Keep at least 1
                indices_to_remove = rng.sample(range(len(r)), k)
                # Sort indices to remove safely
                indices_to_remove.sort(reverse=True)
                for idx in indices_to_remove:
                    removed_customers.append(r.pop(idx))
        
        if not removed_customers:
            continue
            
        # Remove empty routes temporarily for repair
        active_routes = [r for r in new_routes if r]
        
        # --- Repair ---
        # Insert removed customers into the best position in any route
        # Using a greedy heuristic with some randomness (SA)
        
        repair_dist = 0
        repair_valid = True
        
        # Calculate current distance of active routes
        # To speed up, we can just re-evaluate after insertion, but that's slow.
        # Let's try inserting one by one.
        
        # Shuffle removed customers
        rng.shuffle(removed_customers)
        
        for cust in removed_customers:
            best_insertion_cost = float('inf')
            best_route_idx = -1
            best_pos = -1
            best_new_routes = None
            best_new_dist = float('inf')
            
            # Try inserting into each existing route
            for r_idx, r in enumerate(active_routes):
                # Try all positions
                for pos in range(len(r) + 1):
                    trial_route = r[:pos] + [cust] + r[pos:]
                    # Check validity of this route
                    v, d_route, _ = check_route(trial_route)
                    if not v:
                        continue
                    
                    # Estimate total distance change
                    # Old distance of r + New distance of trial_route - Old distance of r
                    # We need old distance of r. 
                    # For simplicity, just evaluate total distance of all routes after this insertion?
                    # That is O(N^2) per insertion. 
                    # Better: calculate delta distance.
                    
                    # Delta distance for insertion at pos in r:
                    # d(prev, next) -> d(prev, cust) + d(cust, next)
                    # If pos == 0: d(depot, r[0]) -> d(depot, cust) + d(cust, r[0])
                    # If pos == len(r): d(r[-1], depot) -> d(r[-1], cust) + d(cust, depot)
                    
                    if pos == 0:
                        old_seg = dist[0, r[0]]
                        new_seg = dist[0, cust] + dist[cust, r[0]]
                    elif pos == len(r):
                        old_seg = dist[r[-1], 0]
                        new_seg = dist[r[-1], cust] + dist[cust, 0]
                    else:
                        prev_node = r[pos-1]
                        next_node = r[pos]
                        old_seg = dist[prev_node, next_node]
                        new_seg = dist[prev_node, cust] + dist[cust, next_node]
                    
                    delta_d = new_seg - old_seg
                    
                    # We need the base distance of the current solution to compare.
                    # Let's just store the candidate and evaluate fully at the end?
                    # No, we need to pick the best one.
                    # Let's assume the rest of the routes stay the same.
                    # We need the current total distance of `active_routes`.
                    # Let's calculate it once outside the loop? 
                    # But `active_routes` changes as we insert.
                    
                    # This greedy approach is tricky with exact distance tracking.
                    # Let's just evaluate the full solution for the best candidate.
                    # Since N is small (25), this might be okay if we limit candidates.
                    
                    # Actually, let's just pick the one with minimum delta_d, assuming validity.
                    # This is an approximation.
                    
                    if delta_d < best_insertion_cost:
                        best_insertion_cost = delta_d
                        best_route_idx = r_idx
                        best_pos = pos
            
            if best_route_idx != -1:
                # Perform insertion
                r = active_routes[best_route_idx]
                active_routes[best_route_idx] = r[:best_pos] + [cust] + r[best_pos:]
            else:
                # No valid insertion found. Create a new route with just this customer
                active_routes.append([cust])
        
        # Reconstruct full routes list
        candidate_routes = active_routes
        
        # Evaluate candidate
        valid, cand_dist = evaluate_routes(candidate_routes)
        
        if not valid:
            # Repair failed, skip
            continue
            
        # Acceptance Criterion (Simulated Annealing)
        delta = cand_dist - current_dist
        
        if delta < 0:
            # Improvement
            current_routes = candidate_routes
            current_dist = cand_dist
            
            if current_dist < best_overall_dist:
                best_overall_dist = current_dist
                best_overall_routes = [list(r) for r in current_routes]
        else:
            # Worse, accept with probability exp(-delta / T)
            if T > T_min:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_routes = candidate_routes
                    current_dist = cand_dist
        
        T *= cooling_rate
        
        # Periodic local search polish (2-Opt) to get out of SA's rough solutions
        if rng.random() < 0.1:
            for i in range(len(current_routes)):
                r = current_routes[i]
                if len(r) < 2: continue
                improved = True
                while improved:
                    improved = False
                    for j in range(len(r) - 1):
                        for k in range(j + 1, len(r)):
                            a = r[j]
                            b = r[j+1]
                            c = r[k]
                            d = r[k+1] if k+1 < len(r) else 0
                            
                            old_cost = dist[a, b] + dist[c, d]
                            new_cost = dist[a, c] + dist[b, d]
                            
                            if new_cost < old_cost - _EPS:
                                new_r = r[:j+1] + r[j+1:k+1][::-1] + r[k+1:]
                                v, d_new, _ = check_route(new_r)
                                if v:
                                    # Check total distance
                                    # We only changed this route.
                                    # Calculate old total and new total?
                                    # Just check if this route improved and is valid.
                                    # Since we are in a polish step, we assume it helps.
                                    current_routes[i] = new_r
                                    improved = True
                                    break
                        if improved: break

    # Final check
    if best_overall_routes is None:
        return fallback_routes
        
    return best_overall_routes
