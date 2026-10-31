import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 118.0  # Safety margin for 120s limit
    
    # Total differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Helper to compute initial greedy solution
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        break
                    x += 1
                row.append(x)
                current_max = x
                used_diffs |= new_diffs
            rows.append(row)
        return rows

    # State variables
    rows = greedy_init()
    
    # Build diff_counts and conflict_count
    diff_counts = {}
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
    conflict_count = sum(1 for v in diff_counts.values() if v > 1)
    
    def get_scope():
        return max(max(r) for r in rows)
    
    current_scope = get_scope()
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    best_conflicts = conflict_count
    
    # Cost function
    CONFLICT_WEIGHT = 100000
    
    def get_cost(conflicts, scope):
        return conflicts * CONFLICT_WEIGHT + scope
    
    current_cost = get_cost(conflict_count, current_scope)
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    min_T = 0.01
    
    # Track no-improvement steps for iterated local search kicks
    no_improve_steps = 0
    no_improve_limit = 5000
    kick_size = 5
    
    while time.time() < deadline - 2.0:
        # Pick a random row and position
        r_idx = random.randint(0, n_rows - 1)
        p_idx = random.randint(1, k)  # Position 1 to k
        
        row = rows[r_idx]
        old_val = row[p_idx]
        
        # Determine range for new value
        min_val = row[p_idx-1] + 1
        max_val = row[p_idx+1] - 1 if p_idx < k else current_scope + 50
        
        # Bias towards smaller values for optimization
        if conflict_count == 0:
            # In optimization mode, prefer smaller values
            if min_val < old_val:
                # 70% chance to try shrinking, 30% to explore
                if random.random() < 0.7:
                    new_val = random.randint(min_val, old_val)
                else:
                    new_val = random.randint(min_val, max_val)
            else:
                new_val = random.randint(min_val, max_val)
        else:
            # In feasibility mode, random exploration
            new_val = random.randint(min_val, max_val)
        
        if new_val == old_val:
            continue
        
        # Calculate change in differences
        removed_diffs = []
        added_diffs = []
        
        # Differences with elements to the left
        for i in range(p_idx):
            removed_diffs.append(old_val - row[i])
            added_diffs.append(new_val - row[i])
            
        # Differences with elements to the right
        for j in range(p_idx + 1, len(row)):
            removed_diffs.append(row[j] - old_val)
            added_diffs.append(row[j] - new_val)
        
        # Simulate update
        temp_conflicts = conflict_count
        
        # Apply removals
        for d in removed_diffs:
            old_c = diff_counts.get(d, 0)
            new_c = old_c - 1
            if new_c == 0:
                del diff_counts[d]
            else:
                diff_counts[d] = new_c
            if old_c == 2 and new_c == 1:
                temp_conflicts -= 1
                
        # Apply additions
        for d in added_diffs:
            old_c = diff_counts.get(d, 0)
            new_c = old_c + 1
            diff_counts[d] = new_c
            if old_c == 1 and new_c == 2:
                temp_conflicts += 1
        
        # Calculate new scope
        row[p_idx] = new_val
        new_row_max = max(row)
        
        # Calculate new global scope
        new_global_scope = 0
        for idx, r in enumerate(rows):
            if idx == r_idx:
                m = new_row_max
            else:
                m = max(r)
            if m > new_global_scope:
                new_global_scope = m
        
        new_cost = get_cost(temp_conflicts, new_global_scope)
        
        # Acceptance criterion with simulated annealing
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or (T > min_T and random.random() < math.exp(-delta_cost / T)):
            # Accept
            conflict_count = temp_conflicts
            current_scope = new_global_scope
            current_cost = new_cost
            no_improve_steps = 0
            
            if conflict_count == 0 and current_scope < best_scope:
                best_scope = current_scope
                best_conflicts = 0
                best_rows = [r[:] for r in rows]
        else:
            # Reject: Rollback
            row[p_idx] = old_val
            
            # Undo additions
            for d in added_diffs:
                c = diff_counts[d]
                if c == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] = c - 1
                if c == 2:
                    conflict_count -= 1
                    
            # Undo removals
            for d in removed_diffs:
                c = diff_counts.get(d, 0)
                diff_counts[d] = c + 1
                if c == 1:
                    conflict_count += 1
                    
            no_improve_steps += 1
        
        T *= cooling
        
        # Apply kick if stuck
        if no_improve_steps > no_improve_limit:
            num_kicks = kick_size
            for _ in range(num_kicks):
                kr_idx = random.randint(0, n_rows - 1)
                kp_idx = random.randint(1, k)
                krow = rows[kr_idx]
                kmin = krow[kp_idx-1] + 1
                kmax = krow[kp_idx+1] - 1 if kp_idx < k else current_scope + 20
                if kmin < kmax:
                    knew = random.randint(kmin, kmax)
                    if knew != krow[kp_idx]:
                        # Apply kick move
                        old_kv = krow[kp_idx]
                        rem_d = []
                        add_d = []
                        for i in range(kp_idx):
                            rem_d.append(old_kv - krow[i])
                            add_d.append(knew - krow[i])
                        for j in range(kp_idx + 1, len(krow)):
                            rem_d.append(krow[j] - old_kv)
                            add_d.append(krow[j] - knew)
                        
                        for d in rem_d:
                            oc = diff_counts.get(d, 0)
                            nc = oc - 1
                            if nc == 0:
                                del diff_counts[d]
                            else:
                                diff_counts[d] = nc
                            if oc == 2 and nc == 1:
                                conflict_count -= 1
                                
                        for d in add_d:
                            oc = diff_counts.get(d, 0)
                            nc = oc + 1
                            diff_counts[d] = nc
                            if oc == 1 and nc == 2:
                                conflict_count += 1
                                
                        krow[kp_idx] = knew
                        
                        new_rm = max(krow)
                        ng_scope = max([max(r) for i, r in enumerate(rows) if i != kr_idx] + [new_rm])
                        current_scope = ng_scope
                        current_cost = get_cost(conflict_count, current_scope)
                        
                        if conflict_count == 0 and current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                            
            no_improve_steps = 0
            T = max(T, 50.0)  # Reset temperature after kick

    # Final polish: Try to reduce scope greedily
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for r_idx in range(n_rows):
            row = rows[r_idx]
            for p_idx in range(k, 0, -1):
                old_val = row[p_idx]
                min_val = row[p_idx-1] + 1
                if min_val >= old_val:
                    continue
                # Try to find a valid smaller value
                for new_val in range(old_val - 1, min_val - 1, -1):
                    rem_d = []
                    add_d = []
                    for i in range(p_idx):
                        rem_d.append(old_val - row[i])
                        add_d.append(new_val - row[i])
                    for j in range(p_idx + 1, len(row)):
                        rem_d.append(row[j] - old_val)
                        add_d.append(row[j] - new_val)
                    
                    # Check if this move is valid
                    valid = True
                    for d in add_d:
                        if d in {x for x in rem_d} or diff_counts.get(d, 0) >= 1:
                            # Need to check more carefully
                            pass
                    
                    # Simpler: just try and see if conflicts decrease or stay same
                    temp_conf = conflict_count
                    for d in rem_d:
                        oc = diff_counts.get(d, 0)
                        nc = oc - 1
                        if nc == 0:
                            del diff_counts[d]
                        else:
                            diff_counts[d] = nc
                        if oc == 2 and nc == 1:
                            temp_conf -= 1
                    for d in add_d:
                        oc = diff_counts.get(d, 0)
                        nc = oc + 1
                        diff_counts[d] = nc
                        if oc == 1 and nc == 2:
                            temp_conf += 1
                    
                    if temp_conf <= conflict_count:
                        # Accept this improvement
                        row[p_idx] = new_val
                        conflict_count = temp_conf
                        current_scope = get_scope()
                        current_cost = get_cost(conflict_count, current_scope)
                        improved = True
                        if conflict_count == 0 and current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                        break
                    else:
                        # Rollback
                        for d in add_d:
                            c = diff_counts[d]
                            if c == 1:
                                del diff_counts[d]
                            else:
                                diff_counts[d] = c - 1
                            if c == 2:
                                conflict_count -= 1
                        for d in rem_d:
                            c = diff_counts.get(d, 0)
                            diff_counts[d] = c + 1
                            if c == 1:
                                conflict_count += 1
                        break

    return best_rows

# Test locally
if __name__ == "__main__":
    rows = construct(4, 7)
    print(rows)
    # Verify
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
            break
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
        if not valid:
            break
    print("Valid:", valid)
    print("Scope:", max(max(r) for r in rows))
