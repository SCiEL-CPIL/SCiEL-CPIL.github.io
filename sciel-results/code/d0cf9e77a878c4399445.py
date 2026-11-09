import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    N = 284
    
    def get_conflicts_from_counts(counts):
        c = 0
        for val in counts.values():
            if val > 1:
                c += val * (val - 1) // 2
        return c

    def is_valid_sidon(marks):
        if len(marks) != len(set(marks)):
            return False
        if any(m < 1 or m > N for m in marks):
            return False
        if not prescribed_set.issubset(set(marks)):
            return False
        
        diff_count = {}
        n = len(marks)
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(marks[i] - marks[j])
                if d in diff_count:
                    return False
                diff_count[d] = 1
        return True

    def solve_for_size(k):
        """
        Try to find a valid Sidon set of size k using Simulated Annealing.
        """
        if k < len(prescribed):
            return None
            
        # Initialize with prescribed elements and random others
        marks = list(prescribed)
        pool = [x for x in range(1, N + 1) if x not in prescribed_set]
        random.shuffle(pool)
        marks.extend(pool[:k - len(prescribed)])
        
        # Check for immediate duplicates (should not happen with pool logic, but safe)
        if len(marks) != len(set(marks)):
            return None

        # Build initial diff_count and current_conflicts
        diff_count = {}
        n = len(marks)
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(marks[i] - marks[j])
                diff_count[d] = diff_count.get(d, 0) + 1
        
        current_conflicts = get_conflicts_from_counts(diff_count)
        
        if current_conflicts == 0:
            return marks
            
        best_conflicts = current_conflicts
        best_marks = list(marks)
        
        # Simulated Annealing parameters
        T = 5.0
        cooling = 0.99995
        min_T = 0.001
        
        steps = 0
        max_steps = 2000000 # High limit, but time-checked
        
        # Indices of non-prescribed marks for faster random selection
        movable_indices = [i for i in range(k) if marks[i] not in prescribed_set]
        
        while T > min_T:
            if time.time() > deadline:
                break
                
            # Pick a random movable index
            if not movable_indices:
                break
            idx = random.choice(movable_indices)
            
            old_val = marks[idx]
            
            # Pick a new value
            # To ensure distinctness, we pick a random value and check if it's in the set
            new_val = random.randint(1, N)
            if new_val == old_val:
                continue
            if new_val in marks:
                continue
                
            # Calculate delta conflicts efficiently
            # We need to update diff_count for old_val and new_val
            
            # Store old counts for the differences involving old_val
            # Instead of recalculating all, we update the specific differences
            
            # 1. Remove old_val's contributions
            old_diffs = []
            for j in range(k):
                if j == idx:
                    continue
                d = abs(old_val - marks[j])
                old_diffs.append(d)
                
            # 2. Add new_val's contributions
            new_diffs = []
            for j in range(k):
                if j == idx:
                    continue
                d = abs(new_val - marks[j])
                new_diffs.append(d)
                
            # Apply changes to diff_count and track conflict change
            delta_conflicts = 0
            
            # Helper to update conflict count when a difference count changes
            def update_conflict_count(d, delta):
                nonlocal delta_conflicts
                old_count = diff_count.get(d, 0)
                new_count = old_count + delta
                if new_count == 0:
                    # Remove from dict
                    if d in diff_count:
                        del diff_count[d]
                    # Conflict change: from old_count*(old_count-1)/2 to 0
                    delta_conflicts -= old_count * (old_count - 1) // 2
                else:
                    # Update dict
                    diff_count[d] = new_count
                    # Conflict change: from old_count*(old_count-1)/2 to new_count*(new_count-1)/2
                    delta_conflicts += new_count * (new_count - 1) // 2 - old_count * (old_count - 1) // 2

            for d in old_diffs:
                update_conflict_count(d, -1)
                
            for d in new_diffs:
                update_conflict_count(d, 1)
                
            new_conflicts = current_conflicts + delta_conflicts
            
            # Acceptance criterion
            if new_conflicts <= current_conflicts or random.random() < math.exp(-(new_conflicts - current_conflicts) / T):
                # Accept move
                marks[idx] = new_val
                current_conflicts = new_conflicts
                
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_marks = list(marks)
                    
                if current_conflicts == 0:
                    return marks
            else:
                # Revert changes
                for d in new_diffs:
                    update_conflict_count(d, -1)
                for d in old_diffs:
                    update_conflict_count(d, 1)
                
            T *= cooling
            steps += 1
            
        if best_conflicts == 0:
            return best_marks
        return None

    # Try sizes from 20 down to 4
    # We prioritize 20, then 19.
    for k in range(20, len(prescribed) - 1, -1):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for larger sizes to increase chance of finding a solution
        # For k=20, it might be hard. Let's give it a good chunk of time.
        # If k=20 fails, try k=19.
        
        # We can run SA for a bit, if it doesn't find 0, restart with new random init?
        # Or just run one long SA.
        # Let's try one SA run per size, but with a generous step limit controlled by time.
        
        result = solve_for_size(k)
        if result is not None:
            # Verify one last time
            if is_valid_sidon(result):
                return result
            
    # Fallback: return the prescribed set if we run out of time or fail
    return prescribed

if __name__ == "__main__":
    res = construct()
    print(len(res))
