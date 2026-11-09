"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def check_valid(elements):
        if len(elements) != 12 or len(set(elements)) != 12:
            return False
        if any(x <= 0 for x in elements):
            return False
        sums = set()
        for mask in range(4096):
            s = 0
            for i in range(12):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums:
                return False
            sums.add(s)
        return True
    
    # Seed: powers of 2
    best_solution = [1 << i for i in range(12)]
    best_max = 2048
    
    # Quick local search to try to improve
    current = best_solution[:]
    
    while time.time() < deadline:
        improved = False
        # Try to reduce each element
        for i in range(12):
            if time.time() > deadline:
                break
            old_val = current[i]
            # Try smaller values
            for new_val in range(old_val - 1, 0, -1):
                if new_val in current:
                    continue
                trial = current[:]
                trial[i] = new_val
                trial.sort()
                if check_valid(trial):
                    current = trial
                    if max(current) < best_max:
                        best_max = max(current)
                        best_solution = current[:]
                    improved = True
                    break
            if time.time() > deadline:
                break
        if not improved:
            break
    
    return best_solution
# EVOLVE-BLOCK-END
