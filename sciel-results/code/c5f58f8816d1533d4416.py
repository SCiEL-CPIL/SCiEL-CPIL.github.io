"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 9
TOTAL = 1 << N

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a snake in Q_9 using backtracking with Warnsdorff's heuristic and forced move detection.
    """
    start_time = time.time()
    time_limit = 115  # Leave margin
    
    best_path = [0]
    best_len = 1
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(TOTAL)]
    for v in range(TOTAL):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    def solve(rng):
        nonlocal best_path, best_len, start_time
        
        path = [0]
        used = [False] * TOTAL
        used[0] = True
        block_count = [0] * TOTAL
        
        def get_valid_candidates(cur, used, block_count):
            cands = []
            for u in neighbors[cur]:
                if not used[u] and block_count[u] == 0:
                    cands.append(u)
            return cands

        def backtrack(depth):
            nonlocal best_path, best_len
            
            if time.time() - start_time > time_limit:
                return False
            
            if depth > best_len:
                best_len = depth
                best_path = list(path)
                
            cur = path[-1]
            candidates = get_valid_candidates(cur, used, block_count)
            
            if not candidates:
                return False

            # Heuristic: Warnsdorff's Rule
            # Score each candidate by the number of valid successors it would have.
            # We prefer candidates with FEWER successors (to avoid isolating them).
            scored_candidates = []
            for u in candidates:
                # Simulate move to u
                used[u] = True
                path.append(u)
                v_prev = path[-2]
                for w in neighbors[v_prev]:
                    if w != u:
                        block_count[w] += 1
                        
                # Count valid successors for u
                next_cands = get_valid_candidates(u, used, block_count)
                score = len(next_cands)
                
                # Revert simulation
                for w in neighbors[v_prev]:
                    if w != u:
                        block_count[w] -= 1
                path.pop()
                used[u] = False
                
                scored_candidates.append((score, u))
            
            # Sort by score ascending (Warnsdorff)
            scored_candidates.sort(key=lambda x: x[0])
            
            # If there's a forced move (score 0 is bad if it's a dead end, but wait:
            # If score is 0, it means u is a dead end. We should only take it if it's the *only* option or if we are finishing.
            # Actually, if score is 0 and there are other options, we should avoid it unless it's the last possible step?
            # No, if u has 0 successors, adding it ends the path.
            # If there are other candidates with score > 0, we should prefer them to extend the path.
            # So sorting by score ascending puts dead-ends first? That's bad.
            # Warnsdorff for Hamiltonian paths usually picks the one with fewest *remaining* neighbors.
            # If a vertex has 0 remaining neighbors, it's a dead end. We should pick it LAST?
            # No, if we pick it first, we stop. If we pick it last, we might get stuck elsewhere.
            # Actually, if a vertex has 0 neighbors, it's a terminal vertex.
            # If we have a choice between a vertex with 5 neighbors and one with 0, we should pick the one with 5?
            # No, Warnsdorff says pick the one with fewest options to keep the "open" vertices available for later.
            # But a vertex with 0 options *now* is a dead end.
            # Let's refine: If score == 0, it's a dead end.
            # If any candidate has score == 0, and it's not the only candidate, we should probably avoid it?
            # Or maybe we should prioritize candidates with score == 1 (forced)?
            
            # Let's try a different heuristic:
            # 1. If any candidate has score 0, and it's the only candidate, take it.
            # 2. If any candidate has score 0, and there are others, skip it (unless we are desperate).
            # 3. Prioritize candidates with score 1 (forced moves).
            # 4. Among others, pick randomly or by score.
            
            # Let's implement a simple priority:
            # Priority 0: Score 1 (forced)
            # Priority 1: Score > 1
            # Priority 2: Score 0 (dead end)
            
            def priority(item):
                score, u = item
                if score == 1:
                    return 0
                elif score > 1:
                    return 1
                else:
                    return 2
            
            scored_candidates.sort(key=priority)
            
            # Add some randomness to ties to explore different branches
            # Group by priority and shuffle within groups
            groups = {}
            for item in scored_candidates:
                p = priority(item)
                if p not in groups:
                    groups[p] = []
                groups[p].append(item)
            
            ordered_candidates = []
            for p in sorted(groups.keys()):
                rng.shuffle(groups[p])
                ordered_candidates.extend(groups[p])
            
            for score, u in ordered_candidates:
                # Try adding u
                used[u] = True
                path.append(u)
                
                v_prev = path[-2]
                for w in neighbors[v_prev]:
                    if w != u:
                        block_count[w] += 1
                        
                backtrack(depth + 1)
                
                # Backtrack
                path.pop()
                used[u] = False
                for w in neighbors[v_prev]:
                    if w != u:
                        block_count[w] -= 1

            return False

        # Run multiple trials
        for trial in range(2000):
            if time.time() - start_time > time_limit:
                break
            rng = random.Random(trial)
            
            # Reset state for each trial? 
            # The current structure reuses path/used/block_count from outer scope.
            # We need to reset them for each trial to start fresh.
            # However, the `backtrack` function uses nonlocal variables.
            # Let's wrap the state in a way that allows reset, or just re-initialize.
            
            # Since `path`, `used`, `block_count` are in `solve`'s scope, 
            # and `solve` is called once, we can't easily reset them between trials 
            # without restructuring.
            
            # Let's restructure `solve` to take start vertex and seed, and create local state.
            # Or, simpler: just call a new inner function for each trial.
            
            # Actually, the current code structure has `solve` creating `path`, `used`, `block_count`.
            # And then the loop calls `backtrack` which uses these.
            # But `backtrack` is defined inside `solve`.
            # If I want multiple trials, I should define `backtrack` to take state, or re-initialize state.
            
            # Let's modify the loop to re-initialize state.
            # But `backtrack` is a closure. 
            # Easiest fix: Make `solve` run one trial, and call `solve` multiple times?
            # No, `solve` sets up `neighbors` etc.
            
            # Let's change the structure:
            # Move `path`, `used`, `block_count` initialization inside the trial loop?
            # But `backtrack` is defined once.
            
            # Alternative: Pass state to `backtrack`? That's slow in Python.
            
            # Let's just re-initialize the lists in the loop.
            # But `backtrack` references `path`, `used`, `block_count` from `solve`'s scope.
            # If I reassign `path = [0]` inside the loop, does `backtrack` see it?
            # Yes, if `backtrack` doesn't assign to them (it only mutates via append/pop).
            # Wait, `backtrack` does `path.append`, `path.pop`. It doesn't rebind `path`.
            # So if I do `path[:] = [0]`, it works.
            
            path[:] = [0]
            for i in range(TOTAL):
                used[i] = False
            used[0] = True
            for i in range(TOTAL):
                block_count[i] = 0
            
            backtrack(1)
            
        return best_path

    return solve(random.Random(0))
# EVOLVE-BLOCK-END
