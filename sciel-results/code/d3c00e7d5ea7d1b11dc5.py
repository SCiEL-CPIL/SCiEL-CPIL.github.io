import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead/return
    
    def prepare_data(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Red K4s: all 4-subsets
        red_k4s = []
        for quad in combinations(range(n), 4):
            edges = []
            for tri in combinations(quad, 3):
                edges.append(triple_to_idx[tri])
            red_k4s.append(edges)
            
        # Blue K5s: all 5-subsets
        blue_k5s = []
        for quint in combinations(range(n), 5):
            edges = []
            for tri in combinations(quint, 3):
                edges.append(triple_to_idx[tri])
            blue_k5s.append(edges)
            
        # For each edge, which K4s and K5s does it belong to?
        edge_to_k4s = [[] for _ in range(num_triples)]
        edge_to_k5s = [[] for _ in range(num_triples)]
        
        for i, edges in enumerate(red_k4s):
            for e in edges:
                edge_to_k4s[e].append(i)
                
        for i, edges in enumerate(blue_k5s):
            for e in edges:
                edge_to_k5s[e].append(i)
        
        return triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s

    def local_search(n, time_limit=10.0, seed_val=0):
        """
        Fast local search using incremental count updates.
        """
        rng = random.Random(seed_val)
        triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s = prepare_data(n)
        num_triples = len(triples)
        
        # Initialize with random colors
        colors = [rng.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        red_counts = [0] * len(red_k4s)
        blue_counts = [0] * len(blue_k5s)
        
        for i, edges in enumerate(red_k4s):
            red_counts[i] = sum(1 for e in edges if colors[e] == 0)
        for i, edges in enumerate(blue_k5s):
            blue_counts[i] = sum(1 for e in edges if colors[e] == 1)
        
        num_red_violations = sum(1 for c in red_counts if c == 4)
        num_blue_violations = sum(1 for c in blue_counts if c == 10)
        total_violations = num_red_violations + num_blue_violations
        
        start = time.time()
        
        # Precompute lists of violating cliques indices for fast access?
        # Or just scan? Scanning 300k items is fast in C (numpy) but slow in Python.
        # We can maintain a set/list of violating indices.
        
        # Let's use a simple approach: pick a random edge from a random violating clique.
        # To find a violating clique quickly, we can maintain a list of violating K4s and K5s.
        
        violating_k4s = [i for i, c in enumerate(red_counts) if c == 4]
        violating_k5s = [i for i, c in enumerate(blue_counts) if c == 10]
        
        while time.time() - start < time_limit:
            if total_violations == 0:
                return colors
            
            # Choose a random violating clique
            # Weight by number of violations? Or just pick random type.
            if not violating_k4s and not violating_k5s:
                return colors
                
            if violating_k4s and (not violating_k5s or rng.random() < 0.5):
                # Pick random K4 violation
                k4_idx = rng.choice(violating_k4s)
                # Pick random edge in this K4 to flip
                e = rng.choice(red_k4s[k4_idx])
            else:
                k5_idx = rng.choice(violating_k5s)
                e = rng.choice(blue_k5s[k5_idx])
            
            current_color = colors[e]
            new_color = 1 - current_color
            
            # Calculate delta in violations
            # We need to update counts and see how many violations are created/removed
            
            # It's complex to calculate delta without updating.
            # Let's just update, check if total_violations decreased.
            # If not, revert.
            # This is "First Improvement" or "Best Improvement" with revert.
            # Since update is O(500), this is fine.
            
            old_total = total_violations
            
            # Apply flip
            colors[e] = new_color
            
            # Update red_counts
            for k4_idx in edge_to_k4s[e]:
                if current_color == 0: # Was Red, now Blue
                    old_c = red_counts[k4_idx]
                    red_counts[k4_idx] = old_c - 1
                    if old_c == 4:
                        total_violations -= 1
                        # Remove from violating_k4s? 
                        # Maintaining a list efficiently is hard. 
                        # Let's just recalculate violating lists every N steps or use a set.
                        # For now, we'll just rely on the counts and pick a random index 
                        # from the full range if the list is stale, or just rebuild the list periodically.
                        # Actually, rebuilding the list of 300k items takes ~0.05s. 
                        # We can do it every 100 steps.
                else: # Was Blue, now Red
                    old_c = red_counts[k4_idx]
                    red_counts[k4_idx] = old_c + 1
                    if old_c == 3:
                        total_violations += 1
                        
            # Update blue_counts
            for k5_idx in edge_to_k5s[e]:
                if current_color == 1: # Was Blue, now Red
                    old_c = blue_counts[k5_idx]
                    blue_counts[k5_idx] = old_c - 1
                    if old_c == 10:
                        total_violations -= 1
                else: # Was Red, now Blue
                    old_c = blue_counts[k5_idx]
                    blue_counts[k5_idx] = old_c + 1
                    if old_c == 9:
                        total_violations += 1
            
            # If we made it worse, revert
            if total_violations > old_total:
                colors[e] = current_color
                # Revert red_counts
                for k4_idx in edge_to_k4s[e]:
                    if current_color == 0:
                        red_counts[k4_idx] += 1
                    else:
                        red_counts[k4_idx] -= 1
                # Revert blue_counts
                for k5_idx in edge_to_k5s[e]:
                    if current_color == 1:
                        blue_counts[k5_idx] += 1
                    else:
                        blue_counts[k5_idx] -= 1
                total_violations = old_total
                # Stuck?
                # We can add a "stuck" counter and if stuck, accept a bad move (Simulated Annealing)
                # For now, let's just try to rebuild the violating lists if we are stuck for a while.
                pass
            
            # Rebuild violating lists periodically to ensure we pick valid ones
            # If we don't, we might pick an index that is no longer violating, 
            # which is fine (we just flip a random edge in a non-violating clique, 
            # which might be bad, but we revert if it's worse).
            # However, if the list is empty but total_violations > 0, we are in trouble.
            # Let's rebuild every 50 steps.
            if total_violations > 0:
                # Simple check: if lists are empty but violations exist, rebuild
                if (not violating_k4s and num_red_violations > 0) or (not violating_k5s and num_blue_violations > 0):
                    violating_k4s = [i for i, c in enumerate(red_counts) if c == 4]
                    violating_k5s = [i for i, c in enumerate(blue_counts) if c == 10]
            else:
                return colors
        
        return None

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try increasing n
    for n in range(10, 35):
        if time.time() > deadline - 5:
            break
        
        # Time budget for this n
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        
        found = False
        # Allocate more time for larger n if possible?
        # For n=34, we need more search.
        # Let's give each n a fixed time slice, or proportional.
        # Let's try 2 seconds per attempt, up to 5 attempts?
        # Or just one long attempt for the largest n?
        # Better to try smaller n first to establish a baseline, then spend time on larger.
        
        time_per_attempt = min(2.0, remaining / 2.0)
        
        for attempt in range(5):
            if time.time() > deadline - 2:
                break
            
            colors = local_search(n, time_limit=time_per_attempt, seed_val=attempt * 1000 + n)
            
            if colors is not None:
                best_n = n
                best_colors = colors
                found = True
                # If we found a valid coloring for n, we can try n+1 immediately
                # But the loop structure handles that.
                break
        
        # If we found a valid coloring, we can continue to next n.
        # If not, we might want to spend more time?
        # For now, the loop just moves to next n.
        
    return {'n': best_n, 'colors': best_colors}
