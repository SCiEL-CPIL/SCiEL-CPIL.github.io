# ... inside construct()
    stuck_counter = 0
    last_x_fail = -1
    
    while time.time() < deadline:
        x_fail = get_fail_point(counts)
        
        if x_fail > best_x_fail:
            best_x_fail = x_fail
            best_residues = residues.copy()
            best_counts = counts.copy()
            stuck_counter = 0
        else:
            if x_fail == last_x_fail:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_x_fail = x_fail
        
        if x_fail >= max_check:
            break
            
        # If stuck for too long, do a stronger kick
        if stuck_counter > 50:
             # Randomize a few residues
             for _ in range(3):
                 idx = np.random.randint(n)
                 p = primes[idx]
                 r_old = residues[idx]
                 r_new = np.random.randint(0, p)
                 counts[r_old if r_old > 0 else p ::p] -= 1
                 counts[r_new if r_new > 0 else p ::p] += 1
                 residues[idx] = r_new
             stuck_counter = 0
             continue

        # Try to fix x_fail
        # ... (same as before)
