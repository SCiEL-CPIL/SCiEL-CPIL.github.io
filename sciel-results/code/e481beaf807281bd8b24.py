    current_cost = 0
    for i in range(N - 1):
        current_cost += C[seq[i]][seq[i+1]]
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 2):
            if time.time() > deadline: break
            for j in range(i + 1, N - 1):
                # Reverse seq[i+1 .. j]
                # Old edges: (i, i+1) and (j, j+1)
                # New edges: (i, j) and (i+1, j+1)
                # Note: If j == N-1, there is no (j, j+1) edge.
                
                old_cost = C[seq[i]][seq[i+1]] + (C[seq[j]][seq[j+1]] if j < N-1 else 0)
                new_cost = C[seq[i]][seq[j]] + (C[seq[i+1]][seq[j+1]] if j < N-1 else 0)
                
                if new_cost < old_cost:
                    # Reverse
                    seq[i+1:j+1] = reversed(seq[i+1:j+1])
                    current_cost += new_cost - old_cost
                    improved = True
                    # After a move, we should restart the inner loop or continue?
                    # Usually restart or continue. Continuing is faster but might miss better moves.
                    # Let's continue.
