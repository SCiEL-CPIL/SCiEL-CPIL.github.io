            # Greedy check if stuck
            if current_cost > 0:
                best_move = None
                best_cost = current_cost
                # Check a subset of edges or all?
                # All is O(N_edges * CostCalc). N_edges ~ 325. CostCalc ~ 3000. 1M ops.
                # Do this every 100 steps if no improvement.
