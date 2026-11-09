def dispatch(ready_operations, time):
        if not ready_operations:
            return 0 # Should not happen, but safe

        best_idx = 0
        best_score = -1.0

        for i, op in enumerate(ready_operations):
            # Get SPT: min duration
            durations = op["machine_options"].values()
            spt = min(durations) if durations else 1.0
            
            # Get Criticality: remaining work
            crit = op["remaining_work"]
            
            # Avoid division by zero
            if spt == 0:
                spt = 1e-9
                
            score = crit / spt
            
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx
