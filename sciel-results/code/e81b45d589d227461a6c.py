import numpy as np

def dispatch(ready_operations, time):
        if not ready_operations:
            return 0
        
        n = len(ready_operations)
        # Pre-calculate max remaining work for normalization
        max_rw = 0
        min_durs = []
        n_machines_list = []
        
        for op in ready_operations:
            mo = op["machine_options"]
            if not mo:
                min_durs.append(float('inf'))
                n_machines_list.append(1) # Treat as 1 to avoid div by 0, or handle separately
                continue
            durs = list(mo.values())
            min_durs.append(min(durs))
            n_machines_list.append(len(durs))
            rw = op.get("remaining_work", 0)
            if rw > max_rw:
                max_rw = rw
        
        if max_rw == 0:
            max_rw = 1.0
            
        scores = np.zeros(n)
        for i, op in enumerate(ready_operations):
            md = min_durs[i]
            nm = n_machines_list[i]
            rw = op.get("remaining_work", 0)
            
            # Constraint: 1/nm (higher is more constrained)
            constraint_score = 1.0 / nm if nm > 0 else 0.0
            
            # Speed: 1/(md + 1) (higher is faster)
            speed_score = 1.0 / (md + 1.0) if md < float('inf') else 0.0
            
            # Load: rw / max_rw (higher is more work)
            load_score = rw / max_rw
            
            # Weighted sum
            # We want to prioritize constrained and fast ops.
            # Let's weight speed and constraint heavily.
            scores[i] = 2.0 * constraint_score + 2.0 * speed_score + 1.0 * load_score
            
        best_idx = int(np.argmax(scores))
        return best_idx
