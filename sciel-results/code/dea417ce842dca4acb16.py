<<<<<<< SEARCH
    def _polish(selected_indices):
        """Final polishing: repeated redundancy removal and local swaps."""
        if time.time() > deadline:
            return selected_indices
        
        result = _remove_redundant(selected_indices)
        
        # Try multiple rounds of WalkSAT-style minimization with increasing effort
        for round_num in range(4):
            if time.time() > deadline - 1.0:
                break
            # Increase steps in later rounds
            steps = 1500 + round_num * 500
            result = _walksat_minimize(result, max_steps=steps)
            result = _remove_redundant(result)
        
        return result
=======
    def _polish(selected_indices):
        """Final polishing: repeated redundancy removal, local swaps, and ILS."""
        if time.time() > deadline:
            return selected_indices
        
        result = _remove_redundant(selected_indices)
        
        # Try multiple rounds of WalkSAT-style minimization with increasing effort
        for round_num in range(3):
            if time.time() > deadline - 3.0:
                break
            steps = 1500 + round_num * 500
            result = _walksat_minimize(result, max_steps=steps)
            result = _remove_redundant(result)
        
        # Apply ILS for final escape from local minima
        if time.time() < deadline - 2.0:
            result = _ils_pass(result, max_iterations=50, perturb_rate=0.03)
            result = _remove_redundant(result)
        
        return result
>>>>>>> REPLACE

