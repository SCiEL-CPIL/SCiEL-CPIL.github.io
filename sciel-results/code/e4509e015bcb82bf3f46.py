            # Sample edges to test to save time
            edges_list = list(remaining)
            rng = random.Random(123) # Fixed seed for determinism in heuristic? Or use global?
            # Actually, we want the result to be deterministic for the verifier?
            # The verifier uses fixed seeds for instance generation.
            # The decomposition should be deterministic to be reproducible?
            # The task says "Minimize the total number of pieces". It doesn't explicitly require determinism of the *algorithm's internal choices*, but the output must be valid.
            # However, using a fixed RNG seed inside the heuristic is good for consistency.
            
            if len(edges_list) > 20:
                edges_to_check = rng.sample(edges_list, 20)
            else:
                edges_to_check = edges_list
            
            for u, v in edges_to_check:
                # ... rest of logic
