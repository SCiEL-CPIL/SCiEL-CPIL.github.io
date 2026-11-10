        # Copy occupied to test
        test_occupied = occupied.copy()
        new_chain, _, valid = _knock_on_move(current_chain, test_occupied, rng)
        
        if valid:
             # Calculate energy
             # If accepted:
             # current_chain = new_chain
             # occupied = test_occupied
