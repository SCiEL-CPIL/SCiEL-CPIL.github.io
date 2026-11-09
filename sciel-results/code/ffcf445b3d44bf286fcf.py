        delta = new_score - current_score
        if delta >= 0:
            accept = True
        else:
            # Avoid overflow/underflow
            if delta / T < -50:
                accept = False
            else:
                accept = rng.random() < math.exp(delta / T)
        
        if accept:
            # ...
