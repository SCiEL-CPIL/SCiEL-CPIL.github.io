                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c
                elif gain == best_gain and gain > 0:
                    if random.random() < 0.5:
                        best_candidate = c
