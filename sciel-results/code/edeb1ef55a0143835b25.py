            if delta_E < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta_E / T)
                
            if accept:
