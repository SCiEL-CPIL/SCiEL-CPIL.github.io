            if delta_cost <= 0:
                accepted = True
            else:
                prob = math.exp(-delta_cost / T)
                accepted = random.random() < prob
