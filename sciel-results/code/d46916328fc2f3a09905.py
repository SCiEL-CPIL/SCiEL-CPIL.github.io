        if delta > 0:
            accept = True
        else:
            accept = random.random() < math.exp(delta / T)
