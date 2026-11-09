def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1, 1)
    
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    layer_sizes = [X_norm.shape[1], 16, 16, 1]
    
    # Flatten initial weights and biases
    w, b = _init_weights(layer_sizes)
    theta = np.concatenate([wi.flatten() for wi in w] + [bi.flatten() for bi in b])
    
    def loss_and_grad(theta):
        # Unflatten
        weights = []
        biases = []
        idx = 0
        for i in range(len(layer_sizes) - 1):
            size_in = layer_sizes[i]
            size_out = layer_sizes[i + 1]
            num_w = size_in * size_out
            weights.append(theta[idx:idx+num_w].reshape(size_in, size_out))
            idx += num_w
            biases.append(theta[idx:idx+size_out].reshape(1, size_out))
            idx += size_out
            
        # Forward
        activations, zs = _forward_stored(X_norm, weights, biases, use_relu=False)
        y_pred = activations[-1]
        
        # Loss
        loss = np.mean((y_pred - y_norm) ** 2)
        
        # Gradients
        d_weights, d_biases = _backward_stored(activations, zs, weights, y_norm, use_relu=False)
        
        grad = np.concatenate([dw.flatten() for dw in d_weights] + [db.flatten() for db in d_biases])
        
        return loss, grad
        
    # Optimize
    result = scipy.optimize.minimize(loss_and_grad, theta, method='L-BFGS-B', jac=True, options={'maxiter': 1000})
    
    # Extract final weights
    final_theta = result.x
    weights = []
    biases = []
    idx = 0
    for i in range(len(layer_sizes) - 1):
        size_in = layer_sizes[i]
        size_out = layer_sizes[i + 1]
        num_w = size_in * size_out
        weights.append(final_theta[idx:idx+num_w].reshape(size_in, size_out))
        idx += num_w
        biases.append(final_theta[idx:idx+size_out].reshape(1, size_out))
        idx += size_out
        
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - X_mean) / X_std
        activations, _ = _forward_stored(X_norm, weights, biases, use_relu=False)
        y_pred_norm = activations[-1]
        y_pred = y_pred_norm * y_std + y_mean
        return y_pred.flatten()

    return predict
