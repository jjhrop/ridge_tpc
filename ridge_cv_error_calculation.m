function cv_error_lambda_i = ridge_cv_error_calculation(y, X, lambda, indices)

    % Calculating the cross-validation error for a training set. Primarily
    % used in ridge_cross_validation.m, where it is applied on the ith 
    % training set within a loop. The error matrix is then used in
    % ridge_cross_validation.m to determine the optimal value of parameter 
    % lambda.
    %
    % Inputs:
    %
    % y: n-by-1 vector of observed responses.
    %
    % X: n-by-p matrix of p predictors at n observations.
    %
    % lambda: A vector of parameters for ridge regression, e.g. 
    % [0 1 10 100 1000 10^4 10^5 10^6]. If lambda consists of m elements, 
    % the calculated b_lambda is p-by-m in size.
    %
    % indices: Indices for the timepoints that correspond to a training
    % set.
    %
    % Outputs:
    %
    % cv_error_lambda_i: The cross-validation error for the training set
    % specified in the inputs, i.e. the ith training set. 
    % Dimensions: 1 x (the length of lambda). The ultimate matrix in
    % ridge_cross_validation.m will be 
    % (the number of training sets) x (the length of lambda) in size. 
    %
    % version 1.1, 2018-12-20; Jonatan Ropponen, Tomi Karjalainen
    
    % The default values of lambda if they are not specified in the inputs
    if nargin < 3 || isempty(lambda)
        lambda = [0 1 10 100 1000 10^4 10^5 10^6];
    end
    
    n_lambda = length(lambda);
    
    % Lambda must not be given negative values.
    for i = 1:n_lambda
        if lambda(i) < 0
            lambda(i) = 0;
            msg = 'Lambda must be non-negative.';
            disp(msg);
        end
    end

    % Matrices containing only the ith training set
    X_i = X(indices, :);
    y_i = y(indices, :);

    y_i_mean = mean(y_i);
    y_i_centered = y_i - y_i_mean;

    Z_i = zscore(X_i);

    excluded_set_size = size(Z_i, 1);

    % Matrices with the ith training set removed
    X_neg_i = X;
    y_neg_i = y;

    X_neg_i(indices, :) = [];
    y_neg_i(indices, :) = [];


    % Compiling the values of ridge coefficients after the removal

    % First centering y
    y_neg_i_mean = mean(y_neg_i);
    y_neg_i_centered = y_neg_i - y_neg_i_mean;

    % Standardizing the design matrix
    Z_neg_i = zscore(X_neg_i);

    % Estimating the ridge coefficients
    p_neg_i = size(X_neg_i, 2);
    b_lambda_neg_i = zeros(p_neg_i, n_lambda);
    [U_neg_i, S_neg_i, V_neg_i] = svd(Z_neg_i, 'econ');
    d_neg_i = diag(S_neg_i);
    A_neg_i = U_neg_i' * y_neg_i_centered;

    f_lambda_neg_i = zeros(excluded_set_size, 1);

    for j = 1:n_lambda

        di_neg_i = d_neg_i ./ (d_neg_i.^2 + lambda(j));
        b_lambda_neg_i(:, j) = V_neg_i * diag(di_neg_i) * A_neg_i;

        % Calculating the cross-validation error.
        % Note that the values of b are applied on the excluded
        % subset, i.e. the training set chosen as the validation
        % set.

        f_lambda_neg_i(:, j) = Z_i * b_lambda_neg_i(:, j);
        cv_error_lambda_i(1, j) = (excluded_set_size)^(-1) * sum((y_i_centered - f_lambda_neg_i(:, j)).^2);

    end

end

