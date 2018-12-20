function [b_lambda, b_lambda_opt, lambda_opt, sigma_b] = ridge_tpc(y, X, lambda, K, num_cores, b_lambda_opt_only, calculate_sigma)

    % A function for estimating ridge regression coefficients.
    %
    % Inputs:
    %
    % y: n-by-1 vector of observed responses.
    %
    % X: n-by-p matrix of p predictors at n observations.
    %
    % lambda: a vector of parameters for ridge regression. If lambda consists of m 
    % elements, the calculated b_lambda is p-by-m in size.
    %
    % Optional inputs:
    %
    % K: The number of training sets used in cross-validation. Each set 
    % is treated as the validation set in turn. The data is split evenly 
    % into the sets by its timepoints. E.g. with K = 2 the two 
    % training sets are the first half and the second half.
    %
    % b_lambda_opt_only: whether the analysis should only be carried out to 
    % determine b with the optimal value of lambda rather than with all its 
    % values. Possible values: 0 and 1. By default, the analysis is 
    % carried out in its entirety.
    %
    % num_cores: The number of cores to be used for parallel processing.
    % Default: 1 (non-parallel).
    %
    % calculate_sigma: Signifies whether sigma_b should be calculated.
    % Possible values: 0 and 1. By default, we proceed with the
    % calculation.
    %
    % Outputs:
    %
    % b_lambda: ridge regression coefficients with various values of parameter
    % lambda.
    %
    % sigma_b: the covariance matrix of b.
    %
    % lambda_opt: the optimal value of lambda.
    %
    % b_lambda_opt: the values of b_lambda that correspond to the optimal value of lambda.
    %
    % version 3.2, 2018-12-20; Jonatan Ropponen, Tomi Karjalainen
    
    % Default values
    
    if nargin < 3
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
    
    if nargin < 4
        K = 2;
    end
    
    if nargin < 5
        b_lambda_opt_only = 0;
    end
    
    % By default, parallel computing is not used.
    if nargin < 6 || num_cores < 1
        num_cores = 1;
    end
    
    if nargin < 7
        calculate_sigma = 1;
    end
    
    % First centering y.
    ymean = mean(y);
    y_centered = y - ymean;

    % Standardizing the design matrix.
    Z = zscore(X);

    % Estimating the ridge coefficients.
    p = size(X, 2);
    b_lambda = zeros(p, n_lambda);
    [U, S, V] = svd(Z, 'econ');
    d = diag(S);
    A = U' * y_centered;
    
    if b_lambda_opt_only == 0
      
        for i = 1:n_lambda
            di = d./(d.^2 + lambda(i));
            b_lambda(:, i) = V * diag(di) * A;
        end 
    else 
        b_lambda = [];
    end
    
    % If only a single value of lambda has been provided, we can skip this 
    % section.
    
    if n_lambda == 1
        lambda_opt = lambda(1);
    else
        % Choosing the optimal lambda by cross-validation.
        lambda_opt = ridge_cross_validation(y, X, lambda, K, num_cores);
    end
    
    % The ridge coefficients with the optimal value of lambda
    
    lambda_index = find(lambda == lambda_opt);
    
    if length(lambda_index) > 1   
        lambda_index = lambda_index(1);
    end
    
    if b_lambda_opt_only == 0
        b_lambda_opt = b_lambda(:, lambda_index);
    else
        d_opt = d./(d.^2 + lambda(lambda_index));
        b_lambda_opt = V * diag(d_opt) * A;    
    end
    
    p = size(X, 2);
    
    if calculate_sigma == 1
    
        % Calculating the degrees of freedom.
        % (Not currently used; perhaps could be implemented in the 
        % estimation of sigma.)        
        %df = sum((d.^2) / (d.^2 + lambda_opt));

        % Calculating the residuals.
        H = Z' * Z + lambda_opt * eye(p);
        yhat = Z * (H \ (Z' * y));
        res = y - yhat;

        % Estimating the standard deviation.
        sse = sum(res.^2);
        sigma = sse / length(res); % Should df be taken into account?
        W = H \ Z' * Z;
        var_b = sigma^2 * W * ((Z' * Z) \ W');
        sigma_b = sqrt(var_b);
    
    else
        
        sigma_b = [];
        
    end

end