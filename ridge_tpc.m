function [b_k, b_k_opt, k_opt, sigma_b] = ridge_tpc(y, X, k, training_sets, num_cores, b_k_opt_only, calculate_sigma)

    % A function for estimating ridge regression coefficients.
    %
    % Inputs:
    %
    % y: n-by-1 vector of observed responses.
    %
    % X: n-by-p matrix of p predictors at n observations.
    %
    % k: a vector of parameters for ridge regression. If k consists of m 
    % elements, the calculated b_k is p-by-m in size.
    %
    % Optional inputs:
    %
    % training_sets: the proportions of the splitting points for the
    % training sets, with the beginning and endpoint for each set in turn, 
    % e.g. [0, 0.5, 0.5, 1].
    %
    % b_k_opt_only: whether the analysis should only be carried out to 
    % determine b with the optimal value of k rather than with all its 
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
    % b_k: ridge regression coefficients with various values of parameter
    % k.
    %
    % sigma_b: the covariance matrix of b.
    %
    % k_opt: the optimal value of k.
    %
    % b_k_opt: the values of b_k that correspond to the optimal value of k.
    %
    % version 3.0, 2018-12-04; Jonatan Ropponen, Tomi Karjalainen
    
    % Default values
    
    if nargin < 3
        k = [0 1 10 100 1000 10^4 10^5 10^6];
    end
    
    if nargin < 4
        training_sets = [0, 0.5, 0.5, 1];
    end
    
    if nargin < 5
        b_k_opt_only = 0;
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
    nk = length(k);
    b_k = zeros(p, nk);
    [U, S, V] = svd(Z, 'econ');
    d = diag(S);
    A = U' * y_centered;
    
    if b_k_opt_only == 0
      
        for i = 1:nk
            di = d./(d.^2 + k(i));
            b_k(:, i) = V * diag(di) * A;
        end 
    else 
        b_k = [];
    end
    
    % If only a single value of k has been provided, we can skip this 
    % section.
    
    if nk == 1
        k_opt = k(1);
    else
        % Choosing the optimal k by cross-validation.
        k_opt = ridge_cross_validation(y, X, k, training_sets, num_cores);
    end
    
    % The ridge coefficients with the optimal value of k
    
    k_index = find(k == k_opt);
    
    if b_k_opt_only == 0
        b_k_opt = b_k(:, k_index);
    else
        d_opt = d./(d.^2 + k(k_index));
        b_k_opt = V * diag(d_opt) * A;    
    end
    
    p = size(X, 2);
    
    if calculate_sigma == 1
    
        % Calculating the degrees of freedom.
        % (Not currently used; perhaps could be implemented in the 
        % estimation of sigma.)        
        %df = sum((d.^2) / (d.^2 + k_opt));

        % Calculating the residuals.
        H = Z' * Z + k_opt * eye(p);
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