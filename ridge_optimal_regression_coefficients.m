function B_lambda_opt = ridge_optimal_regression_coefficients(Y, regressor_matrix, lambda_opt_universal, num_cores)

    % Determining the regression coefficients for all voxels in the image 
    % with the universal value of parameter lambda. 
    %
    % Inputs:
    %
    % Y: a matrix of observed responses.
    %
    % regressor_matrix: a matrix containing regressors.
    %
    % lambda_opt_universal: the universal optimal value of lambda, to be applied to 
    % all voxels.
    %
    % Optional input:
    %
    % num_cores: the number of cores to be used for parallel processing.
    % Default: 1 (non-parallel).
    %
    % Outputs:
    %
    % B_lambda_opt: the regression coefficients with the optimal value of lambda.
    %
    % version 1.4, 2019-03-08, Jonatan Ropponen

    
    % Lambda must not be given negative values.
    if lambda_opt_universal < 0
        lambda_opt_universal = 0;
        msg = 'Lambda must be non-negative.';
        disp(msg);
    end
    
    % By default, parallel computing is not used.
    if nargin < 4 || num_cores < 1
        num_cores = 1;
    end
    
    M = size(Y, 2);
    
    nx = size(regressor_matrix, 2);
    dimensions = [M, nx, 1];
    B_lambda_opt = zeros(dimensions);

    if num_cores > 1
        
        [~, par_workers] = create_parpool(num_cores);

        parfor i = 1:M
            
            y = Y(:, i);

            % Using Matlab's default ridge function
            b = ridge(y, regressor_matrix, lambda_opt_universal);

            % Alternative: with ridge_tpc
            % K = 2;
            % cv_randomized = 1;
            % b_lambda_opt_only = 1;
            % calculate_sigma = 0;
            % [~, b_lambda_opt] = ridge_tpc(y, X, lambda, K, cv_randomized, b_lambda_opt_only, calculate_sigma);

            B_lambda_opt(i, :, :) = b;
        end
        
        if ~isempty(par_workers)
            delete(par_workers);
        end

    else

        for i = 1:M

            y = Y(:, i);

            % Using Matlab's default ridge function
            b = ridge(y, regressor_matrix, lambda_opt_universal);

            % Alternative: with ridge_tpc
            % K = 2;
            % cv_randomized = 1;
            % b_lambda_opt_only = 1;
            % calculate_sigma = 0;
            % [~, b_lambda_opt] = ridge_tpc(y, X, lambda, K, cv_randomized, b_lambda_opt_only, calculate_sigma);

            B_lambda_opt(i, :, :) = b;
        end 
    end

end

