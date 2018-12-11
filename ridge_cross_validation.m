function [lambda_opt, cv_error_lambda] = ridge_cross_validation(y, X, lambda, K, num_cores)
    
    % Performing cross-validation on ridge regression data to determine the
    % optimal value of parameter lambda for a single voxel. The function 
    % applies K-fold cross-validation, i.e. cross-validation with K
    % subsets of elements, removing each subset in turn. Each time, we 
    % choose one subset as the validation set and the others as the 
    % training set.
    %
    % Inputs:
    %
    % y: n-by-1 vector of observed responses.
    %
    % X: n-by-p matrix of p predictors at n observations.
    %
    % Optional inputs:
    %
    % lambda: A vector of parameters for ridge regression, e.g. 
    % [0 1 10 100 1000 10^4 10^5 10^6]. If lambda consists of m elements, 
    % the calculated b_lambda is p-by-m in size.
    %
    % K: The number of training sets used in cross-validation. Each set 
    % is treated as the validation set in turn. The data is split evenly 
    % into the sets by its timepoints. E.g. with K = 2 the two 
    % training sets are the first half and the second half.
    %
    % num_cores: The number of cores to be used for parallel processing.
    % Default: 1 (non-parallel).
    %
    % Outputs:
    %
    % lambda_opt: the optimal value of lambda.
    %
    % cv_error_lambda: the error terms calculated by cross-validation with
    % various values of lambda.
    %
    % version 2.0, 2018-12-11; Jonatan Ropponen, Tomi Karjalainen
    
    % Default values
    if nargin < 3
        lambda = [0 1 10 100 1000 10^4 10^5 10^6];
    end
    
    if nargin < 4
        K = 2;
    end
    
    % By default, parallel computing is not used.
    if nargin < 5 || num_cores < 1
        num_cores = 1;
    end
    
    n_lambda = length(lambda);
    
    % If only a single value of lambda has been provided, we can simply return
    % it.
    
    if n_lambda == 1
    
        lambda_opt = lambda(1);
        cv_error_lambda = [];
        
    else

        number_of_indices = size(X, 1);        
        
        % Determining the training sets.
        
        subset_size = number_of_indices / K;
        
        training_set_indices = {};
        
        for i = 1:K
        
            first_index = round((i-1) * subset_size) + 1;
            last_index = min(round(i * subset_size), number_of_indices);
            
            new_training_set = first_index : last_index;
        
            training_set_indices = [training_set_indices, {new_training_set}];
        end

        
        % Choosing the optimal lambda by cross-validation

        % Applying K-fold cross-validation, i.e. cross-validation with K
        % subsets of elements, removing each subset in turn. We choose one
        % subset as the validation set and the others as training sets.
        
        if num_cores > 1
            
            [~, par_workers] = create_parpool(num_cores);
        
            parfor i = 1:K

                current_indices = cell2mat(training_set_indices(i));

                % Matrices containing only the ith training set
                X_i = X(current_indices, :);
                y_i = y(current_indices, :);

                y_i_mean = mean(y_i);
                y_i_centered = y_i - y_i_mean;

                Z_i = zscore(X_i);

                excluded_set_size = size(Z_i, 1);

                % Matrices with the ith training set removed
                X_neg_i = X;
                y_neg_i = y;

                X_neg_i(current_indices, :) = [];
                y_neg_i(current_indices, :) = [];


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
                    cv_error_lambda_i(i, j) = (excluded_set_size)^(-1) * sum((y_i_centered - f_lambda_neg_i(:, j)).^2);

                end
            end
            
            if ~isempty(par_workers)
                delete(par_workers);
            end
        
        else
            
            for i = 1:K

                current_indices = cell2mat(training_set_indices(i));

                % Matrices containing only the ith training set
                X_i = X(current_indices, :);
                y_i = y(current_indices, :);

                y_i_mean = mean(y_i);
                y_i_centered = y_i - y_i_mean;

                Z_i = zscore(X_i);

                excluded_set_size = size(Z_i, 1);

                % Matrices with the ith training set removed
                X_neg_i = X;
                y_neg_i = y;

                X_neg_i(current_indices, :) = [];
                y_neg_i(current_indices, :) = [];


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
                    cv_error_lambda_i(i, j) = (excluded_set_size)^(-1) * sum((y_i_centered - f_lambda_neg_i(:, j)).^2);

                end
            end
        end
        
        % The error overall for each value of lambda
        
        for i = 1:n_lambda
            cv_error_lambda(i) = K^(-1) * sum(cv_error_lambda_i(:, i));
        end
        
        [~, opt_idx] = min(cv_error_lambda);
        lambda_opt = lambda(opt_idx(1));
    
    end
end

