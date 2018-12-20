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
    % version 2.2, 2018-12-20; Jonatan Ropponen, Tomi Karjalainen
    
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
    
    % By default, parallel computing is not used.
    if nargin < 5 || num_cores < 1
        num_cores = 1;
    end
    
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
                cv_error_lambda_i(i, :) = ridge_cv_error_calculation(y, X, lambda, current_indices);
            end
            
            if ~isempty(par_workers)
                delete(par_workers);
            end
        
        else
            
            for i = 1:K
                current_indices = cell2mat(training_set_indices(i));
                cv_error_lambda_i(i, :) = ridge_cv_error_calculation(y, X, lambda, current_indices);
            end
        end
        
        % Determining the error overall for each value of lambda.
        for i = 1:n_lambda
            cv_error_lambda(i) = mean(cv_error_lambda_i(:, i));
        end
        
        [~, opt_idx] = min(cv_error_lambda);
        lambda_opt = lambda(opt_idx(1));
        
    end
end

