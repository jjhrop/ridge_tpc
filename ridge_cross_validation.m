function [k_opt, cv_error_k] = ridge_cross_validation(y, X, k, training_sets, num_cores)
    
    % Performing cross-validation on ridge regression data to determine the
    % optimal value of parameter k for a single voxel. The function 
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
    % k: A vector of parameters for ridge regression, e.g. 
    % [0 1 10 100 1000 10^4 10^5 10^6]. If k consists of m elements, 
    % the calculated b_k is p-by-m in size.
    %
    % training_sets: the proportions of the splitting points for the
    % training sets, with the beginning and endpoint for each set in turn, 
    % e.g. [0, 0.5, 0.5, 1]. It is preferable for measuring error if the
    % sets are roughly equal in size. Furthermore, overlaps are removed.
    %
    % num_cores: The number of cores to be used for parallel processing.
    % Default: 1 (non-parallel).
    %
    % Outputs:
    %
    % k_opt: the optimal value of k.
    %
    % cv_error_k: the error terms calculated by cross-validation with
    % various values of k.
    %
    % version 1.0, 2018-12-04; Jonatan Ropponen, Tomi Karjalainen
    
    % Default values
    if nargin < 3
        k = [0 1 10 100 1000 10^4 10^5 10^6];
    end
    
    if nargin < 4
        training_sets = [0, 0.5, 0.5, 1];
    end
    
    % By default, parallel computing is not used.
    if nargin < 5 || num_cores < 1
        num_cores = 1;
    end
    
    nk = length(k);
    
    % If only a single value of k has been provided, we can simply return
    % it.
    
    if nk == 1
    
        k_opt = k(1);
        cv_error_k = [];
        
    else

        number_of_indices = size(X, 1);        
        
        % Determining the training sets.
        
        training_set_indices = {};
        
        number_of_training_sets = floor(length(training_sets) / 2);
        
        for i = 1:number_of_training_sets
        
            first_splitting_point = training_sets(2 * i - 1);
            
            if first_splitting_point < 0
                first_splitting_point = 0;
            elseif first_splitting_point > 1
                first_splitting_point = 1;
            end
            
            last_splitting_point = training_sets(2 * i);
            
            if last_splitting_point < 0
                last_splitting_point = 0;
            elseif last_splitting_point > 1
                last_splitting_point = 1;
            elseif last_splitting_point < first_splitting_point
                last_splitting_point = first_splitting_point;
            end
            
            first_index = ceil(number_of_indices * first_splitting_point);
            
            if first_index < 1
                first_index = 1;
            end
            
            last_index = floor(number_of_indices * last_splitting_point);
            
            new_training_set = first_index : last_index;
            
            % Removing overlapping indices.
            new_training_set = setdiff(new_training_set, cell2mat(training_set_indices));
            
            training_set_indices = [training_set_indices, {new_training_set}];
        end
        
        % Choosing the optimal k by cross-validation

        % Applying K-fold cross-validation, i.e. cross-validation with K
        % subsets of elements, removing each subset in turn. We choose one
        % subset as the validation set and the others as training sets.
        
        if num_cores > 1
            
            [~, par_workers] = create_parpool(num_cores);
        
            parfor i = 1:number_of_training_sets

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
                b_k_neg_i = zeros(p_neg_i, nk);
                [U_neg_i, S_neg_i, V_neg_i] = svd(Z_neg_i, 'econ');
                d_neg_i = diag(S_neg_i);
                A_neg_i = U_neg_i' * y_neg_i_centered;

                f_k_neg_i = zeros(excluded_set_size, 1);

                for j = 1:nk

                    di_neg_i = d_neg_i ./ (d_neg_i.^2 + k(j));
                    b_k_neg_i(:, j) = V_neg_i * diag(di_neg_i) * A_neg_i;

                    % Calculating the cross-validation error.
                    % Note that the values of b are applied on the excluded
                    % subset, i.e. the training set chosen as the validation
                    % set.

                    f_k_neg_i(:, j) = Z_i * b_k_neg_i(:, j);
                    cv_error_k_i(i, j) = (excluded_set_size)^(-1) * sum((y_i_centered - f_k_neg_i(:, j)).^2);

                end
            end
            
            if ~isempty(par_workers)
                delete(par_workers);
            end
        
        else
            
            for i = 1:number_of_training_sets

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
                b_k_neg_i = zeros(p_neg_i, nk);
                [U_neg_i, S_neg_i, V_neg_i] = svd(Z_neg_i, 'econ');
                d_neg_i = diag(S_neg_i);
                A_neg_i = U_neg_i' * y_neg_i_centered;

                f_k_neg_i = zeros(excluded_set_size, 1);

                for j = 1:nk

                    di_neg_i = d_neg_i ./ (d_neg_i.^2 + k(j));
                    b_k_neg_i(:, j) = V_neg_i * diag(di_neg_i) * A_neg_i;

                    % Calculating the cross-validation error.
                    % Note that the values of b are applied on the excluded
                    % subset, i.e. the training set chosen as the validation
                    % set.

                    f_k_neg_i(:, j) = Z_i * b_k_neg_i(:, j);
                    cv_error_k_i(i, j) = (excluded_set_size)^(-1) * sum((y_i_centered - f_k_neg_i(:, j)).^2);

                end
            end
        end
        
        % The error overall for each value of k
        
        for i = 1:nk
            cv_error_k(i) = number_of_training_sets^(-1) * sum(cv_error_k_i(:, i));
        end
        
        [~, opt_idx] = min(cv_error_k);
        k_opt = k(opt_idx(1));
    
    end
end

