function [lambda_opt_universal, lambda_opt_list_sample, sample_indices] = ridge_optimal_universal_parameter(Y, regressor_matrix, lambda_values, sample_fraction, K, num_cores)

    % Determining the optimal value of the ridge regression parameter lambda for
    % all voxels. The calculation can be carried out for a sample of voxels
    % rather than all of them.
    %
    % Inputs:
    %
    % Y: observed responses as indicated by the image file.
    %
    % regressor_matrix: a matrix containing regressors.
    %
    % Optional inputs:
    %
    % lambda_values: an array containing the lambda values utilized in ridge
    % regression, e.g. [0 0.1 1 10 100 1000 10^4 10^5 10^6].
    %
    % sample_fraction: the fraction of all voxels that is picked for the
    % random sample, e.g. 0.1.
    %
    % K: The number of training sets used in cross-validation. Each set 
    % is treated as the validation set in turn. The data is split evenly 
    % into the sets by its timepoints. E.g. with K = 2 the two 
    % training sets are the first half and the second half.
    %
    % num_cores: the number of cores to be used for parallel processing.
    % Default: 1 (non-parallel).
    %
    % Outputs:
    %
    % lambda_opt_universal: the value of lambda chosen as the universal value for
    % determining estimates. The most frequent of the optimal values of lambda 
    % for each voxel.
    %
    % lambda_opt_list_sample: the optimal value of lambda for each voxel in the 
    % sample.
    %
    % sample_indices: the indices of the voxels chosen for the sample.
    
    % version 1.2, 2018-12-18, Jonatan Ropponen
    
    
    % Default entries for optional inputs:
    if nargin < 3 || isempty(lambda_values)
        lambda_values = [0 0.1 1 10 100 1000 10^4 10^5 10^6];
    end
    
    if nargin < 4 || isempty(sample_fraction)
        sample_fraction = 1;
    end

    % Default settings for cross-validation:
    % Splitting the data into two subsets by its timepoints.
    
    if nargin < 5 || isempty(K)
        K = 2;
    end
    
    % By default, parallel computing is not used.
    if nargin < 6 || num_cores < 1
        num_cores = 1;
    end
        

    % First we pick a sample of voxels at random and 
    % determine the optimal value of lambda for each of the voxels.
    % Then we choose the most common lambda for our universal lambda.
    % Finally, we run the analysis for regression coefficients and 
    % estimates.
    
    % If only a single value of lambda is given, this can be skipped.

    if length(lambda_values) == 1
    
        lambda_opt_list_sample = [];
        lambda_opt_universal = lambda_values(1);
        
    else
        
        M = size(Y, 2);
        sample_size = ceil(M * sample_fraction);
        sample_indices = randperm(M, sample_size);
        Y_sample = Y(:, sample_indices);

        lambda_opt_list_sample = zeros(1, sample_size);
        
        calculate_sigma = 0;
        b_lambda_opt_only = 1;
        
        % The parallelization is carried out at this level, so it cannot be
        % applied within ridge_tpc.m.
        num_cores_ridge = 1;

        if num_cores > 1
            
            [~, par_workers] = create_parpool(num_cores);
        
            parfor i = 1:sample_size

                y = Y_sample(:, i);

                [~, ~, lambda_opt] = ridge_tpc(y, regressor_matrix, lambda_values, K, num_cores_ridge, b_lambda_opt_only, calculate_sigma);
                lambda_opt_list_sample(i) = lambda_opt;
            end
            
            if ~isempty(par_workers)
                delete(par_workers);
            end
            
        else
              
            for i = 1:sample_size

                y = Y_sample(:, i);

                [~, ~, lambda_opt] = ridge_tpc(y, regressor_matrix, lambda_values, K, num_cores_ridge, b_lambda_opt_only, calculate_sigma);
                lambda_opt_list_sample(i) = lambda_opt;
            end
        end

        lambda_opt_universal = mode(lambda_opt_list_sample);
        
    end
end

