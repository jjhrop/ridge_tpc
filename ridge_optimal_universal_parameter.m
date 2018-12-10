function [k_opt_universal, k_opt_list_sample, sample_indices] = ridge_optimal_universal_parameter(Y, regressor_matrix, k_values, sample_fraction, training_sets, num_cores)

    % Determining the optimal value of the ridge regression parameter k for
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
    % k_values: an array containing the k values utilized in ridge
    % regression, e.g. [0 0.1 1 10 100 1000 10^4 10^5 10^6].
    %
    % sample_fraction: the fraction of all voxels that is picked for the
    % random sample, e.g. 0.1.
    %
    % training_sets: the proportions of the splitting points for the
    % training sets, with the beginning and endpoint for each set in turn, 
    % e.g. [0, 0.5, 0.5, 1].
    %
    % num_cores: the number of cores to be used for parallel processing.
    % Default: 1 (non-parallel).
    %
    % Outputs:
    %
    % k_opt_universal: the value of k chosen as the universal value for
    % determining estimates. The most frequent of the optimal values of k 
    % for each voxel.
    %
    % k_opt_list_sample: the optimal value of k for each voxel in the 
    % sample.
    %
    % sample_indices: the indices of the voxels chosen for the sample.
    
    % version 1.0, 2018-12-04, Jonatan Ropponen
    
    
    % Default entries for optional inputs:
    if nargin < 3 || isempty(k_values)
        k_values = [0 0.1 1 10 100 1000 10^4 10^5 10^6];
    end
    
    if nargin < 4 || isempty(sample_fraction)
        sample_fraction = 1;
    end

    % Default settings for cross-validation:
    % Splitting the data into two subsets by its timepoints, at 
    % a proportion of the entire data. We treat the temporally 
    % prior subset as the training dataset and the latter as the 
    % validation set. In other words, the second subset is removed to 
    % form the training dataset.
    
    if nargin < 5 || isempty(training_sets)
        training_sets = [0, 0.5, 0.5, 1];
    end
    
    % By default, parallel computing is not used.
    if nargin < 6 || num_cores < 1
        num_cores = 1;
    end
        

    % First we pick a sample of voxels at random and 
    % determine the optimal value of k for each of the voxels.
    % Then we choose the most common k for our universal k.
    % Finally, we run the analysis for regression coefficients and 
    % estimates.
    
    % If only a single value of k is given, this can be skipped.

    if length(k_values) == 1
    
        k_opt_list_sample = [];
        k_opt_universal = k_values(1);
        
    else
        
        M = size(Y, 2);
        sample_size = ceil(M * sample_fraction);
        sample_indices = randperm(M, sample_size);
        Y_sample = Y(:, sample_indices);

        k_opt_list_sample = zeros(1, sample_size);
        
        calculate_sigma = 0;
        b_k_opt_only = 1;

        if num_cores > 1
        
            parfor i = 1:sample_size

                y = Y_sample(:, i);

                [~, ~, k_opt] = ridge_tpc(y, regressor_matrix, k_values, training_sets, num_cores, b_k_opt_only, calculate_sigma);
                k_opt_list_sample(i) = k_opt;
            end
            
        else
              
            for i = 1:sample_size

                y = Y_sample(:, i);

                [~, ~, k_opt] = ridge_tpc(y, regressor_matrix, k_values, training_sets, num_cores, b_k_opt_only, calculate_sigma);
                k_opt_list_sample(i) = k_opt;
            end
        end

        k_opt_universal = mode(k_opt_list_sample);
        
    end
end

