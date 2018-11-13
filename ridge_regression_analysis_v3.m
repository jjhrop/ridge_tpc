function [Y_hat_k_opt, B_k_opt, k_opt_universal, k_opt_list_sample, sample_indices] = ridge_regression_analysis_v3(image_file_path, regressor_matrix, k_values, default_mask_applied, mask_file_path, k_opt_only, num_cores)

    % Analyzing signal intensity data with the ridge regression function 
    % ridge_tpc and its variants.
    %
    % Inputs:
    %
    % image_file_path: the path of the .nii file to be examined, e.g. 
    % '/scratch/shared/Matlab_code/EPI_preprocessed/103/epi_preprocessed.nii'.
    %
    % regressor_matrix: a matrix containing regressors.
    %
    % Optional inputs:
    %
    % k_values: an array containing the k values utilized in ridge
    % regression, e.g. [0 0.1 1 10 100 1000].
    %
    % default_mask_applied: whether a default mask should be used. 
    % Possible values: 0 and 1. By default, the mask is applied.
    %
    % mask_file_path: the path of a .nii file with a mask, e.g. 
    % '/scratch/shared/Matlab_code/Ridge_regression_files/MNI152_T1_2mm_brain_mask.nii'.
    %
    % k_opt_only: whether the analysis should only be carried out to 
    % determine the optimal value of k. Possible values: 0 and 1.
    %
    % num_cores: the number of cores to be used for parallel processing.
    % Default: 1 (non-parallel).
    %
    % Outputs:
    %
    % Y_hat_k_opt: The estimate of the signal intensities with the optimal
    % value of k.
    %
    % B_k_opt: the regression coefficients with the optimal value of k.
    %
    % k_opt_universal: the value of k chosen as the universal value for
    % determining estimates. The most frequent of the optimal values of k 
    % for each voxel.
    %
    % k_opt_list_sample: the optimal value of k for each voxel in the 
    % sample.
    %
    % sample_indices: the indices of the voxels chosen for the sample.
    
    % version 3.0, 2018-11-13, Jonatan Ropponen

    % example script
    % image_file_path = '/scratch/shared/Matlab_code/EPI_preprocessed/103/epi_preprocessed.nii';
    % load('/scratch/shared/Matlab_code/localizer_regressors/Emotion/103.mat');
    % regressor_matrix = R;
    % k_values = [0 0.1 1 10 100 1000];
    % default_mask_applied = 1;
    % mask_file_path = '/scratch/shared/Matlab_code/Ridge_regression_files/MNI152_T1_2mm_brain_mask.nii';
    % k_opt_only = 0;
    % num_cores = 1;
    
    tic
    
    V = spm_vol(image_file_path);
    img = spm_read_vols(V);
    size_original = size(img);
    
    number_of_time_points = size_original(4);
    number_of_voxels = prod(size_original(1:3));
    
    % The default values of k if they are not specified in the inputs
    if nargin < 3
        k_values = [0 0.1 1 10 100 1000];
    end
    
    % By default, parallel computing is not used.
    if nargin < 7 || num_cores < 1
        num_cores = 1;
    end
    
    % The default mask is applied if specified or if no input is given on
    % whether to apply it.
    if nargin < 4 || default_mask_applied ~= 0
        mask_file_path = '/scratch/shared/Matlab_code/Ridge_regression_files/MNI152_T1_2mm_brain_mask.nii';
    end
    
    % If the mask is not specified and the default mask is not in use, we 
    % proceed without a mask.
    if nargin > 3 && default_mask_applied == 0 && isempty(mask_file_path)
    
        img_2 = ones(size_original(1), size_original(2), size_original(3));
        indices = 1:number_of_voxels;
        
    else    
        
        % Reading the brain mask image
        V_2 = spm_vol(mask_file_path);
        img_2 = spm_read_vols(V_2);
        img_2 = logical(img_2);

        % Saving the indices of the mask
        indices = find(img_2 == true);

    end
        
    siz = size(img_2);

    % Applying the mask to the image
    Y = zeros(length(indices), number_of_time_points);

    for i = 1:number_of_time_points
         img_temp = img(:, :, :, i);
         img_temp_2 = img_temp(img_2);
         Y(:, i) = reshape(img_temp_2, length(indices), 1);
    end

    Y = Y';

    M = size(Y, 2);

    if num_cores > 1
        [~, par_workers] = create_parpool(5);
    end
    
    % First we pick a sample of voxels at random and 
    % determine the optimal value of k for each of the voxels.
    % Then we choose the most common k for our universal k.
    % Finally, we run the analysis for regression coefficients and 
    % estimates.
    
    % If only a single value of k is given, this can be skipped.

    if length(k_values) == 1
    
        k_opt_list_sample = {};
        k_opt_universal = k_values(1);
        
    else
    
        sample_size = ceil(M / 10);
        sample_indices = randperm(M, sample_size);
        Y_sample = Y(:, sample_indices);

        k_opt_list_sample = zeros(1, sample_size);
        
        calculate_sigma = 0;

        if num_cores > 1
        
            parfor i = 1:sample_size

                y = Y_sample(:, i);

                [~, ~, k_opt] = ridge_tpc_v2(y, regressor_matrix, k_values, calculate_sigma);
                k_opt_list_sample(i) = k_opt;
            end
            
        else
              
            for i = 1:sample_size

                y = Y_sample(:, i);

                [~, ~, k_opt] = ridge_tpc_v2(y, regressor_matrix, k_values, calculate_sigma);
                k_opt_list_sample(i) = k_opt;
            end
        end

        k_opt_universal = mode(k_opt_list_sample);

    end
    
    
    if nargin < 6 || k_opt_only == 0
    
        nx = size(regressor_matrix, 2);
        dimensions = [M, nx, 1];
        B_k_opt_1 = zeros(dimensions);
        
        if num_cores > 1

            parfor i = 1:M

                y = Y(:, i);

                % Using Matlab's default ridge function for speed
                b = ridge(y, regressor_matrix, k_opt_universal);
                
                % Alternative: with ridge_tpc
                % calculate_sigma = 0;
                % b = ridge_tpc_v2(y, regressor_matrix, k_opt_universal, calculate_sigma);

                B_k_opt_1(i, :, :) = b;
            end

        else
            
            for i = 1:M

                y = Y(:, i);

                % Using Matlab's default ridge function for speed
                b = ridge(y, regressor_matrix, k_opt_universal);
                
                % Alternative: with ridge_tpc
                % calculate_sigma = 0;
                % b = ridge_tpc_v2(y, regressor_matrix, k_opt_universal, calculate_sigma);

                B_k_opt_1(i, :, :) = b;
            end 
        end

        if num_cores > 1
        
            if ~isempty(par_workers)
                delete(par_workers);
            end
        end

        % Rearranging B so that its indices are in line with the original 
        % image
        dim_1 = prod(size_original(1:3));
        dimensions_2 = [dim_1, nx, 1];
        B_k_opt_2 = zeros(dimensions_2);

        for i = 1:length(indices)
            index = indices(i);
            B_k_opt_2(index, :, :) = B_k_opt_1(i, :, :);
        end

        % Reshaping B_k_opt_2 into the shape of the original image
        B_k_opt = reshape(B_k_opt_2, siz(1), siz(2), siz(3), nx);

        Z = zscore(regressor_matrix);

        % Calculating the estimate of Y with the universal optimal value 
        % of k
        dimensions_3 = [prod(siz(1:3)), number_of_time_points];
        Y_hat = zeros(dimensions_3);

        for i = 1:size(B_k_opt_2, 1)

            b_k_opt = B_k_opt_2(i, :);
            y_hat = Z * b_k_opt';
            Y_hat(i, :) = y_hat;

        end

        Y_hat_k_opt = reshape(Y_hat, siz(1), siz(2), siz(3), number_of_time_points);
    
    else
        
        Y_hat_k_opt = {[]};
        B_k_opt = {[]};
    
    end
    
    toc

end