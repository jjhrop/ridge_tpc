function [Y_hat_k_opt, B_k_opt, k_opt_universal, k_opt_list_sample, sample_indices] = ridge_image(image_file_path, regressor_matrix, k_values, mask_file_path, k_opt_only, sample_fraction, training_sets, num_cores)

    % Analyzing observed responses with the ridge regression function 
    % ridge_tpc. A universal value of parameter k is determined for all
    % voxels The estimate of observed responses is universally calculated 
    % with this value of k. 
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
    % regression, e.g. [0 0.1 1 10 100 1000 10^4 10^5 10^6].
    %
    % mask_file_path: the path of a .nii file with a mask, e.g. 
    % '/scratch/shared/Matlab_code/Ridge_regression_files/MNI152_T1_2mm_brain_mask.nii'.
    %
    % k_opt_only: whether the analysis should only be carried out to 
    % determine the optimal value of k. Possible values: 0 and 1. By 
    % default, the analysis is carried out in its entirety.
    %
    % sample_fraction: the fraction of all voxels that is picked for the
    % random sample, e.g. 0.1. Used when determining the universal optimal 
    % value of parameter k.
    %
    % training_sets: the proportions of the splitting points for the
    % training sets, with the beginning and endpoint for each set in turn, 
    % e.g. [0, 0.5, 0.5, 1]. Used in cross-validation.
    %
    % num_cores: the number of cores to be used for parallel processing.
    % Default: 1 (non-parallel).
    %
    % Outputs:
    %
    % Y_hat_k_opt: An estimate of the observed responses with the optimal
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
    
    % version 1.0, 2018-12-04, Jonatan Ropponen

    % example script
    %
    % image_file_path = '/example_directory/EPI_preprocessed/103/epi_preprocessed.nii';
    % load('/example_directory/localizer_regressors/Emotion/103.mat');
    % regressor_matrix = R;
    % k_values = [0 1 10 100 1000 10^4 10^5 10^6];
    % mask_file_path = '/example_directory/Ridge_regression_files/MNI152_T1_2mm_brain_mask.nii';
    % k_opt_only = 0;
    % sample_fraction = 0.1;
    % training_sets = [0, 0.5, 0.5, 1];
    % num_cores = 1;
    % [Y_hat_k_opt, B_k_opt, k_opt_universal, k_opt_list_sample, sample_indices] = ridge_image(image_file_path, regressor_matrix, k_values, mask_file_path, k_opt_only, sample_fraction, training_sets, num_cores);

    % Default values
    
    % The default values of k if they are not specified in the inputs
    if nargin < 3 || isempty(k_values)
        k_values = [0 1 10 100 1000 10^4 10^5 10^6];
    end
    
    if nargin < 5 || isempty(k_opt_only)
        k_opt_only = 0;
    end
    
    if nargin < 6 || isempty(sample_fraction)
        sample_fraction = 0.1;
    end
    
    if nargin < 7 || isempty(training_sets)
        training_sets = [0, 0.5, 0.5, 1];
    end
    
    % By default, parallel computing is not used.
    if nargin < 8 || isempty(num_cores) || num_cores < 1
        num_cores = 1;
    end
        
    
    V = spm_vol(image_file_path);
    img = spm_read_vols(V);
    size_original = size(img);
    
    number_of_time_points = size_original(4);
    number_of_voxels = prod(size_original(1:3));
    
    % If no mask has been specified, we determine a mask based on the data.
    if nargin < 4 || isempty(mask_file_path)
    
        % The mean values over the time points
        mean_img = mean(img, 4, 'omitnan');
        img_max_value = max(img(:));
        threshold = 0.8;
        img_2 = mean_img > threshold * img_max_value;
        
    else
        
        % Reading the brain mask image.
        V_2 = spm_vol(mask_file_path);
        img_2 = spm_read_vols(V_2);
        img_2 = logical(img_2);

    end
    
    % Saving the indices of the mask.
    indices = find(img_2 == true);
        
    siz = size(img_2);

    % Applying the mask to the image.
    Y = zeros(length(indices), number_of_time_points);

    for i = 1:number_of_time_points
         img_temp = img(:, :, :, i);
         img_temp_2 = img_temp(img_2);
         Y(:, i) = reshape(img_temp_2, length(indices), 1);
    end

    Y = Y';

    if num_cores > 1
        [~, par_workers] = create_parpool(num_cores);
    end
    
    % First we pick a sample of voxels at random and 
    % determine the optimal value of k for each of the voxels.
    % Then we choose the most common k for our universal k.
    % Finally, we run the analysis for regression coefficients and 
    % estimates.
    
    % If only a single value of k is given, this can be skipped.

    if length(k_values) == 1
    
        k_opt_universal = k_values(1);
        k_opt_list_sample = [];
        sample_indices = [];
        
    else
        
        [k_opt_universal, k_opt_list_sample, sample_indices] = ridge_optimal_universal_parameter(Y, regressor_matrix, k_values, sample_fraction, training_sets, num_cores);
    
    end
    
    
    if k_opt_only == 0
    
        nx = size(regressor_matrix, 2);

        B_k_opt_1 = ridge_optimal_regression_coefficients(Y, regressor_matrix, k_opt_universal, num_cores);

        % Rearranging B so that its indices are in line with the original 
        % image.
        dimensions = [number_of_voxels, nx, 1];
        B_k_opt_2 = zeros(dimensions);

        for i = 1:length(indices)
            index = indices(i);
            B_k_opt_2(index, :, :) = B_k_opt_1(i, :, :);
        end

        % Reshaping B_k_opt_2 into the shape of the original image.
        B_k_opt = reshape(B_k_opt_2, siz(1), siz(2), siz(3), nx);

        Z = zscore(regressor_matrix);

        % Calculating an estimate of Y with the universal optimal value 
        % of k.
        dimensions_2 = [prod(siz(1:3)), number_of_time_points];
        Y_hat = zeros(dimensions_2);

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
    
    if num_cores > 1
        
        if ~isempty(par_workers)
            delete(par_workers);
        end
    end
end
