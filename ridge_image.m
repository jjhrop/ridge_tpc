function [Y_hat_lambda_opt, B_lambda_opt, lambda_opt_universal, lambda_opt_list_sample, sample_indices, cv_error_lambda_sample_mean, cv_error_lambda_sample_matrix] = ridge_image(image_file_path, regressor_matrix, lambda, mask_file_path, lambda_opt_only, sample_fraction, K, cv_randomized, num_cores, warnings_on)

    % Analyzing observed responses with the ridge regression function 
    % ridge_tpc. A universal value of parameter lambda is determined for all
    % voxels. The estimate of observed responses is universally calculated 
    % with this value of lambda. 
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
    % lambda: an array containing the lambda values utilized in ridge
    % regression, e.g. [0 0.1 1 10 100 1000 10^4 10^5 10^6].
    %
    % mask_file_path: the path of a .nii file with a mask, e.g. 
    % '/scratch/shared/Matlab_code/Ridge_regression_files/MNI152_T1_2mm_brain_mask.nii'.
    %
    % lambda_opt_only: whether the analysis should only be carried out to 
    % determine the optimal value of lambda. Possible values: 0 and 1. By 
    % default, the analysis is carried out in its entirety.
    %
    % sample_fraction: the fraction of all voxels that is picked for the
    % random sample, e.g. 0.1. Used when determining the universal optimal 
    % value of parameter lambda.
    %
    % K: The number of training sets used in cross-validation. Each set 
    % is treated as the validation set in turn. The data is split evenly 
    % into the sets by its data points. E.g. with K = 2 the two 
    % training sets are the first half and the second half.
    %
    % cv_randomized: Signifies whether the order of the data points is 
    % randomized for cross-validation. For instance, they are not
    % randomized when the data represents a time series, but randomization
    % is more suitable when the data points represent different test 
    % subjects instead. Possible values: 0 and 1. By default, the data 
    % points are randomized.
    %
    % num_cores: the number of cores to be used for parallel processing.
    % Default: 1 (non-parallel).
    %
    % warnings_on: Whether warning messages are displayed. Default: 1.
    %
    % Outputs:
    %
    % Y_hat_lambda_opt: An estimate of the observed responses with the optimal
    % value of lambda.
    %
    % B_lambda_opt: the regression coefficients with the optimal value of lambda.
    %
    % lambda_opt_universal: The value of lambda chosen as the universal value for
    % determining estimates. The value with the lowest mean
    % cross-validation error over sample voxels is chosen.
    %
    % lambda_opt_list_sample: the optimal value of lambda for each voxel in the 
    % sample.
    %
    % sample_indices: the indices of the voxels chosen for the sample.
    %
    % cv_error_lambda_sample_mean: the mean cross-valdiation error over the
    % voxels in the sample for each value of lambda.
    %
    % cv_error_lambda_sample_matrix: a matrix of cross-validation errors
    % for the voxels in the sample and for each value of lambda. The first
    % coordinate corresponds to voxels and the second to the values of
    % lambda.
    %
    % version 2.3, 2019-04-26, Jonatan Ropponen

    % example script
    %
    % image_file_path = '/example_directory/EPI_preprocessed/103/epi_preprocessed.nii';
    % load('/example_directory/localizer_regressors/Emotion/103.mat');
    % regressor_matrix = R;
    % lambda = [0 1 10 100 1000 10^4 10^5 10^6];
    % mask_file_path = '/example_directory/Ridge_regression_files/MNI152_T1_2mm_brain_mask.nii';
    % lambda_opt_only = 0;
    % sample_fraction = 0.1;
    % K = 2;
    % cv_randomized = 1;
    % num_cores = 1;
    % [Y_hat_lambda_opt, B_lambda_opt, lambda_opt_universal, lambda_opt_list_sample, sample_indices] = ridge_image(image_file_path, regressor_matrix, lambda, mask_file_path, lambda_opt_only, sample_fraction, K, cv_randomized, num_cores);

    
    % Default values
    
    if nargin < 10
        warnings_on = 1;
    end
    
    % The default values of lambda if they are not specified in the inputs
    if nargin < 3 || isempty(lambda)
        lambda = [0 1 10 100 1000 10^4 10^5 10^6];
    end
    
    n_lambda = length(lambda);
    
    % Lambda must not be given negative values.
    for i = 1:n_lambda
        if lambda(i) < 0
            lambda(i) = 0;
            
            if warnings_on
                msg = 'Lambda must be non-negative.';
                disp(msg)
            end
        end
    end
    
    if nargin < 5 || isempty(lambda_opt_only)
        lambda_opt_only = 0;
    end
    
    if nargin < 6 || isempty(sample_fraction)
        sample_fraction = 0.1;
    end
    
    if nargin < 7 || isempty(K)
        K = 2;
    end
    
    if nargin < 8
        cv_randomized = 1;
    end
    
    % By default, parallel computing is not used.
    if nargin < 9 || isempty(num_cores) || num_cores < 1
        num_cores = 1;
    end
        
    
    V = spm_vol(image_file_path);
    img = spm_read_vols(V);
    size_original = size(img);
    
    number_of_data_points = size_original(4);
    number_of_voxels = prod(size_original(1:3));
    
    % If no mask has been specified, we determine a mask based on the data.
    if nargin < 4 || isempty(mask_file_path)
    
        % The mean values over the data points
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
    Y = zeros(length(indices), number_of_data_points);

    for i = 1:number_of_data_points
         img_temp = img(:, :, :, i);
         img_temp_2 = img_temp(img_2);
         Y(:, i) = reshape(img_temp_2, length(indices), 1);
    end

    Y = Y';

    if num_cores > 1
        [~, par_workers] = create_parpool(num_cores);
    end
    
    % First we pick a sample of voxels at random and 
    % determine the optimal value of lambda for each of the voxels.
    % Then we choose an universal value for lambda by minimizing the mean 
    % cross-validation error over sample voxels.
    % Finally, we run the analysis for regression coefficients and 
    % estimates.
    
    % If only a single value of lambda is given, this can be skipped.

    if length(lambda) == 1
    
        lambda_opt_universal = lambda(1);
        lambda_opt_list_sample = [];
        sample_indices = [];
        cv_error_lambda_sample_mean = [];
        cv_error_lambda_sample_matrix = [];
        
    else
        
        [lambda_opt_universal, lambda_opt_list_sample, sample_indices, cv_error_lambda_sample_mean, cv_error_lambda_sample_matrix] = ridge_optimal_universal_parameter(Y, regressor_matrix, lambda, sample_fraction, K, cv_randomized, num_cores);
    
    end
    
    
    if lambda_opt_only == 0
    
        nx = size(regressor_matrix, 2);

        B_lambda_opt_1 = ridge_optimal_regression_coefficients(Y, regressor_matrix, lambda_opt_universal, num_cores);

        % Rearranging B so that its indices are in line with the original 
        % image.
        dimensions = [number_of_voxels, nx, 1];
        B_lambda_opt_2 = zeros(dimensions);

        for i = 1:length(indices)
            index = indices(i);
            B_lambda_opt_2(index, :, :) = B_lambda_opt_1(i, :, :);
        end

        % Reshaping B_lambda_opt_2 into the shape of the original image.
        B_lambda_opt = reshape(B_lambda_opt_2, siz(1), siz(2), siz(3), nx);

        % An error is thrown the regressor matrix contains non-zero columns.
        cols_with_all_zeros = all(regressor_matrix == 0);
        cols_with_all_zeros_exist = sum(cols_with_all_zeros) > 0;

        if cols_with_all_zeros_exist
            msg = 'The columns in the design matrix must be non-negative.';
            error(msg)
        end
        
        if warnings_on

            % A warning message is displayed if the regressor matrix 
            % contains linearly dependent columns.
            if rank(regressor_matrix) < size(regressor_matrix, 2)
                msg = 'The design matrix contains linearly dependent columns.';
                disp(msg)
            end     
        end
        
        Z = zscore(regressor_matrix);

        % Calculating an estimate of Y with the universal optimal value 
        % of lambda.
        dimensions_2 = [prod(siz(1:3)), number_of_data_points];
        Y_hat = zeros(dimensions_2);

        for i = 1:size(B_lambda_opt_2, 1)

            b_lambda_opt = B_lambda_opt_2(i, :);
            y_hat = Z * b_lambda_opt';
            Y_hat(i, :) = y_hat;

        end

        Y_hat_lambda_opt = reshape(Y_hat, siz(1), siz(2), siz(3), number_of_data_points);
    
    else
        
        Y_hat_lambda_opt = {[]};
        B_lambda_opt = {[]};
    
    end
    
    if num_cores > 1
        
        if ~isempty(par_workers)
            delete(par_workers);
        end
    end
end

