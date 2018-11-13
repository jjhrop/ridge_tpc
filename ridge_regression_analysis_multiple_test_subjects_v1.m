function [Y_hat_k_opt_list, B_k_opt_list, k_opt_universal, k_opt_by_file, k_opt_by_voxel] = ridge_regression_analysis_multiple_test_subjects_v1(image_file_path_list, regressor_matrix_list, k_values, default_mask_applied, mask_file_path, num_cores)

    % Performs ridge regression analysis for multiple test subjects. An
    % optimal value of k is chosen universally for all the examined test
    % subjects, and the analysis is carried out with that value.
    %
    % Inputs:
    %
    % image_file_path_list: the paths of the .nii files to be examined for 
    % each test subject, e.g. 
    % '/scratch/shared/Matlab_code/EPI_preprocessed/103/epi_preprocessed.nii'.
    % Provided as a cell array.
    %
    % regressor_matrix_list: the matrices containing regressors for each 
    % test subject, e.g. extracted from
    % '/scratch/shared/Matlab_code/localizer_regressors/Emotion/103.mat'.
    % Provided as a cell array.
    %
    % Optional inputs:
    %
    % k_values: an array containing the k values utilized in ridge
    % regression, e.g. [0 0.1 1 10 100 1000].
    %
    % default_mask_applied: whether a default mask should be applied. 
    % Possible values: 0 and 1. By default, the mask is applied.
    %
    % mask_file_path: the path of a .nii file with a mask, e.g. 
    % '/scratch/shared/Matlab_code/Ridge_regression_files/MNI152_T1_2mm_brain_mask.nii'.
    %
    % num_cores: the number of cores to be used for parallel processing
    % (for ridge_regression_analysis). Default: 1 (non-parallel).
    %
    % Outputs:
    %
    % Y_hat_k_opt_list: the estimates of the signal intensities for each 
    % image file with the optimal value of k.
    %
    % B_k_opt_list: the regression coefficients with the optimal value of 
    % k.
    %
    % k_opt_universal: the value of k chosen as the universal value for
    % determining estimates. The most frequent of the optimal values of k 
    % for each voxel.
    %
    % k_opt_by_file: the optimal values of k for each image file.
    %
    % k_opt_by_voxel: the optimal values of k for each voxel.
    
    % version 1.0, 2018-11-13, Jonatan Ropponen

    % example script
    % image_file_path_list = [{'/scratch/shared/Matlab_code/EPI_preprocessed/103/epi_preprocessed.nii'}, {'/scratch/shared/Matlab_code/EPI_preprocessed/107/epi_preprocessed.nii'}];
    % load('/scratch/shared/Matlab_code/localizer_regressors/Emotion/103.mat');
    % regressor_matrix_103 = R;
    % load('/scratch/shared/Matlab_code/localizer_regressors/Emotion/107.mat');
    % regressor_matrix_107 = R;
    % regressor_matrix_list = [{regressor_matrix_103}, {regressor_matrix_107}];
    % k_values = [0 0.1 1 10 100 1000];
    % default_mask_applied = 1;
    % mask_file_path = '/scratch/shared/Matlab_code/Ridge_regression_files/MNI152_T1_2mm_brain_mask.nii';
    % num_cores = 1;
    
    tic

    % The default values of k if they are not specified in the inputs
    if nargin < 3
        k_values = [0 0.1 1 10 100 1000];
    end
    
    % By default, parallel computing is not used.
    if nargin < 7 || num_cores < 1
        num_cores = 1;
    end
    
    if nargin < 4 
        default_mask_applied = 1;
    end
    
    % The default mask is applied if specified or if no input is given on
    % whether to apply it.
    if nargin < 4 || default_mask_applied ~= 0
        mask_file_path = '/scratch/shared/Matlab_code/Ridge_regression_files/MNI152_T1_2mm_brain_mask.nii';
    end
    
    % The list of image files paths and the list of regressor matrices 
    % should be equal in length.
    if ~isempty(image_file_path_list) && ~isempty(regressor_matrix_list) && length(image_file_path_list) ~= length(regressor_matrix_list)
    
        msg = 'The list of image file paths and the list of regressor matrices must be equal in length.';
        disp(msg);
    end

    Y_hat_k_opt_list = {};
    B_k_opt_list = {};
    k_opt_by_file = [];
    k_opt_by_voxel = [];
    
    %[~, par_workers] = create_parpool(5);

    % Running the analysis for each image file and regressor matrix to
    % determine the optimal value of k that will be applied universally.
    
    % For now, we are only determining the optimal k, not the estimates.
    k_opt_only = 1;
    
    % parfor i = 1:length(image_file_path_list)
    for i = 1:length(image_file_path_list)
        
        image_file_path = cell2mat(image_file_path_list(i));
        regressor_matrix = cell2mat(regressor_matrix_list(i));
        
        if ~isempty(image_file_path) && ~isempty(regressor_matrix)
            
            [~, ~, k_opt_universal_current, k_opt_list_sample] = ridge_regression_analysis(image_file_path, regressor_matrix, k_values, default_mask_applied, mask_file_path, k_opt_only, num_cores);
            
            k_opt_by_file(i) = k_opt_universal_current;
            k_opt_by_voxel = [k_opt_by_voxel, k_opt_list_sample];
        
        else
            
            k_opt_by_file(i) = NaN;
            
        end  
    end
    
    % We choose the value of k that is optimal for the greatest number of
    % voxels. (Alternative: the value that is optimal for the greatest 
    % number of files.)
    
    k_opt_universal = mode(k_opt_by_voxel);
    %k_opt_universal = mode(k_opt_by_file);
    
    
    % Running the analysis for each image file and regressor file with the
    % optimal k.
    
    k_opt_only = 0;
    
    % parfor i = 1:length(image_file_path_list)
    for i = 1:length(image_file_path_list)
        
        image_file_path = cell2mat(image_file_path_list(i));
        regressor_matrix = cell2mat(regressor_matrix_list(i));
        
        if ~isempty(image_file_path) && ~isempty(regressor_matrix)
        
            [Y_hat_k_opt, B_k_opt] = ridge_regression_analysis(image_file_path, regressor_matrix, k_opt_universal, default_mask_applied, mask_file_path, k_opt_only, num_cores);

            Y_hat_k_opt_list(i) = {Y_hat_k_opt};
            B_k_opt_list(i) = {B_k_opt};
        
        else
            
            Y_hat_k_opt_list(i) = {[]};
            B_k_opt_list(i) = {[]};
            
        end  
    end

    %if ~isempty(par_workers)
    %    delete(par_workers);
    %end
    
    toc

end

