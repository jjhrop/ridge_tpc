# ridge_tpc
Analyzing signal intensity data with ridge regression and cross-validation

The contents of this repository provide Matlab functions for analyzing signal intensity data with ridge regression and cross-validation. The functions have been intended for analysis of brain images in particular, but they may also be suitable for other relevant applications. The fit of the model on the data is validated by cross-validation, more specifically by its K-fold variant. The data may be split into a number of training sets of equal size, each of which is picked in turn as the validation set.

The code has been developed at the Human Emotion Systems Laboratory at the University of Turku and Turku PET Centre.

Author: Jonatan Ropponen
Contributions by Tomi Karjalainen

13 May 2019


Includes functions: ridge_image.m, ridge_tpc.m, ridge_cross_validation.m, ridge_cv_error_calculation.m, ridge_optimal_universal_parameter.m, ridge_optimal_regression_coefficients.m, create_parpool.m (auxiliary function applied in the context of parallel processing)


Example script for ridge_image.m:

image_file_path = '/example_directory/EPI_preprocessed/103/epi_preprocessed.nii';
load('/example_directory/localizer_regressors/Emotion/103.mat'); 
regressor_matrix = R; 
lambda = [0 1 10 100 1000 10^4 10^5 10^6];  
mask_file_path = '/example_directory/Ridge_regression_files/MNI152_T1_2mm_brain_mask.nii'; 
lambda_opt_only = 0; 
sample_fraction = 0.1; 
K = 2; 
cv_randomized = 1; 
num_cores = 1; 
warnings_on = 1;
[Y_hat_lambda_opt, B_lambda_opt, lambda_opt_universal, lambda_opt_list_sample, sample_indices] = ridge_image(image_file_path, regressor_matrix, lambda, mask_file_path, lambda_opt_only, sample_fraction, K, cv_randomized, num_cores, warnings_on);


References for the mathematical formulas adopted for ridge regression and cross-validation:

Regularization: Ridge Regression and the LASSO
University of Stanford; Statistics 305: Autumn Quarter 2006/2007
Accessed 23 May 2018, <http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf>

Ridge Regression: Biased Estimation for Nonorthogonal Problems
Authors: Arthur E. Hoerl and Robert W. Kennard
Source: Technometrics, Vol. 12, No. 1 (Feb. 1970), pp. 55-67.
Published by: Taylor & Francis Ltd.
Accessed 23 May 2018, <https://www.jstor.org/stable/1267351>
