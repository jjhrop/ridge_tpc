function [b_k, sigma_b, k_opt, b_k_opt] = ridge_tpc(y, X, k, calculate_sigma)

    % A function for estimating ridge regression coefficients.
    %
    % Optional input:
    %
    % calculate_sigma: Signifies whether sigma_b should be calculated.
    % Possible values: 0 and 1. By default, we proceed with the
    % calculation.
    %
    % version 2.0, 2018-11-13; Jonatan Ropponen, Tomi Karjalainen
    
    % First center y
    ymean = mean(y);
    y_centered = y - ymean;

    % Standardize the design matrix
    Z = zscore(X);

    % Estimate the ridge coefficients
    p = size(X, 2);
    nk = length(k);
    b_k = zeros(p, nk);
    [U, S, V] = svd(Z, 'econ');
    d = diag(S);
    A = U' * y_centered;
       
    for i = 1:nk
        di = d./(d.^2 + k(i));
        b_k(:, i) = V * diag(di) * A;
    end

    % Visualize the coefficients as a function of k
    %figure(); plot(k,b_k); xlabel('k'); ylabel('Beta'); box off; title('Ridge coefficient paths');
    %figure(); plot(k,b_k); xlabel('k'); ylabel('Beta'); box off; legend(1:p); title('Ridge coefficient paths');

    
    % If only a single value of k has been provided, we can skip this 
    % section.
    
    if nk == 1
    
        k_opt = k(1);
        
    else
    
        % Choosing the optimal k by cross-validation

        % Applying K-fold cross-validation, i.e. cross-validation with K
        % subsets of elements, removing each subset in turn

        % example
        %K = 3;

        % N = size(X, 1);
        % indices = randperm(N);
        % number_of_indices = length(indices);
        % subset_size = ceil(N / K);
        %      
        % list_of_indices = {};
        %
        % for i = 1:K
        %
        %     first_index = (i-1) * subset_size + 1;
        %     last_index = min(i * subset_size, number_of_indices);
        %
        %     list_of_indices = [list_of_indices, {indices(first_index : last_index)}];
        %  end


        % Implemented with K = 1 (two subsets in total)
        
        % Splitting the data into two subsets by its timepoints, at 
        % proportion L of the entire data. We always treat the temporally 
        % prior subset as the training dataset and the latter as the 
        % validation set. In other words, the second subset is removed to 
        % form the training dataset.

        K = 1;

        % The proportion of the splitting point can be adjusted between 0 
        % and 1.
        L = 0.6;

        N = size(X, 1);
        splitting_point = floor(N * L);

        list_of_indices = [{splitting_point + 1 : N}, {1 : splitting_point}];


        for i = 1:K

            current_indices = cell2mat(list_of_indices(i));


            % Matrices containing only the ith subset
            X_i = X(current_indices, :);
            y_i = y(current_indices, :);

            y_i_mean = mean(y_i);
            y_i_centered = y_i - y_i_mean;

            Z_i = zscore(X_i);

            excluded_set_size = size(Z_i, 1);


            % Matrices with the ith subset removed
            X_neg_i = X;
            y_neg_i = y;

            X_neg_i(current_indices, :) = [];
            y_neg_i(current_indices, :) = [];


            % Compiling the values again after the removal

            % First center y
            y_neg_i_mean = mean(y_neg_i);
            y_neg_i_centered = y_neg_i - y_neg_i_mean;

            % Standardize the design matrix
            Z_neg_i = zscore(X_neg_i);

            % Estimate the ridge coefficients
            p_neg_i = size(X_neg_i, 2);     
            b_k_neg_i = zeros(p_neg_i, nk);
            [U_neg_i, S_neg_i, V_neg_i] = svd(Z_neg_i, 'econ');
            d_neg_i = diag(S_neg_i);
            A_neg_i = U_neg_i' * y_neg_i_centered;


            f_k_neg_i = zeros(excluded_set_size, 1);

            for j = 1:nk
                di_neg_i = d_neg_i ./ (d_neg_i.^2 + k(j));
                b_k_neg_i(:, j) = V_neg_i * diag(di_neg_i) * A_neg_i;

                % Calculating the cross validation error.
                % Note that the values of beta are applied on the excluded
                % subset.
                f_k_neg_i(:, j) = Z_i * b_k_neg_i(:, j);
                cv_error_k_i(i, j) = (excluded_set_size)^(-1) * sum((y_i_centered - f_k_neg_i(:, j)).^2);
            end
        end

        % The error overall for each value of k
        for i = 1:nk
            cv_error_k(i) = K^(-1) * sum(cv_error_k_i(:, i));
        end

        [~, opt_idx] = min(cv_error_k);
        k_opt = k(opt_idx(1));
    
    end
    
    % The ridge coefficients with the optimal value of k
    p = size(X, 2);
    k_index = find(k == k_opt);
    b_k_opt = b_k(:, k_index);
    
    if nargin < 4 || calculate_sigma == 1
    
        % Calculating the degrees of freedom.
        % (Not currently used; perhaps could be implemented in the 
        % estimation of sigma.)        
        %df = sum((d.^2) / (d.^2 + k_opt));

        % Calculating the residuals
        H = Z' * Z + k_opt * eye(p);
        yhat = Z * (H \ (Z' * y));
        res = y - yhat;

        % Estimating the standard deviation
        sse = sum(res.^2);
        sigma = sse / length(res); % Should df be taken into account?
        W = H \ Z' * Z;
        var_b = sigma^2 * W * ((Z' * Z) \ W');
        sigma_b = sqrt(var_b);
    
    else    
        
        sigma_b = NaN;
        
    end

end