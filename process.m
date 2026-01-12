%% 1. Load Data
% Ensure your 'rsm data.csv' is in the current folder.
data = readtable('rsm data.csv');

% Define Factors and Response
X = [data.temperature, data.time];
y = data.yield; % Change to data.adsorption for the other response

%% 2. Initialize Models
% We use Cross-Validation (K-Fold = 5) due to the small sample size (n=13)
cv_num = 5;

% Model A: Random Forest (Bagged Trees)
mdlRF = fitrensemble(X, y, 'Method', 'Bag', 'NumLearningCycles', 100);

% Model B: Gradient Boosting (Boosted Trees)
mdlGB = fitrensemble(X, y, 'Method', 'LSBoost', 'NumLearningCycles', 100);

% Model C: Support Vector Machine (Gaussian Kernel)
mdlSVM = fitrsvm(X, y, 'KernelFunction', 'gaussian', 'Standardize', true);

%% 3. Cross-Validation Performance
% Perform cross-validation for each model
cvRF  = crossval(mdlRF, 'KFold', cv_num);
cvGB  = crossval(mdlGB, 'KFold', cv_num);
cvSVM = crossval(mdlSVM, 'KFold', cv_num);

% Calculate Mean Squared Error
mseRF  = kfoldLoss(cvRF);
mseGB  = kfoldLoss(cvGB);
mseSVM = kfoldLoss(cvSVM);

% Predict for Plotting
y_predRF  = kfoldPredict(cvRF);
y_predGB  = kfoldPredict(cvGB);
y_predSVM = kfoldPredict(cvSVM);

%% 4. Calculate R-Squared
% Helper function to calculate R2: 1 - (SS_res / SS_tot)
calcR2 = @(actual, pred) 1 - sum((actual - pred).^2) / sum((actual - mean(actual)).^2);

r2RF  = calcR2(y, y_predRF);
r2GB  = calcR2(y, y_predGB);
r2SVM = calcR2(y, y_predSVM);

%% 5. Visualization
figure('Position', [100, 100, 1200, 400]);

% Random Forest Plot
subplot(1,3,1);
scatter(y, y_predRF, 'filled', 'MarkerFaceColor', [1 0.6 0.6]); hold on;
plot([min(y) max(y)], [min(y) max(y)], 'r--', 'LineWidth', 2);
title(['Random Forest (R^2 = ', num2mstr(r2RF, 3), ')']);
xlabel('Experimental Yield'); ylabel('Predicted Yield'); grid on;

% Gradient Boosting Plot
subplot(1,3,2);
scatter(y, y_predGB, 'filled', 'MarkerFaceColor', [0.4 0.7 1]); hold on;
plot([min(y) max(y)], [min(y) max(y)], 'r--', 'LineWidth', 2);
title(['Gradient Boosting (R^2 = ', num2str(r2GB, 3), ')']);
xlabel('Experimental Yield'); grid on;

% SVM Plot
subplot(1,3,3);
scatter(y, y_predSVM, 'filled', 'MarkerFaceColor', [0.6 1 0.6]); hold on;
plot([min(y) max(y)], [min(y) max(y)], 'r--', 'LineWidth', 2);
title(['SVM (R^2 = ', num2str(r2SVM, 3), ')']);
xlabel('Experimental Yield'); grid on;

%% 6. Feature Importance (Using Random Forest)
figure;
imp = predictorImportance(mdlRF);
bar(imp, 'FaceColor', [0.2 0.2 0.5]);
set(gca, 'xticklabel', {'Temperature', 'Time'});
title('Feature Importance (Biochar Factors)');
ylabel('Importance Score');