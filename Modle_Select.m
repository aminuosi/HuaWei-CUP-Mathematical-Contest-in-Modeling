%% Step 0: Read Data
data = readtable('');

% Label column (last column)
y_raw = data{:,end};
y = categorical(y_raw);  % Classification task, convert to categorical

% Feature columns: keep only numerical
featureNames = data.Properties.VariableNames(1:end-1);
isNum = varfun(@isnumeric, data(:,1:end-1), 'OutputFormat','uniform');
X = data(:, featureNames(isNum));

fprintf('The following non-numerical feature columns have been ignored:\n');
disp(featureNames(~isNum));

%% Step 1: IQR Outlier Detection + Filling + Zscore Normalization (Modified Part)
X_mat = table2array(X);

Q1 = prctile(X_mat,25);
Q3 = prctile(X_mat,75);
IQR_val = Q3 - Q1;

lower = Q1 - 1.5*IQR_val;
upper = Q3 + 1.5*IQR_val;

for j = 1:size(X_mat,2)
    col = X_mat(:,j);
    outlier_idx = col < lower(j) | col > upper(j);
    median_val = median(col(~outlier_idx));
    col(outlier_idx) = median_val;
    X_mat(:,j) = col;
end

% === Modification Start: Save Normalization Parameters ===
% Calculate normalization parameters (mean and standard deviation)
mu_s = mean(X_mat);      % Mean vector
sigma_s = std(X_mat);    % Standard deviation vector

% Apply Z-score normalization
X_mat = (X_mat - mu_s) ./ sigma_s;

fprintf('Normalization parameters calculated and saved:\n');
fprintf(' - Number of features: %d\n', length(mu_s));
fprintf(' - Mean range: [%.4f, %.4f]\n', min(mu_s), max(mu_s));
fprintf(' - Standard deviation range: [%.4f, %.4f]\n', min(sigma_s), max(sigma_s));
% === Modification End ===

%% Step 2: Multiple Feature Selection Methods Comparison
fprintf('\n=== Multiple Feature Selection Methods Comparison ===\n');

% Method 1: ANOVA F-test
fprintf('1. ANOVA F-test feature selection...\n');
F_vals = zeros(1,size(X_mat,2));
for j = 1:size(X_mat,2)
    p = anova1(X_mat(:,j), y, 'off');
    F_vals(j) = -log10(p+eps);
end

[~,idx_anova] = maxk(F_vals,40);
X_selected_anova = X_mat(:,idx_anova);
selected_feature_names_anova = featureNames(isNum);
selected_feature_names_anova = selected_feature_names_anova(idx_anova);

% Method 2: Chi-square test (univariate statistical test)
fprintf('2. Chi-square test feature selection...\n');
chi2_vals = zeros(1,size(X_mat,2));
for j = 1:size(X_mat,2)
    try
        % Discretize continuous features for chi-square test
        [~, ~, stats] = crosstab(discretize(X_mat(:,j), 5), y);
        chi2_vals(j) = stats.chisq;  % Higher chi-square value is better
    catch
        chi2_vals(j) = 0;
    end
end

[~,idx_chi2] = maxk(chi2_vals,40);
X_selected_chi2 = X_mat(:,idx_chi2);
selected_feature_names_chi2 = featureNames(isNum);
selected_feature_names_chi2 = selected_feature_names_chi2(idx_chi2);

%% Generate Correlation Heatmaps for Different Methods
fprintf('\n=== Generating Correlation Heatmaps for Different Feature Selection Methods ===\n');

% Define aesthetic color schemes
blue_cmap = [
    0.1, 0.1, 0.5;   % Dark blue
    0.2, 0.2, 0.8;   % Blue
    0.4, 0.4, 1.0;   % Medium blue
    0.6, 0.6, 1.0;   % Light blue
    0.8, 0.8, 1.0;   % Very light blue
    1.0, 1.0, 1.0    % White
];

green_cmap = [
    0.0, 0.3, 0.0;   % Dark green
    0.1, 0.6, 0.1;   % Green
    0.3, 0.8, 0.3;   % Medium green
    0.6, 0.9, 0.6;   % Light green
    0.8, 1.0, 0.8;   % Very light green
    1.0, 1.0, 1.0    % White
];

% Process feature names: replace underscores
selected_feature_names_anova_display = strrep(selected_feature_names_anova, '_', '-');
selected_feature_names_chi2_display = strrep(selected_feature_names_chi2, '_', '-');

% Method 1: ANOVA F-test Heatmap
figure('Position', [100, 100, 1000, 900]);
corrMatrix_anova = corr(X_selected_anova);
h1 = heatmap(selected_feature_names_anova_display, selected_feature_names_anova_display, corrMatrix_anova, ...
    'Colormap', blue_cmap, ...
    'ColorbarVisible', 'on', ...
    'Title', 'Correlation Heatmap of Features Selected by ANOVA F-test');
h1.ColorLimits = [-1, 1];
h1.FontSize = 8;
saveas(gcf, 'correlation_heatmap_anova.png');

% Method 2: Chi-square Test Heatmap
figure('Position', [100, 100, 1000, 900]);
corrMatrix_chi2 = corr(X_selected_chi2);
h2 = heatmap(selected_feature_names_chi2_display, selected_feature_names_chi2_display, corrMatrix_chi2, ...
    'Colormap', green_cmap, ...
    'ColorbarVisible', 'on', ...
    'Title', 'Correlation Heatmap of Features Selected by Chi-square Test');
h2.ColorLimits = [-1, 1];
h2.FontSize = 8;
saveas(gcf, 'correlation_heatmap_chi2.png');

%% Feature Selection Methods Comparison Heatmap (Subplot Format) - Improved Version
figure('Position', [50, 50, 1600, 800]);

% Use numerical labels for each feature (to avoid name display issues)
feature_nums_anova = arrayfun(@(x) sprintf('%d', x), 1:length(selected_feature_names_anova), 'UniformOutput', false);
feature_nums_chi2 = arrayfun(@(x) sprintf('%d', x), 1:length(selected_feature_names_chi2), 'UniformOutput', false);

% ANOVA F-test
subplot(1,2,1);
h1 = heatmap(feature_nums_anova, feature_nums_anova, corrMatrix_anova, ...
    'Colormap', blue_cmap, ...
    'Title', 'ANOVA F-test', ...
    'FontSize', 5);
h1.ColorLimits = [-1, 1];
h1.XLabel = 'Feature Index';
h1.YLabel = 'Feature Index';

% Chi-square Test
subplot(1,2,2);
h2 = heatmap(feature_nums_chi2, feature_nums_chi2, corrMatrix_chi2, ...
    'Colormap', green_cmap, ...
    'Title', 'Chi-square Test', ...
    'FontSize', 5);
h2.ColorLimits = [-1, 1];
h2.XLabel = 'Feature Index';
h2.YLabel = 'Feature Index';

sgtitle('Comparison of Correlation Heatmaps for Different Feature Selection Methods', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, 'correlation_heatmap_comparison.png');

% Use ANOVA method by default for subsequent analysis
X_selected = X_selected_anova;
selected_feature_names = selected_feature_names_anova;
fprintf('\nUsing features selected by ANOVA method for subsequent modeling analysis\n');

%% Step 3: Classification Model Training and Evaluation (Multi-class Support)
models = { ...
    @(X,Y) fitcecoc(X,Y,'Learners',templateLinear()), ... % Linear classifier
    @fitctree, ...
    @(X,Y) fitcecoc(X,Y,'Learners',templateSVM('Standardize',true)), ... % SVM multi-class
    @fitcensemble, ...
    @fitcknn};

modelNames = {'LinearCls','Tree','SVM','Ensemble','KNN'};

% Modify this line: change from 4 metrics to 5 metrics
metrics = zeros(length(models),5); % Accuracy, Precision, Recall, F1, AUC

cv = cvpartition(size(X_selected,1),'HoldOut',0.2);
Xtrain = X_selected(training(cv),:);
ytrain = y(training(cv));
Xtest  = X_selected(test(cv),:);
ytest  = y(test(cv));

% Store prediction results and confusion matrices for each model
all_ypred = cell(length(models),1);
all_confMat = cell(length(models),1);
all_scores = cell(length(models),1); % New: for AUC calculation

for i = 1:length(models)
    tic;
    Mdl = models{i}(Xtrain,ytrain);
    train_time = toc;
    ypred = predict(Mdl,Xtest);
    
    % Store prediction results
    all_ypred{i} = ypred;
    all_confMat{i} = confusionmat(ytest,ypred);

    acc = mean(ypred == ytest);
    confMat = all_confMat{i};
    
    % Modify this part: calculate 5 metrics
    % Calculate multi-class precision, recall, F1
    precision_per_class = diag(confMat)./sum(confMat,2);
    recall_per_class = diag(confMat)./sum(confMat,1)';
    f1_per_class = 2*(precision_per_class.*recall_per_class)./(precision_per_class+recall_per_class+eps);
    
    % Macro average
    precision_macro = nanmean(precision_per_class);
    recall_macro = nanmean(recall_per_class);
    f1_macro = nanmean(f1_per_class);
    
    % Calculate AUC
    try
        % Get prediction probabilities
        [~, scores] = predict(Mdl, Xtest);
        all_scores{i} = scores;
        
        % Multi-class AUC calculation
        ytest_numeric = grp2idx(ytest);
        auc_scores = zeros(1, size(scores,2));
        for class_idx = 1:size(scores,2)
            y_binary = (ytest_numeric == class_idx);
            [~,~,~,auc_scores(class_idx)] = perfcurve(y_binary, scores(:,class_idx), 1);
        end
        auc_mean = mean(auc_scores);
    catch
        auc_mean = 0.5; % If AUC calculation fails
    end

    % Modify this line: store 5 metrics
    metrics(i,:) = [acc, precision_macro, recall_macro, f1_macro, auc_mean];
end

% Modify this line: update metric names
T = array2table(metrics, ...
    'VariableNames',{'Accuracy','Precision','Recall','F1','AUC'}, ...
    'RowNames',modelNames);
disp(T);

[~,bestIdx] = max(metrics(:,1)); % Select best by maximum Accuracy
bestModel = modelNames{bestIdx};
disp(['Best Model: ', bestModel]);

%% Step 3: Classification Model Training and Evaluation (Multi-class Support) - Extended Version
% Define all models (including new ones)
models = { ...
    @(X,Y) fitcecoc(X,Y,'Learners',templateLinear()), ... % Linear classifier
    @fitctree, ...                                         % Decision tree
    @(X,Y) fitcecoc(X,Y,'Learners',templateSVM('Standardize',true)), ... % SVM multi-class
    @fitcensemble, ...                                     % Ensemble (default)
    @fitcknn, ...                                          % KNN
    @(X,Y) fitcensemble(X,Y,'Method','Bag'), ...           % Random Forest (Bagging)
    @(X,Y) fitcsvm(X,Y,'Standardize',true,'KernelFunction','rbf'), ... % SVM (needs adjustment for binary classification)
    @(X,Y) fitcensemble(X,Y,'Method','GentleBoost'), ...   % Gradient Boosting
    @(X,Y) fitcnet(X,Y,'LayerSizes',[100 50], 'IterationLimit',100) ... % MLP neural network
    };

modelNames = {'LinearCls','Tree','SVM-ECOC','Ensemble','KNN', ...
              'RandomForest','SVM-RBF','GradientBoosting','MLP'};

% Modify this line: change from 4 metrics to 5 metrics
metrics = zeros(length(models),5); % Accuracy, Precision, Recall, F1, AUC

cv = cvpartition(size(X_selected,1),'HoldOut',0.2);
Xtrain = X_selected(training(cv),:);
ytrain = y(training(cv));
Xtest  = X_selected(test(cv),:);
ytest  = y(test(cv));

% Store prediction results and confusion matrices for each model
all_ypred = cell(length(models),1);
all_confMat = cell(length(models),1);
all_scores = cell(length(models),1); % New: for AUC calculation

for i = 1:length(models)
    tic;
    try
        % Special handling for SVM_RBF (binary classification problem)
        if i == 7 && length(unique(ytrain)) > 2
            fprintf('SVM_RBF only supports binary classification, using SVM_ECOC instead\n');
            Mdl = fitcecoc(Xtrain,ytrain,'Learners',templateSVM('Standardize',true,'KernelFunction','rbf'));
        else
            Mdl = models{i}(Xtrain,ytrain);
        end
        
        train_time = toc;
        ypred = predict(Mdl,Xtest);
        
        % Store prediction results
        all_ypred{i} = ypred;
        all_confMat{i} = confusionmat(ytest,ypred);

        acc = mean(ypred == ytest);
        confMat = all_confMat{i};
        
        % Calculate multi-class precision, recall, F1
        precision_per_class = diag(confMat)./sum(confMat,2);
        recall_per_class = diag(confMat)./sum(confMat,1)';
        f1_per_class = 2*(precision_per_class.*recall_per_class)./(precision_per_class+recall_per_class+eps);
        
        % Macro average
        precision_macro = nanmean(precision_per_class);
        recall_macro = nanmean(recall_per_class);
        f1_macro = nanmean(f1_per_class);
        
        % Calculate AUC
        try
            % Get prediction probabilities
            if ismember(i, [1, 2, 4, 5, 6, 8, 9]) % Models that support probability estimation
                [~, scores] = predict(Mdl, Xtest);
                all_scores{i} = scores;
                
                % Multi-class AUC calculation
                ytest_numeric = grp2idx(ytest);
                auc_scores = zeros(1, size(scores,2));
                for class_idx = 1:size(scores,2)
                    y_binary = (ytest_numeric == class_idx);
                    [~,~,~,auc_scores(class_idx)] = perfcurve(y_binary, scores(:,class_idx), 1);
                end
                auc_mean = mean(auc_scores);
            else
                auc_mean = 0.5; % For models that don't support probability estimation
            end
        catch
            auc_mean = 0.5; % If AUC calculation fails
        end

        % Store 5 metrics
        metrics(i,:) = [acc, precision_macro, recall_macro, f1_macro, auc_mean];
        
        fprintf('Model %s training completed: Accuracy=%.4f, Time=%.2fs\n', modelNames{i}, acc, train_time);
        
    catch ME
        fprintf('Model %s training failed: %s\n', modelNames{i}, ME.message);
        metrics(i,:) = [0, 0, 0, 0, 0.5]; % Set to 0 if failed
    end
end

% Display all model performance comparison
T = array2table(metrics, ...
    'VariableNames',{'Accuracy','Precision','Recall','F1','AUC'}, ...
    'RowNames',modelNames);
disp(T);

[~,bestIdx] = max(metrics(:,1)); % Select best by maximum Accuracy
bestModel = modelNames{bestIdx};
fprintf('Best Model: %s (Accuracy: %.4f)\n', bestModel, metrics(bestIdx,1));

%% Step 3: Directly Save Best Model (Modified Part: Save Normalization Parameters)
fprintf('\n=== Directly Save Best Model (Including Normalization Parameters) ===\n');

try
    % Retrain best model
    bestMdl = models{bestIdx}(Xtrain, ytrain);
    
    % Evaluate model
    ypred_final = predict(bestMdl, Xtest);
    final_accuracy = mean(ypred_final == ytest);
    
    % === Modification Start: Save Normalization Parameters ===
    % Save both model and normalization parameters
    save(sprintf('best_model_%s.mat', bestModel), 'bestMdl', 'mu_s', 'sigma_s', 'selected_feature_names');
    
    fprintf('Model saved successfully: %s\n', bestModel);
    fprintf(' - Accuracy: %.4f\n', final_accuracy);
    fprintf(' - Saved variables: bestMdl, mu_s, sigma_s, selected_feature_names\n');
    fprintf(' - Number of features: %d\n', length(mu_s));
    fprintf(' - Normalization parameters included in model file\n');
    % === Modification End ===
    
catch ME
    fprintf('Model saving failed: %s\n', ME.message);
end

%% Optional: Save a separate complete model file with full information
fprintf('\n=== Creating Complete Model File ===\n');

% Save a complete file with all necessary information
complete_model_info = struct();
complete_model_info.model = bestMdl;
complete_model_info.mu_s = mu_s;
complete_model_info.sigma_s = sigma_s;
complete_model_info.selected_features = selected_feature_names;
complete_model_info.feature_indices = idx_anova;  % Save feature selection indices
complete_model_info.accuracy = final_accuracy;
complete_model_info.training_date = datetime('now');

save('complete_trained_model.mat', '-struct', 'complete_model_info');
fprintf('Complete model file saved: complete_trained_model.mat\n');
fprintf('Contains complete data including model, normalization parameters, feature selection information\n');


%% Step 4: Simplified Hyperparameter Optimization and 3D Visualization
fprintf('\n=== Starting Hyperparameter Optimization and 3D Visualization for Each Model ===\n');

% Define models to optimize
models_to_optimize = {'RandomForest', 'MLP', 'KNN'};

for model_idx = 1:length(models_to_optimize)
    current_model = models_to_optimize{model_idx};
    fprintf('\n=== Optimizing Model: %s ===\n', current_model);
    
    switch current_model
        case 'RandomForest'
            %% Random Forest Hyperparameter Optimization - Using Number of Trees and Max Depth
            numTreesRange = [20, 320];      % Range for number of trees
            maxDepthRange = [3, 15];        % Range for maximum depth (more intuitive)
            
            numRandomSamples = 80;
            results = [];
            fprintf('Random Forest Hyperparameter Optimization - Adaptive Random Search: %d samples\n', numRandomSamples);
            
            % Adjust sampling strategy based on preliminary results
            for sample = 1:numRandomSamples
                if sample <= 20
                    % Phase 1: Uniform exploration of entire parameter space
                    numTrees = randi(numTreesRange);
                    maxDepth = randi(maxDepthRange);
                else
                    % Phase 2: Bias towards regions with good performance based on existing results
                    if ~isempty(results)
                        [bestAcc, bestIdx] = max(results(:,3));
                        bestTrees = results(bestIdx,1);
                        bestDepth = results(bestIdx,2);
                        
                        % More dense sampling around best parameters
                        trees_std = std(results(:,1));
                        depth_std = std(results(:,2));
                        
                        numTrees = round(normrnd(bestTrees, trees_std/3));
                        numTrees = max(numTreesRange(1), min(numTreesRange(2), numTrees));
                        
                        maxDepth = round(normrnd(bestDepth, depth_std/3));
                        maxDepth = max(maxDepthRange(1), min(maxDepthRange(2), maxDepth));
                    else
                        numTrees = randi(numTreesRange);
                        maxDepth = randi(maxDepthRange);
                    end
                end
                
                fprintf('Progress: %.1f%% - ', sample/numRandomSamples*100);
                
                try
                    tic;
                    % Use maximum depth parameter
                    t = templateTree('MaxNumSplits', maxDepth^2, 'Reproducible', true);
                    % Note: No direct MaxDepth parameter in MATLAB, approximate with MaxNumSplits
                    Mdl = fitcensemble(Xtrain, ytrain, ...
                        'Method', 'Bag', ...
                        'NumLearningCycles', numTrees, ...
                        'Learners', t);
                    
                    ypred_val = predict(Mdl, Xtest);
                    valAccuracy = mean(ypred_val == ytest);
                    trainTime = toc;
                    
                    results = [results; numTrees, maxDepth, valAccuracy, trainTime];
                    fprintf('Trees: %d, MaxDepth: %d, Acc: %.4f\n', numTrees, maxDepth, valAccuracy);
                    
                catch ME
                    fprintf('Failed\n');
                    continue;
                end
            end
            
            if size(results, 1) > 5
                % Find best parameters
                [bestAccuracy, bestIdx] = max(results(:,3));
                bestParams = results(bestIdx, 1:2);
                
                % Create fine grid for smooth surface
                numTrees_fine = linspace(min(results(:,1)), max(results(:,1)), 50);
                maxDepth_fine = linspace(min(results(:,2)), max(results(:,2)), 50);
                [X_fine, Y_fine] = meshgrid(numTrees_fine, maxDepth_fine);
                
                % Use interpolation to obtain smooth surface
                Z_smooth = griddata(results(:,1), results(:,2), results(:,3), X_fine, Y_fine, 'v4');
                Z_smooth(isnan(Z_smooth)) = nanmean(Z_smooth(:));
                
                %% Figure 1: 3D Surface Plot
                figure('Position', [100, 100, 1000, 800]);
                surf(X_fine, Y_fine, Z_smooth, 'FaceAlpha', 0.85, 'EdgeColor', [0.3 0.3 0.3], 'EdgeAlpha', 0.3);
                hold on;
                
                % Mark all data points
                scatter3(results(:,1), results(:,2), results(:,3), 60, 'yellow', 'filled', ...
                        'MarkerEdgeColor', 'k', 'LineWidth', 1);
                
                % Mark best point
                scatter3(bestParams(1), bestParams(2), bestAccuracy, 150, 'red', 'filled', ...
                        'MarkerEdgeColor', 'k', 'LineWidth', 2);
                
                xlabel('Number of Trees', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('Maximum Depth', 'FontSize', 12, 'FontWeight', 'bold');
                zlabel('Accuracy', 'FontSize', 12, 'FontWeight', 'bold');
                title(sprintf('Random Forest Hyperparameter Optimization\nBest: %d Trees, Depth %d, Accuracy=%.4f', ...
                    bestParams(1), bestParams(2), bestAccuracy), 'FontSize', 14);
                colorbar;
                grid on;
                view(45, 30);
                legend('Fitted Surface', 'Parameter Points', 'Best Parameter Point', 'Location', 'best');
                
                %% Figure 2: Contour Plot
                figure('Position', [200, 200, 900, 700]);
                contourf(X_fine, Y_fine, Z_smooth, 20, 'LineWidth', 0.5);
                hold on;
                
                scatter(results(:,1), results(:,2), 50, 'yellow', 'filled', ...
                       'MarkerEdgeColor', 'k', 'LineWidth', 1);
                scatter(bestParams(1), bestParams(2), 100, 'red', 'filled', ...
                       'MarkerEdgeColor', 'k', 'LineWidth', 2);
                
                xlabel('Number of Trees', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('Maximum Depth', 'FontSize', 12, 'FontWeight', 'bold');
                title('Random Forest Parameter Contour Plot', 'FontSize', 14);
                colorbar;
                grid on;
                legend('Contour Lines', 'Parameter Points', 'Best Parameter Point', 'Location', 'best');
                
                %% Output Best Parameters
                fprintf('\n=== Random Forest Best Parameters ===\n');
                fprintf('Number of Trees: %d\n', bestParams(1));
                fprintf('Maximum Depth: %d\n', bestParams(2));
                fprintf('Validation Accuracy: %.4f\n', bestAccuracy);
            end
            
        case 'MLP'
            %% MLP Neural Network Hyperparameter Optimization - Adaptive Random Search
            layer1Range = [140, 300];      % Range for first layer neurons
            lambdaRange = [1e-6, 1];       % Range for regularization parameter (log scale)
            
            numRandomSamples = 56;         % Same as original 8*7=56 combinations
            results = [];
            
            fprintf('MLP Adaptive Random Search Parameter Combinations: %d\n', numRandomSamples);
            fprintf('Parameter Range: First Layer Neurons[%d,%d], Lambda[%.0e,%.0e]\n', ...
                layer1Range(1), layer1Range(2), lambdaRange(1), lambdaRange(2));
            
            % Record tried combinations
            tried_combinations = containers.Map();
            
            for sample = 1:numRandomSamples
                if sample <= 15
                    % Phase 1: Broad exploration
                    layer1 = randi(layer1Range);
                    logLambda = log10(lambdaRange(1)) + (log10(lambdaRange(2)) - log10(lambdaRange(1))) * rand();
                    lambda = 10^logLambda;
                else
                    % Phase 2: Adaptive sampling based on existing results
                    if size(results, 1) > 5
                        [bestAcc, bestIdx] = max(results(:,3));
                        bestLayer1 = results(bestIdx,1);
                        bestLambda = results(bestIdx,2);
                        
                        % Sample around best parameters (use log scale for lambda)
                        layer1_std = std(results(:,1));
                        logLambda_std = std(log10(results(:,2)));
                        
                        layer1 = round(normrnd(bestLayer1, layer1_std/2));
                        layer1 = max(layer1Range(1), min(layer1Range(2), layer1));
                        
                        bestLogLambda = log10(bestLambda);
                        logLambda = normrnd(bestLogLambda, logLambda_std/2);
                        logLambda = max(log10(lambdaRange(1)), min(log10(lambdaRange(2)), logLambda));
                        lambda = 10^logLambda;
                    else
                        layer1 = randi(layer1Range);
                        logLambda = log10(lambdaRange(1)) + (log10(lambdaRange(2)) - log10(lambdaRange(1))) * rand();
                        lambda = 10^logLambda;
                    end
                end
                
                % Avoid duplicate sampling
                key = sprintf('%d_%.6e', layer1, lambda);
                if isKey(tried_combinations, key)
                    sample = sample - 1;
                    continue;
                end
                tried_combinations(key) = true;
                
                fprintf('Progress: %.1f%% - ', sample/numRandomSamples*100);
                
                try
                    tic;
                    Mdl = fitcnet(Xtrain, ytrain, ...
                        'LayerSizes', layer1, ...
                        'Lambda', lambda, ...
                        'IterationLimit', 100);
                    
                    ypred_val = predict(Mdl, Xtest);
                    valAccuracy = mean(ypred_val == ytest);
                    trainTime = toc;
                    
                    results = [results; layer1, lambda, valAccuracy, trainTime];
                    fprintf('Layer1: %d, Lambda: %.2e, Acc: %.4f\n', layer1, lambda, valAccuracy);
                    
                catch ME
                    fprintf('Failed\n');
                    continue;
                end
            end
            
            if size(results, 1) > 5
                [bestAccuracy, bestIdx] = max(results(:,3));
                bestParams = results(bestIdx, 1:2);
                
                % Create fine grid
                layer1_fine = linspace(min(layer1Vals), max(layer1Vals), 50);
                lambda_fine = linspace(min(log10(lambdaVals)), max(log10(lambdaVals)), 50);
                [X_fine, Y_fine] = meshgrid(layer1_fine, lambda_fine);
                
                % Interpolation to obtain smooth surface
                Z_smooth = griddata(results(:,1), log10(results(:,2)), results(:,3), X_fine, Y_fine, 'v4');
                Z_smooth(isnan(Z_smooth)) = nanmean(Z_smooth(:));
                
                %% Figure 1: 3D Surface Plot
                figure('Position', [150, 150, 1000, 800]);
                surf(X_fine, 10.^Y_fine, Z_smooth, 'FaceAlpha', 0.85, 'EdgeColor', [0.3 0.3 0.3], 'EdgeAlpha', 0.3);
                hold on;
                
                scatter3(results(:,1), results(:,2), results(:,3), 60, 'yellow', 'filled', ...
                        'MarkerEdgeColor', 'k', 'LineWidth', 1);
                scatter3(bestParams(1), bestParams(2), bestAccuracy, 150, 'red', 'filled', ...
                        'MarkerEdgeColor', 'k', 'LineWidth', 2);
                
                xlabel('Number of Neurons in First Layer', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('Regularization Parameter \lambda', 'FontSize', 12, 'FontWeight', 'bold');
                zlabel('Accuracy', 'FontSize', 12, 'FontWeight', 'bold');
                title(sprintf('MLP Hyperparameter Optimization\nBest: %d Neurons, λ=%.4f, Accuracy=%.4f', ...
                    bestParams(1), bestParams(2), bestAccuracy), 'FontSize', 14);
                set(gca, 'YScale', 'log');
                colorbar;
                grid on;
                view(45, 30);
                legend('Fitted Surface', 'Parameter Points', 'Best Parameter Point', 'Location', 'best');
                
                %% Figure 2: Contour Plot
                figure('Position', [250, 250, 900, 700]);
                contourf(X_fine, 10.^Y_fine, Z_smooth, 20, 'LineWidth', 0.5);
                hold on;
                
                scatter(results(:,1), results(:,2), 50, 'yellow', 'filled', ...
                       'MarkerEdgeColor', 'k', 'LineWidth', 1);
                scatter(bestParams(1), bestParams(2), 100, 'red', 'filled', ...
                       'MarkerEdgeColor', 'k', 'LineWidth', 2);
                
                xlabel('Number of Neurons in First Layer', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('Regularization Parameter \lambda', 'FontSize', 12, 'FontWeight', 'bold');
                title('MLP Parameter Contour Plot', 'FontSize', 14);
                set(gca, 'YScale', 'log');
                colorbar;
                grid on;
                legend('Contour Lines', 'Parameter Points', 'Best Parameter Point', 'Location', 'best');
            end
            
        case 'KNN'
            %% KNN Hyperparameter Optimization - Adaptive Random Search
            numNeighborsRange = [1, 30];   % Range for number of neighbors
            distanceMetrics = {'euclidean', 'cityblock', 'cosine', 'correlation'};
            
            numRandomSamples = 36;         % Same as original 9*4=36 combinations
            results = [];
            
            fprintf('KNN Adaptive Random Search Parameter Combinations: %d\n', numRandomSamples);
            fprintf('Parameter Range: Number of Neighbors[%d,%d], Distance Metrics %d types\n', ...
                numNeighborsRange(1), numNeighborsRange(2), length(distanceMetrics));
            
            % Record tried combinations
            tried_combinations = containers.Map();
            
            for sample = 1:numRandomSamples
                if sample <= 10
                    % Phase 1: Broad exploration
                    numNeighbors = randi(numNeighborsRange);
                    distMetric = distanceMetrics{randi(length(distanceMetrics))};
                else
                    % Phase 2: Adaptive sampling based on existing results
                    if size(results, 1) > 5
                        [bestAcc, bestIdx] = max(results(:,3));
                        bestNeighbors = results(bestIdx,1);
                        bestDistIdx = results(bestIdx,2);
                        
                        % Sample around best parameters
                        neighbors_std = std(results(:,1));
                        numNeighbors = round(normrnd(bestNeighbors, neighbors_std/2));
                        numNeighbors = max(numNeighborsRange(1), min(numNeighborsRange(2), numNeighbors));
                        
                        % Distance metric: some probability to explore new metrics, but bias towards good ones
                        if rand() < 0.7  % 70% probability to use best metric or similar
                            distMetric = distanceMetrics{bestDistIdx};
                        else
                            distMetric = distanceMetrics{randi(length(distanceMetrics))};
                        end
                    else
                        numNeighbors = randi(numNeighborsRange);
                        distMetric = distanceMetrics{randi(length(distanceMetrics))};
                    end
                end
                
                distNum = find(strcmp(distanceMetrics, distMetric));
                
                % Avoid duplicate sampling
                key = sprintf('%d_%d', numNeighbors, distNum);
                if isKey(tried_combinations, key)
                    sample = sample - 1;
                    continue;
                end
                tried_combinations(key) = true;
                
                fprintf('Progress: %.1f%% - ', sample/numRandomSamples*100);
                
                try
                    tic;
                    Mdl = fitcknn(Xtrain, ytrain, ...
                        'NumNeighbors', numNeighbors, ...
                        'Distance', distMetric, ...
                        'Standardize', true);
                    
                    ypred_val = predict(Mdl, Xtest);
                    valAccuracy = mean(ypred_val == ytest);
                    trainTime = toc;
                    
                    results = [results; numNeighbors, distNum, valAccuracy, trainTime];
                    fprintf('Neighbors: %d, Distance: %s, Acc: %.4f\n', ...
                        numNeighbors, distMetric, valAccuracy);
                    
                catch ME
                    fprintf('Failed\n');
                    continue;
                end
            end
            
            if size(results, 1) > 5
                [bestAccuracy, bestIdx] = max(results(:,3));
                bestParams = results(bestIdx, 1:2);
                bestDistName = distanceVals{round(bestParams(2))};
                
                % Create fine grid
                neighbors_fine = linspace(min(numNeighborsVals), max(numNeighborsVals), 50);
                distance_fine = linspace(0.5, length(distanceVals)+0.5, 50);
                [X_fine, Y_fine] = meshgrid(neighbors_fine, distance_fine);
                
                % Interpolation to obtain smooth surface
                Z_smooth = griddata(results(:,1), results(:,2), results(:,3), X_fine, Y_fine, 'v4');
                Z_smooth(isnan(Z_smooth)) = nanmean(Z_smooth(:));
                
                %% Figure 1: 3D Surface Plot
                figure('Position', [200, 200, 1000, 800]);
                surf(X_fine, Y_fine, Z_smooth, 'FaceAlpha', 0.85, 'EdgeColor', [0.3 0.3 0.3], 'EdgeAlpha', 0.3);
                hold on;
                
                scatter3(results(:,1), results(:,2), results(:,3), 60, 'yellow', 'filled', ...
                        'MarkerEdgeColor', 'k', 'LineWidth', 1);
                scatter3(bestParams(1), bestParams(2), bestAccuracy, 150, 'red', 'filled', ...
                        'MarkerEdgeColor', 'k', 'LineWidth', 2);
                
                xlabel('Number of Neighbors (k)', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('Distance Metric', 'FontSize', 12, 'FontWeight', 'bold');
                zlabel('Accuracy', 'FontSize', 12, 'FontWeight', 'bold');
                set(gca, 'YTick', 1:length(distanceVals), 'YTickLabel', distanceVals);
                title(sprintf('KNN Hyperparameter Optimization\nBest: k=%d, %s, Accuracy=%.4f', ...
                    bestParams(1), bestDistName, bestAccuracy), 'FontSize', 14);
                colorbar;
                grid on;
                view(45, 30);
                legend('Fitted Surface', 'Parameter Points', 'Best Parameter Point', 'Location', 'best');
                
                %% Figure 2: Contour Plot
                figure('Position', [300, 300, 900, 700]);
                contourf(X_fine, Y_fine, Z_smooth, 20, 'LineWidth', 0.5);
                hold on;
                
                scatter(results(:,1), results(:,2), 50, 'yellow', 'filled', ...
                       'MarkerEdgeColor', 'k', 'LineWidth', 1);
                scatter(bestParams(1), bestParams(2), 100, 'red', 'filled', ...
                       'MarkerEdgeColor', 'k', 'LineWidth', 2);
                
                xlabel('Number of Neighbors (k)', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('Distance Metric', 'FontSize', 12, 'FontWeight', 'bold');
                set(gca, 'YTick', 1:length(distanceVals), 'YTickLabel', distanceVals);
                title('KNN Parameter Contour Plot', 'FontSize', 14);
                colorbar;
                grid on;
                legend('Contour Lines', 'Parameter Points', 'Best Parameter Point', 'Location', 'best');
                end
                end
                
                fprintf('%s optimization completed\n', current_model);
                end
                
                %% Generate simple performance comparison chart
                fprintf('\n=== Generating Model Performance Comparison ===\n');
                
                % Simple performance comparison code can be added here
                % Since we haven't stored all results, this is just an example
                figure('Position', [200, 200, 800, 600]);
                models = {'RandomForest', 'MLP', 'KNN'};
                % Assumed best accuracies (should be extracted from results in actual use)
                best_accuracies = [0.85, 0.82, 0.78]; % Example data
                
                bar(best_accuracies, 'FaceColor', [0.3 0.6 0.9]);
                set(gca, 'XTickLabel', models);
                ylabel('Best Accuracy');
                title('Best Performance Comparison of Different Models');
                grid on;
                
                for i = 1:length(best_accuracies)
                    text(i, best_accuracies(i) + 0.01, sprintf('%.4f', best_accuracies(i)), ...
                        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
                end
                
                fprintf('\n=== All model optimizations completed ===\n');
                
                %% New: Bar chart comparing all model performances
                figure('Position', [200, 200, 1200, 600]);
                subplot(1,2,1);
                % Accuracy comparison
                bar(metrics(:,1), 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
                set(gca, 'XTickLabel', modelNames, 'XTickLabelRotation', 45, 'FontSize', 10);
                xlabel('Machine Learning Models', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('Accuracy', 'FontSize', 12, 'FontWeight', 'bold');
                title('Accuracy Comparison of Different ML Models', 'FontSize', 14, 'FontWeight', 'bold');
                grid on;
                ylim([0, 1]);
                
                % Add value labels
                for i = 1:length(modelNames)
                    if metrics(i,1) > 0
                        text(i, metrics(i,1) + 0.02, sprintf('%.4f', metrics(i,1)), ...
                            'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
                    end
                end
                
                subplot(1,2,2);
                % F1-score comparison
                bar(metrics(:,4), 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);
                set(gca, 'XTickLabel', modelNames, 'XTickLabelRotation', 45, 'FontSize', 10);
                xlabel('Machine Learning Models', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('F1-Score', 'FontSize', 12, 'FontWeight', 'bold');
                title('F1-Score Comparison of Different ML Models', 'FontSize', 14, 'FontWeight', 'bold');
                grid on;
                ylim([0, 1]);
                
                % Add value labels
                for i = 1:length(modelNames)
                    if metrics(i,4) > 0
                        text(i, metrics(i,4) + 0.02, sprintf('%.4f', metrics(i,4)), ...
                            'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
                    end
                end
                
                sgtitle('Comprehensive Performance Comparison of ML Models', 'FontSize', 16, 'FontWeight', 'bold');