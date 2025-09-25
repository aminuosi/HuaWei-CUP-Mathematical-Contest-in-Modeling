clc; clear; close all;

% ============ Configuration ============ 
dataPath = ''; % Modify path 
output_csv = '';

files = dir(fullfile(dataPath, '*.mat'));

% ============ Construct feature names (English) ============ 
feature_names = {};       % English names (for internal counting)

% --- Time domain 12 features --- 
time_domain_names = {'mean','std','var','rms','peak','p2p', ...
    'skewness','kurtosis','crest_factor','impulse_factor', ...
    'shape_factor','clearance_factor'}; 
for t = 1:numel(time_domain_names)
    feature_names{end+1} = ['time_' time_domain_names{t}];
end

% --- Frequency domain 11 features --- 
freq_domain_names = {'spectral_centroid','spectral_variance','spectral_skewness','spectral_kurtosis', ...
    'bpfo_peak','bpfi_peak','bsf_peak','ftf_peak', ...
    'low_energy_ratio','mid_energy_ratio','high_energy_ratio'}; 
for t = 1:numel(freq_domain_names)
    feature_names{end+1} = ['freq_' freq_domain_names{t}];
end

% --- Frequency band energy 22 features --- 
for b = 0:21
    feature_names{end+1} = sprintf('freq_band_energy_%d', b);
end

% --- Wavelet packet energy 16 + entropy 1 = 17 features --- 
for w = 0:15
    feature_names{end+1} = sprintf('wavelet_band_energy_%d', w);
end
feature_names{end+1} = 'wavelet_entropy';

% --- Envelope statistics 4 features --- 
env_names = {'env_mean','env_std','env_skewness','env_kurtosis'}; 
for t = 1:numel(env_names)
    feature_names{end+1} = env_names{t};
end

% --- Time-frequency reserved 3 features --- 
for r = 0:2
    feature_names{end+1} = sprintf('time_freq_reserved_%d', r);
end

% --- Envelope spectrum 4 features --- 
env_spec_names = {'env_spec_mean','env_spec_std','env_spec_max','env_spec_peak_freq'}; 
for t = 1:numel(env_spec_names)
    feature_names{end+1} = env_spec_names{t};
end

% --- Envelope spectrum frequency points 48 features --- 
for e = 0:47
    feature_names{end+1} = sprintf('env_spec_freq_%d', e);
end

% Check feature count (should be 121) 
expected_feat_count = numel(feature_names); % Should be 121 
fprintf('Expected features per row (excluding first 4 columns FileName/VarName/Position/RPM) = %d\n', expected_feat_count);

% Header (FileName, VarName, Position, RPM) + English feature names 
header = [{'FileName','VarName','Position','RPM'}, feature_names];

% ============ Main feature extraction loop ============ 
all_features = {}; % Each row is a cell row, column count should be 4 + expected_feat_count

for fi = 1:length(files)
    file_path = fullfile(files(fi).folder, files(fi).name);
    S = load(file_path);
    vars = fieldnames(S);

    % Basic information 
    [~, baseName, ~] = fileparts(files(fi).name);

    % Try to find RPM field (case insensitive) 
    rpm_idx = find(~cellfun(@isempty, regexpi(vars,'RPM')), 1);
    if isempty(rpm_idx)
        rpmVal = NaN;
        warning('File %s: RPM field not found, setting RPM to NaN', files(fi).name);
    else
        rv = S.(vars{rpm_idx});
        if isnumeric(rv) && numel(rv) >= 1
            rpmVal = double(rv(1)); % Take first value as RPM scalar
        else
            rpmVal = NaN;
        end
    end

    % Iterate through variables, select variables containing DE/FE/BA (exclude RPM) 
    for vk = 1:numel(vars)
        vname = vars{vk};
        if vk == rpm_idx
            continue; % Skip RPM field
        end
        % Only process signal variables containing DE or FE or BA
        if isempty(regexpi(vname,'DE')) && isempty(regexpi(vname,'FE')) && isempty(regexpi(vname,'BA'))
            continue;
        end

        x = S.(vname);
        % Only continue with numeric vectors
        if ~isnumeric(x) || ~isvector(x) || numel(x) < 10
            warning('Skipping %s in %s (not numeric vector or too short)', vname, files(fi).name);
            continue;
        end
        x = double(x(:)); % Column vector

        feats = zeros(1, expected_feat_count); % Pre-allocate

        % ---------- Time domain 12 features ---------- 
        idx = 1;
        feats(idx) = mean(x); idx = idx + 1;
        feats(idx) = std(x);  idx = idx + 1;
        feats(idx) = var(x);  idx = idx + 1;
        feats(idx) = rms(x);  idx = idx + 1;
        feats(idx) = max(abs(x)); idx = idx + 1;
        feats(idx) = peak2peak(x); idx = idx + 1;
        feats(idx) = skewness(x); idx = idx + 1;
        feats(idx) = kurtosis(x); idx = idx + 1;
        feats(idx) = max(abs(x))/rms(x); idx = idx + 1;
        feats(idx) = max(abs(x))/mean(abs(x)); idx = idx + 1;
        feats(idx) = rms(x)/mean(abs(x)); idx = idx + 1;
        feats(idx) = max(abs(x)) / (mean(sqrt(abs(x)))^2); idx = idx + 1;

        % ---------- Frequency domain 11 features ---------- 
        N = length(x);
        Y = fft(x);
        P2 = abs(Y)/N;
        halfN = floor(N/2)+1;
        P1 = P2(1:halfN);            % Single-sided spectrum
        freqs = (0:(halfN-1))'/N;    % Column vector (normalized frequency)
        X_power = (P1.^2);

        % Spectrum related features
        denom = sum(X_power) + eps;
        feats(idx) = dot(freqs, X_power)/denom; idx = idx + 1; % spectral_centroid
        feats(idx) = var(X_power); idx = idx + 1;              % spectral_variance
        feats(idx) = skewness(X_power); idx = idx + 1;         % spectral_skewness
        feats(idx) = kurtosis(X_power); idx = idx + 1;         % spectral_kurtosis

        % Four "fault frequency peaks" - default to frequency of max spectral peak
        [~,peak_idx] = max(X_power);
        peak_freq = freqs(peak_idx);
        feats(idx) = peak_freq; idx = idx + 1; % bpfo_peak
        feats(idx) = peak_freq; idx = idx + 1; % bpfi_peak
        feats(idx) = peak_freq; idx = idx + 1; % bsf_peak
        feats(idx) = peak_freq; idx = idx + 1; % ftf_peak

        % Low/mid/high energy ratio (equal spectrum segments)
        L = numel(X_power);
        low_end = floor(L/3);
        mid_end = floor(2*L/3);
        feats(idx) = sum(X_power(1:low_end))/denom; idx = idx + 1;
        feats(idx) = sum(X_power(low_end+1:mid_end))/denom; idx = idx + 1;
        feats(idx) = sum(X_power(mid_end+1:end))/denom; idx = idx + 1;

        % ---------- Frequency band energy 22 features ---------- 
        band_edges = round(linspace(1, L+1, 23)); % 22 bands
        for b = 1:22
            a = band_edges(b);
            bidx = band_edges(b+1)-1;
            if a<=bidx
                feats(idx) = sum(X_power(a:bidx));
            else
                feats(idx) = 0;
            end
            idx = idx + 1;
        end

        % ---------- Wavelet packet 16 band energies + entropy ---------- 
        node_energies = zeros(1,16);
        try
            wpt = wpdec(x, 4, 'db4'); % 4 levels -> 16 leaf nodes
            for n = 0:15
                c = wpcoef(wpt, n);
                node_energies(n+1) = sum(double(c(:)).^2);
            end
            totalE = sum(node_energies) + eps;
            wavelet_entropy = -sum((node_energies/totalE) .* log((node_energies/totalE) + eps));
        catch ME
            % If Wavelet Toolbox unavailable or error occurs, set to 0 and warn
            warning('Wavelet packet calculation failed (%s), setting corresponding features to 0. Error: %s', vname, ME.message);
            node_energies = zeros(1,16);
            wavelet_entropy = 0;
        end
        feats(idx:idx+15) = node_energies; idx = idx + 16;
        feats(idx) = wavelet_entropy; idx = idx + 1;

        % ---------- Envelope statistics 4 features ---------- 
        env = abs(hilbert(x));
        feats(idx) = mean(env); idx = idx + 1;
        feats(idx) = std(env);  idx = idx + 1;
        feats(idx) = skewness(env); idx = idx + 1;
        feats(idx) = kurtosis(env); idx = idx + 1;

        % ---------- Time-frequency reserved 3 features ---------- 
        feats(idx:idx+2) = [0,0,0]; idx = idx + 3;

        % ---------- Envelope spectrum 4 features ---------- 
        EnvSpecFull = abs(fft(env));
        halfE = floor(length(env)/2)+1;
        EnvSpec = EnvSpecFull(1:halfE);
        EnvSpec = double(EnvSpec(:));
        denomE = sum(EnvSpec) + eps;
        feats(idx) = mean(EnvSpec); idx = idx + 1;
        feats(idx) = std(EnvSpec);  idx = idx + 1;
        feats(idx) = max(EnvSpec);  idx = idx + 1;
        [~, idxm] = max(EnvSpec);
        % Envelope spectrum peak frequency (normalized)
        feats(idx) = (idxm-1)/length(env); idx = idx + 1;

        % ---------- Envelope spectrum 48 frequency points (pad with 0 if insufficient) ---------- 
        for e = 1:48
            if e <= numel(EnvSpec)
                feats(idx) = EnvSpec(e);
            else
                feats(idx) = 0;
            end
            idx = idx + 1;
        end

        % Final check if feats length matches expected
        if numel(feats) ~= expected_feat_count
            warning('File %s variable %s feature length %d does not match expected %d, auto-correcting (padding with 0 or truncating).', ...
                files(fi).name, vname, numel(feats), expected_feat_count);
            if numel(feats) < expected_feat_count
                feats = [feats, zeros(1, expected_feat_count - numel(feats))];
            else
                feats = feats(1:expected_feat_count);
            end
        end

        % Position label (DE/FE/BA) 
        if ~isempty(regexpi(vname,'DE')) 
            pos = 'DE'; 
        elseif ~isempty(regexpi(vname,'FE')) 
            pos = 'FE'; 
        elseif ~isempty(regexpi(vname,'BA')) 
            pos = 'BA'; 
        else 
            pos = 'NAN'; 
        end

        % Assemble one row (FileName, VarName, Position, RPM, feats...) 
        row = [{baseName, vname, pos, rpmVal}, num2cell(feats)]; 
        % Append row to all_features (will throw exception if column count differs) 
        all_features = [all_features; row]; 
    end
end

% ============ Final check and write table ============ 
if isempty(all_features)
    error('No features extracted (all_features is empty). Please check data files and variable naming.');
end

% Convert all_features to matrix size check
[nRows, nCols] = size(all_features);
nHeader = numel(header);
fprintf('Data rows=%d, Data columns=%d, Header columns=%d\n', nRows, nCols, nHeader);

% If column count doesn't match, auto-correct header or data (shouldn't normally happen)
if nCols ~= nHeader
    warning('Column mismatch: data columns=%d, header columns=%d, system will auto-correct header to match data columns.', nCols, nHeader);
    if nHeader < nCols
        % Extend header (pad with generic names)
        for add = (nHeader+1):nCols
            header{end+1} = sprintf('extra_col_%d', add-nHeader);
        end
    else
        % Header longer than data -> truncate header
        header = header(1:nCols);
    end
end

% Ensure all header elements are character vectors (cell array of char)
for hi = 1:numel(header)
    if ~ischar(header{hi})
        header{hi} = char(header{hi});
    end
end

% Convert to table and write CSV (specify UTF-8 encoding)
T = cell2table(all_features, 'VariableNames', header);
writetable(T, output_csv, 'Encoding', 'UTF-8');

fprintf('Complete: %d rows written to file: %s\n', height(T), output_csv);