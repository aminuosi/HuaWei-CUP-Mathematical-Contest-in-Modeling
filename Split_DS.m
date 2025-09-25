% Set folder path and segmentation parameters
folder_path = ''; % Replace with your folder path
window_size = 4096;  % Window size
overlap_rate = 0.75;  % Overlap rate
split_folder = fullfile(folder_path, ''); % Path for new segmented folder

% Create split folder if it doesn't exist
if ~exist(split_folder, 'dir')
    mkdir(split_folder);
end

% Get all .mat files in the folder
files = dir(fullfile(folder_path, '*.mat'));

% Process each file
for i = 1:length(files)
    filename = files(i).name;
    file_path = fullfile(folder_path, filename);

    % Load .mat file
    data = load(file_path);

    % Ensure the file contains FE_time variable
    if isfield(data, 'FE_time')
        signal = data.FE_time(:);  % Get vibration signal data and flatten to 1D

        % Calculate sliding window step size
        step_size = floor(window_size * (1 - overlap_rate));  % Step size = window size * (1 - overlap rate)

        % Segment the signal
        num_segments = floor((length(signal) - window_size) / step_size) + 1;
        for j = 0:num_segments-1
            start_idx = j * step_size + 1;
            end_idx = start_idx + window_size - 1;
            segment = signal(start_idx:end_idx);

            % Create new filename and save segmented data
            [~, base_name, ~] = fileparts(filename);  % Get filename without extension
            new_filename = sprintf('%s_%d.mat', base_name, j);
            new_file_path = fullfile(split_folder, new_filename);

            % Save segmented data (FE_time: segmented data, RPM: keep original)
            new_data = struct();  % Create new structure to save data
            new_data.FE_time = segment;  % Store segmented FE_time data

            % If file contains RPM variable, keep it unchanged
            if isfield(data, 'RPM')
                new_data.RPM = data.RPM;  % Keep RPM unchanged
            end

            % Save new .mat file
            save(new_file_path, '-struct', 'new_data');
            disp(['Saved: ', new_filename]);
        end
    else
        disp(['Skipping file (no FE_time): ', filename]);
    end
end