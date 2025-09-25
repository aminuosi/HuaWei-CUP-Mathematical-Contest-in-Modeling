% Set target sampling frequency to 32kHz
target_fs = 32000;

% Data directory path
data_dir = '~~';  % Please replace with your data directory

% Output directory
output_dir = '~~';  % Please replace with your results save directory

% Get all .mat files
files = dir(fullfile(data_dir, '**', '*.mat'));

% Process each file
for i = 1:length(files)
    % Load original data
    file_path = fullfile(files(i).folder, files(i).name);
    data = load(file_path);
    
    % Extract key information from filename (such as size, fault type, load, etc.)
    [~, file_name, ext] = fileparts(files(i).name);
    info = extract_info_from_filename(file_name, files(i).folder); % Extract file information
    
    % Select corresponding sampling signal (DE, FE, BA, etc.)
    signal = data.(info.signal_type); % Select corresponding signal based on parsed information
    
    % Get original sampling frequency (parsed from filename, assuming filename contains RPM information)
    original_fs = extract_fs_from_filename(file_name); % Extract original sampling frequency
    
    % Resample the signal
    resampled_signal = resample(signal, target_fs, original_fs);
    
    % Create new .mat filename
    new_file_name = generate_new_filename(info, target_fs); % Generate new filename
    
    % Save resampled data to new .mat file
    save(fullfile(output_dir, new_file_name), 'resampled_signal', '-v7.3');
end

% Extract key information from filename, such as size, fault type, position, load, sampling point, etc.
function info = extract_info_from_filename(file_name, folder)
    % Parse filename to get key information
    info = struct();
    
    % Parse filename to get fault type, size, position, etc.
    parts = strsplit(file_name, '_');
    
    % Determine data source and extract corresponding information
    if contains(file_name, 'B')  % Ball fault
        info.failure_type = 'B';
    elseif contains(file_name, 'IR')  % Inner race fault
        info.failure_type = 'IR';
    elseif contains(file_name, 'OR')  % Outer race fault
        info.failure_type = 'OR';
    else  % Normal data
        info.failure_type = 'N';
    end
    
    % Extract size information
    if contains(file_name, '0.007')
        info.size = 0.007;
    elseif contains(file_name, '0.014')
        info.size = 0.014;
    elseif contains(file_name, '0.021')
        info.size = 0.021;
    elseif contains(file_name, '0.028')
        info.size = 0.028;
    else
        info.size = NaN; % Normal data has no size information
    end
    
    % Extract position information (DE, FE, BA, etc.)
    if contains(file_name, 'DE')
        info.signal_type = 'DE'; % Drive end signal
    elseif contains(file_name, 'FE')
        info.signal_type = 'FE'; % Fan end signal
    elseif contains(file_name, 'BA')
        info.signal_type = 'BA'; % Base signal
    elseif contains(file_name, 'Centered')
        info.signal_type = 'Centered'; % 6 o'clock position
    elseif contains(file_name, 'Opposite')
        info.signal_type = 'Opposite'; % 12 o'clock position
    elseif contains(file_name, 'Orthogonal')
        info.signal_type = 'Orthogonal'; % 3 o'clock position
    else
        info.signal_type = 'Unknown'; % Unable to determine signal type
    end
    
    % Extract load information (0, 1, 2, 3 horsepower)
    if contains(file_name, '0')
        info.load = 0;
    elseif contains(file_name, '1')
        info.load = 1;
    elseif contains(file_name, '2')
        info.load = 2;
    elseif contains(file_name, '3')
        info.load = 3;
    else
        info.load = NaN; % No load information provided
    end
end

% Extract original sampling frequency, assuming filename contains RPM information (e.g., '1797rpm')
function fs = extract_fs_from_filename(file_name)
    % Extract sampling frequency
    if contains(file_name, '12kHz')
        fs = 12000;
    elseif contains(file_name, '48kHz')
        fs = 48000;
    else
        % Extract sampling frequency from RPM in filename
        rpm_match = regexp(file_name, '\d{4}rpm', 'match');
        if ~isempty(rpm_match)
            rpm_value = str2double(strrep(rpm_match{1}, 'rpm', ''));
            fs = rpm_value;
        else
            fs = 12000; % Default 12kHz
        end
    end
end

% Generate new filename containing size, fault type, position, and other information
function new_file_name = generate_new_filename(info, target_fs)
    % Initialize filename string
    new_file_name = '';
    
    % If it's normal data
    if strcmp(info.failure_type, 'N')
        % Normal data format: N_32kHz.mat
        new_file_name = sprintf('N_%dHz.mat', target_fs);
    else
        % If it's fault data
        % Concatenate fault type, size, position, and load information
        new_file_name = sprintf('%s_%.3f_inch_%s_%dHz', ...
            info.failure_type, info.size, info.signal_type, target_fs);
        
        % Add load information if available
        if ~isnan(info.load)
            new_file_name = sprintf('%s_L%d.mat', new_file_name, info.load);
        else
            new_file_name = [new_file_name, '.mat']; % No load information, directly add .mat suffix
        end
    end
end