function add_smos_product_to_points(varargin)
% Add point-matched AMSR/SMOS snow product values to the combined point table.
%
% Example:
%   addpath('random_forest');
%   add_smos_product_to_points( ...
%       'AMSRDir', 'G:\AU_SI12_1-20250103_145027', ...
%       'OutCsv', 'data\combined_points_with_smos_product.csv');
%
% The output CSV adds:
%   SMOS_Product_Snow_Depth_m
%   SMOS_Product_Match_Distance_Km
%   SMOS_Product_File
%
% Product variable:
%   /HDFEOS/GRIDS/NpPolarGrid12km/Data Fields/SI_12km_NH_SNOWDEPTH_5DAY
%
% The product units are cm. Raw values >= 110 are product flags
% (missing, land, open water, multiyear ice, variability, melt), so they are
% written as NaN.

p = inputParser;
addParameter(p, 'AMSRDir', 'G:\AU_SI12_1-20250103_145027', @(x)ischar(x) || isstring(x));
addParameter(p, 'OutCsv', fullfile('data', 'combined_points_with_smos_product.csv'), @(x)ischar(x) || isstring(x));
addParameter(p, 'IceBridgeMat', fullfile('data', 'Master_Dataset_AllFeatures.mat'), @(x)ischar(x) || isstring(x));
addParameter(p, 'ImbCsv', fullfile('data', 'Validation_Dataset_IMB_AllBuoys_Combined_AllData.csv'), @(x)ischar(x) || isstring(x));
addParameter(p, 'InputCsv', '', @(x)ischar(x) || isstring(x));
addParameter(p, 'ProductPath', '/HDFEOS/GRIDS/NpPolarGrid12km/Data Fields/SI_12km_NH_SNOWDEPTH_5DAY', @(x)ischar(x) || isstring(x));
addParameter(p, 'ProductScale', 0.01, @isnumeric);
addParameter(p, 'InvalidMinRaw', 110, @isnumeric);
parse(p, varargin{:});
args = p.Results;

amsrDir = char(args.AMSRDir);
outCsv = char(args.OutCsv);

if strlength(string(args.InputCsv)) > 0
    T = readtable(char(args.InputCsv));
    if ~ismember('source', T.Properties.VariableNames)
        T.source = repmat("csv", height(T), 1);
    end
else
    T = buildCombinedPointTable(char(args.IceBridgeMat), char(args.ImbCsv));
end

required = {'Year','Month','Day','Latitude','Longitude'};
missingRequired = setdiff(required, T.Properties.VariableNames);
if ~isempty(missingRequired)
    error('Point table is missing required columns: %s', strjoin(missingRequired, ', '));
end

he5Files = dir(fullfile(amsrDir, '**', '*.he5'));
if isempty(he5Files)
    error('No .he5 files found under: %s', amsrDir);
end

fileDates = strings(numel(he5Files), 1);
filePaths = strings(numel(he5Files), 1);
for i = 1:numel(he5Files)
    token = regexp(he5Files(i).name, '(20\d{6})', 'match', 'once');
    if ~isempty(token)
        fileDates(i) = string(token);
        filePaths(i) = string(fullfile(he5Files(i).folder, he5Files(i).name));
    end
end
validFiles = fileDates ~= "";
fileDates = fileDates(validFiles);
filePaths = filePaths(validFiles);
if isempty(fileDates)
    error('No dated .he5 files found under: %s', amsrDir);
end
[fileDates, uniqueIdx] = unique(fileDates, 'stable');
filePaths = filePaths(uniqueIdx);

sampleFile = char(filePaths(1));
fprintf('Reading grid from: %s\n', sampleFile);
gridLat = double(h5read(sampleFile, '/HDFEOS/GRIDS/NpPolarGrid12km/lat'));
gridLon = double(h5read(sampleFile, '/HDFEOS/GRIDS/NpPolarGrid12km/lon'));

gridXYZ = latlonToUnitXYZ(gridLat(:), gridLon(:));
fprintf('Building KD-tree with %d grid cells...\n', size(gridXYZ, 1));
tree = KDTreeSearcher(gridXYZ);

pointDate = string(compose('%04d%02d%02d', T.Year, T.Month, T.Day));
productValue = nan(height(T), 1);
matchDistanceKm = nan(height(T), 1);
matchedFile = strings(height(T), 1);

uniquePointDates = unique(pointDate);
for i = 1:numel(uniquePointDates)
    dateStr = uniquePointDates(i);
    fileIdx = find(fileDates == dateStr, 1, 'first');
    if isempty(fileIdx)
        continue;
    end

    rowMask = pointDate == dateStr & isfinite(T.Latitude) & isfinite(T.Longitude);
    if ~any(rowMask)
        continue;
    end

    productFile = char(filePaths(fileIdx));
    fprintf('Matching %s: %d points using %s\n', char(dateStr), sum(rowMask), productFile);
    raw = double(h5read(productFile, char(args.ProductPath)));
    product = raw * args.ProductScale;
    product(raw >= args.InvalidMinRaw | raw < 0) = nan;

    queryXYZ = latlonToUnitXYZ(T.Latitude(rowMask), T.Longitude(rowMask));
    [idx, chordDistance] = knnsearch(tree, queryXYZ);
    km = 6371 * 2 * asin(min(max(chordDistance / 2, 0), 1));

    rowIdx = find(rowMask);
    productValue(rowIdx) = product(idx);
    matchDistanceKm(rowIdx) = km;
    matchedFile(rowIdx) = string(getFileName(productFile));
end

T.SMOS_Product_Snow_Depth_m = productValue;
T.SMOS_Product_Match_Distance_Km = matchDistanceKm;
T.SMOS_Product_File = matchedFile;

outDir = fileparts(outCsv);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end
writetable(T, outCsv);

matched = isfinite(T.SMOS_Product_Snow_Depth_m);
fprintf('\nSaved: %s\n', outCsv);
fprintf('Rows: %d\n', height(T));
fprintf('Rows with matched valid product: %d\n', sum(matched));
if any(matched)
    fprintf('Product snow depth summary (m): min=%.4f, median=%.4f, mean=%.4f, max=%.4f\n', ...
        min(T.SMOS_Product_Snow_Depth_m(matched)), ...
        median(T.SMOS_Product_Snow_Depth_m(matched)), ...
        mean(T.SMOS_Product_Snow_Depth_m(matched)), ...
        max(T.SMOS_Product_Snow_Depth_m(matched)));
    if ismember('source', T.Properties.VariableNames)
        disp('Matched rows by source:');
        disp(groupcounts(T(matched, :), 'source'));
    end
end
end

function T = buildCombinedPointTable(iceBridgeMat, imbCsv)
S = load(iceBridgeMat, 'Final_Data');
D = double(S.Final_Data);
if size(D, 2) < 19 && size(D, 1) >= 19
    D = D.';
end
if size(D, 2) < 19
    error('Expected Final_Data to have at least 19 columns, got %dx%d.', size(D, 1), size(D, 2));
end

iceNames = {
    'Year','Month','Day','Latitude','Longitude', ...
    'TB_18V','TB_18H','TB_23V','TB_23H','TB_36V','TB_36H','TB_89V','TB_89H', ...
    'Ice_Thickness_m','Mean_Freeboard_m','Snow_Depth_m', ...
    'Surface_Roughness','KT19_Surface_Temp','MY_Ice_Fraction'};
Ice = array2table(D(:, 1:19), 'VariableNames', iceNames);
Ice.source = repmat("icebridge", height(Ice), 1);
Ice.Hour = nan(height(Ice), 1);
Ice.Distance_Km = nan(height(Ice), 1);

Imb = readtable(imbCsv);
Imb.source = repmat("imb", height(Imb), 1);
if ~ismember('Mean_Freeboard_m', Imb.Properties.VariableNames)
    Imb.Mean_Freeboard_m = nan(height(Imb), 1);
end
if ~ismember('Surface_Roughness', Imb.Properties.VariableNames)
    Imb.Surface_Roughness = nan(height(Imb), 1);
end
if ~ismember('KT19_Surface_Temp', Imb.Properties.VariableNames)
    Imb.KT19_Surface_Temp = nan(height(Imb), 1);
end
if ~ismember('MY_Ice_Fraction', Imb.Properties.VariableNames)
    Imb.MY_Ice_Fraction = nan(height(Imb), 1);
end
if ~ismember('Distance_Km', Imb.Properties.VariableNames)
    Imb.Distance_Km = nan(height(Imb), 1);
end
if ~ismember('Hour', Imb.Properties.VariableNames)
    Imb.Hour = nan(height(Imb), 1);
end

common = {
    'source','Year','Month','Day','Hour','Latitude','Longitude','Distance_Km', ...
    'TB_18V','TB_18H','TB_23V','TB_23H','TB_36V','TB_36H','TB_89V','TB_89H', ...
    'Snow_Depth_m','Ice_Thickness_m','Mean_Freeboard_m','Surface_Roughness', ...
    'KT19_Surface_Temp','MY_Ice_Fraction'};
T = [Ice(:, common); Imb(:, common)];
T = T(isfinite(T.Snow_Depth_m) & T.Snow_Depth_m > 0, :);
end

function xyz = latlonToUnitXYZ(lat, lon)
lon = mod(double(lon), 360);
latRad = deg2rad(double(lat));
lonRad = deg2rad(lon);
xyz = [cos(latRad).*cos(lonRad), cos(latRad).*sin(lonRad), sin(latRad)];
end

function name = getFileName(pathText)
[~, base, ext] = fileparts(pathText);
name = [base ext];
end
