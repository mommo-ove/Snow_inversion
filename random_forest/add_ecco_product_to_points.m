function add_ecco_product_to_points(varargin)
% Add point-matched ECCO L4 sea-ice/snow variables to a point table.
%
% Example:
%   addpath('random_forest');
%   add_ecco_product_to_points( ...
%       'InputCsv', 'data\combined_points_with_smos_product.csv', ...
%       'ECCODir', 'G:\ECCO_L4_SEA_ICE_CONC_THICKNESS_05DEG_DAILY_V4R4_V4r4-20250104_024658', ...
%       'OutCsv', 'data\combined_points_with_smos_ecco_products.csv');
%
% ECCO variables:
%   SIhsnow: area-averaged snow thickness over the whole grid cell, m
%   SIarea: sea-ice concentration, 0-1
%   SIheff : area-averaged sea-ice thickness over the whole grid cell, m
%
% For point observations located on sea ice, the ice-covered-fraction snow
% thickness can be estimated as SIhsnow / SIarea. Both are written.

p = inputParser;
addParameter(p, 'InputCsv', fullfile('data', 'combined_points_with_smos_product.csv'), @(x)ischar(x) || isstring(x));
addParameter(p, 'OutCsv', fullfile('data', 'combined_points_with_smos_ecco_products.csv'), @(x)ischar(x) || isstring(x));
addParameter(p, 'ECCODir', 'G:\ECCO_L4_SEA_ICE_CONC_THICKNESS_05DEG_DAILY_V4R4_V4r4-20250104_024658', @(x)ischar(x) || isstring(x));
addParameter(p, 'MinIceArea', 0.15, @isnumeric);
parse(p, varargin{:});
args = p.Results;

inputCsv = char(args.InputCsv);
outCsv = char(args.OutCsv);
eccoDir = char(args.ECCODir);

if ~exist(inputCsv, 'file')
    error('Input CSV not found: %s', inputCsv);
end

opts = detectImportOptions(inputCsv, 'FileType', 'delimitedtext', 'Delimiter', ',');
opts.VariableNamingRule = 'preserve';
T = readtable(inputCsv, opts);
required = {'Year','Month','Day','Latitude','Longitude'};
missingRequired = setdiff(required, T.Properties.VariableNames);
if ~isempty(missingRequired)
    error('Point table is missing required columns: %s', strjoin(missingRequired, ', '));
end

ncFiles = dir(fullfile(eccoDir, '**', '*.nc'));
if isempty(ncFiles)
    error('No ECCO .nc files found under: %s', eccoDir);
end

fileDates = strings(numel(ncFiles), 1);
filePaths = strings(numel(ncFiles), 1);
for i = 1:numel(ncFiles)
    token = regexp(ncFiles(i).name, '(20\d{2}-\d{2}-\d{2})', 'match', 'once');
    if ~isempty(token)
        fileDates(i) = string(strrep(token, '-', ''));
        filePaths(i) = string(fullfile(ncFiles(i).folder, ncFiles(i).name));
    end
end
validFiles = fileDates ~= "";
fileDates = fileDates(validFiles);
filePaths = filePaths(validFiles);
if isempty(fileDates)
    error('No dated ECCO .nc files found under: %s', eccoDir);
end
[fileDates, uniqueIdx] = unique(fileDates, 'stable');
filePaths = filePaths(uniqueIdx);

sampleFile = char(filePaths(1));
latVec = double(ncread(sampleFile, 'latitude'));
lonVec = double(ncread(sampleFile, 'longitude'));
lonVec = normalizeLon180(lonVec);

pointDate = string(compose('%04d%02d%02d', T.Year, T.Month, T.Day));
n = height(T);
areaAvgSnow = nan(n, 1);
iceCoveredSnow = nan(n, 1);
iceConc = nan(n, 1);
iceThick = nan(n, 1);
matchDistanceKm = nan(n, 1);
matchedFile = strings(n, 1);

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

    eccoFile = char(filePaths(fileIdx));
    rows = find(rowMask);
    fprintf('Matching ECCO %s: %d points using %s\n', char(dateStr), numel(rows), eccoFile);

    hsnow = squeeze(double(ncread(eccoFile, 'SIhsnow')));
    area = squeeze(double(ncread(eccoFile, 'SIarea')));
    heff = squeeze(double(ncread(eccoFile, 'SIheff')));
    hsnow = cleanEccoVar(hsnow);
    area = cleanEccoVar(area);
    heff = cleanEccoVar(heff);
    hsnow(hsnow < 0) = 0;
    heff(heff < 0) = 0;
    area(area < 0) = 0;
    area(area > 1) = 1;

    [latIdx, lonIdx, km] = nearestEccoGrid(T.Latitude(rows), T.Longitude(rows), latVec, lonVec);
    idx = sub2ind(size(hsnow), lonIdx, latIdx);

    hsnowPoint = hsnow(idx);
    areaPoint = area(idx);
    heffPoint = heff(idx);
    iceSnowPoint = hsnowPoint ./ areaPoint;
    iceSnowPoint(~isfinite(iceSnowPoint) | areaPoint < args.MinIceArea) = nan;

    areaAvgSnow(rows) = hsnowPoint;
    iceCoveredSnow(rows) = iceSnowPoint;
    iceConc(rows) = areaPoint;
    iceThick(rows) = heffPoint;
    matchDistanceKm(rows) = km;
    matchedFile(rows) = string(getFileName(eccoFile));
end

T.ECCO_AreaAvg_Snow_Depth_m = areaAvgSnow;
T.ECCO_IceCovered_Snow_Depth_m = iceCoveredSnow;
T.ECCO_Sea_Ice_Concentration = iceConc;
T.ECCO_Sea_Ice_Thickness_m = iceThick;
T.ECCO_Product_Match_Distance_Km = matchDistanceKm;
T.ECCO_Product_File = matchedFile;

outDir = fileparts(outCsv);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end
writetable(T, outCsv);

fprintf('\nSaved: %s\n', outCsv);
fprintf('Rows: %d\n', height(T));
printSummary(T, 'ECCO_AreaAvg_Snow_Depth_m');
printSummary(T, 'ECCO_IceCovered_Snow_Depth_m');
if ismember('source', T.Properties.VariableNames)
    matched = isfinite(T.ECCO_IceCovered_Snow_Depth_m);
    disp('ECCO ice-covered snow matched rows by source:');
    disp(groupcounts(T(matched, :), 'source'));
end
end

function out = cleanEccoVar(x)
out = double(x);
out(abs(out) > 1e20) = nan;
end

function lon = normalizeLon180(lon)
lon = mod(double(lon) + 180, 360) - 180;
end

function [latIdx, lonIdx, km] = nearestEccoGrid(lat, lon, latVec, lonVec)
lat = double(lat);
lon = normalizeLon180(lon);
latIdx = zeros(numel(lat), 1);
lonIdx = zeros(numel(lon), 1);
for i = 1:numel(lat)
    [~, latIdx(i)] = min(abs(latVec - lat(i)));
    dlon = abs(normalizeLon180(lonVec - lon(i)));
    [~, lonIdx(i)] = min(dlon);
end
gridLat = latVec(latIdx);
gridLon = lonVec(lonIdx);
km = haversineKm(lat, lon, gridLat, gridLon);
end

function d = haversineKm(lat1, lon1, lat2, lon2)
radiusKm = 6371;
phi1 = deg2rad(lat1);
phi2 = deg2rad(lat2);
dphi = deg2rad(lat2 - lat1);
dlambda = deg2rad(normalizeLon180(lon2 - lon1));
a = sin(dphi / 2).^2 + cos(phi1) .* cos(phi2) .* sin(dlambda / 2).^2;
d = radiusKm * 2 * asin(min(1, sqrt(a)));
end

function name = getFileName(pathText)
[~, base, ext] = fileparts(pathText);
name = [base ext];
end

function printSummary(T, col)
values = T.(col);
matched = isfinite(values);
fprintf('%s matched: %d\n', col, sum(matched));
if any(matched)
    fprintf('%s summary (m): min=%.4f, median=%.4f, mean=%.4f, max=%.4f\n', ...
        col, min(values(matched)), median(values(matched)), mean(values(matched)), max(values(matched)));
end
end
