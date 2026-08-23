clc; clear; close all;

% ==== Load data ====
P99 = load('data99okeh.mat');     % pastiin nama file bener
result = P99.result;              % cell array sinyal
if isfield(P99,'fs')
    fs = P99.fs;
else
    fs = 8000;                    
end

% ==== Indeks & label ====
idx    = [10 20 40 99 70];
labels = {'bronchial','asthma','crackle','frictionrub','stridor'};

% ==== PREPROCESS helper: mean removal + normalization [-1,1] ====
preproc = @(x) ( (x - mean(x)) ./ max(abs(x - mean(x)) + eps) );

% ==== Plot Waveform ====
figure('Name','Waveform 1D (preprocessed)');
tiledlayout(numel(idx),1);
for i = 1:numel(idx)
    x  = result{idx(i),1};
    x  = preproc(x);                              % <- mean removal + norm
    tt = (0:numel(x)-1)/fs;
    nexttile; plot(tt,x); grid on;
    xlabel('Time (s)'); ylabel('Amp');
    title(sprintf('Signal #%d — %s (pre-processed)', idx(i), labels{i}));
end