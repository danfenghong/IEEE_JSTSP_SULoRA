clc;
clear;
close all;

addpath('synthetic data'); %load data
addpath('functions'); %load functions used for SULoRA

load synthetic_data.mat; %load the synthetic data
load endmembers.mat; %load endmembers extracted from HSI using VCA

%M : real endmembers (l by p)
%A_true : real abundance maps (m*n by p)
%noise : add the Gaussian noise (m*n by l)
%X : spectral signatures (m*n by l)

%% Endmembers extracted by VCA algorithm, which can be seen as an initialization for unmixing step

if ~exist('endmembers','var')
    
    num_endmembers = 5; %the number of endmembers; you have to give it in advance or estimate it using e.g.,HySime
    endmembers = VCA(Y, 'Endmembers',num_endmembers,'SNR',30);
    s = 1-pdist2(endmembers',M','cosine');
    [mi,ind] = sort(s,'descend');
    index = ind(1,:);
else    
    
    s = 1-pdist2(endmembers',M','cosine');
    [mi,ind] = sort(s,'descend');
    index = ind(1,:);
end

A = endmembers;

%% Generating the HSI
Y = X + noise;
Y = Y';

%% Initializing the abundances using SPCLSU
X_PCLSU = sunsal(A,Y,'lambda',0,'ADDONE','no','POSITIVITY','yes', ...
                'AL_iters',200,'TOL', 1e-3, 'verbose','yes');
X_SPCLSU = X_PCLSU./repmat(sum(X_PCLSU),size(X_PCLSU,1),1);

%% parameter setting in SULoRA
param.alfa = 0.1;
param.beta = 0.01;
param.gama = 8e-3;
maxiter = 1000;

%% Run the SULoRA to unmix the HSI
[X, theta] = SULoRA(Y,A,X_SPCLSU,param,maxiter);

%% Compute ARMSE for quantitative evaluation
ARMSE_SULoRA = ARMSE(X(index, :), A_true');

