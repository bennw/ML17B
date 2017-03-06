addpath('NN')
addpath('misc')
addpath('DBN')


type = 2; % 1 is autoencoder (AE), 2 is classifier

% load MNIST data
load('data4students.mat')

% validation inputs
val_x = double(datasetInputs{3});
% validation targets
val_y = double(datasetTargets{3});

%if no validation exists then
% val_x = [];
% val_y = [];

% training inputs
train_x = double(datasetInputs{1});
% training targets
train_y = double(datasetTargets{1});

% test inputs
test_x  = double(datasetInputs{2});
% test targets
test_y  = double(datasetTargets{2});

inputSize = size(train_x,2);

% image normalisation: simply subtract mean
meanImage = sum(train_x,1)/size(train_x,1);
train_x = train_x - repmat(meanImage,size(train_x,1),1);
test_x = test_x - repmat(meanImage,size(test_x,1),1);
val_x = val_x - repmat(meanImage,size(val_x,1),1);

if type == 1 % AE
   outputSize  = inputSize; % in case of AE it should be equal to the number of inputs

   %if type = 1, i.e., AE then the last layer should be linear and usually a
   % series of decreasing layers are used
    hiddenActivationFunctions = {'sigm','sigm','sigm','linear'}; 
    hiddenLayers = [1000 500 250 50 250 500 1000 outputSize]; 
    
elseif type == 2 % classifier
    outputSize = size(train_y,2); % in case of classification it should be equal to the number of classes

    hiddenActivationFunctions = {'ReLu','ReLu','softmax'};
    m = 7;
    N = 21250;
    firstLayer = floor(2*sqrt(N/(m+2))+sqrt(N*(m+2)))+1;
    secondLayer = floor(m*sqrt(N/(m+2)))+1;
    hiddenLayers = [900 454 outputSize]; 
    
end


% parameters used for visualisation of first layer weights
visParams.noExamplesPerSubplot = 60; % number of images to show per row
visParams.noSubplots = floor(hiddenLayers(1) / visParams.noExamplesPerSubplot);
visParams.col = 30;% number of image columns
visParams.row = 30;% number of image rows 

inputActivationFunction = 'linear'; %sigm for binary inputs, linear for continuous input

% normalise data
% we assume that data are images so each image is z-normalised. If other
% types of data are used then each feature should be z-normalised on the
% training set and then mean and standard deviation should be applied to
% validation and test sets.
train_x = normaliseData(inputActivationFunction, train_x, []);
val_x = normaliseData(inputActivationFunction, val_x, []);
test_x = normaliseData(inputActivationFunction, test_x, []);

%initialise NN params
nn = paramsNNinit(hiddenLayers, hiddenActivationFunctions);

% Set some NN params
%-----
nn.epochs = 55;

% set initial learning rate
nn.trParams.lrParams.initialLR = 0.01; 
% set the threshold after which the learning rate will decrease (if type
% = 1 or 2)
nn.trParams.lrParams.lrEpochThres = 10;
% set the learning rate update policy (check manual)
% 1 = initialLR*lrEpochThres / max(lrEpochThres, T), 2 = scaling, 3 = lr / (1 + currentEpoch/lrEpochThres)
nn.trParams.lrParams.schedulingType = 1;

nn.trParams.momParams.schedulingType = 1;
%set the epoch where the learning will begin to increase
nn.trParams.momParams.momentumEpochLowerThres = 5;
%set the epoch where the learning will reach its final value (usually 0.9)
nn.trParams.momParams.momentumEpochUp40perThres = 20;

% set weight constraints
nn.weightConstraints.weightPenaltyL1 = 0;
nn.weightConstraints.weightPenaltyL2 = 0.1;
nn.weightConstraints.maxNormConstraint = 4;

% show diagnostics to monnitor training  
nn.diagnostics = 1;
% show diagnostics every "showDiagnostics" epochs
nn.showDiagnostics = 100;

% show training and validation loss plot
nn.showPlot = 1;

% if 1 then early stopping is used
nn.earlyStopping = 1;
nn.max_fail = 20;

nn.type = type;

% set the type of weight initialisation (check manual for details)
nn.weightInitParams.type = 9;

% set training method
% 1: SGD, 2: SGD with momentum, 3: SGD with nesterov momentum, 4: Adagrad, 5: Adadelta,
% 6: RMSprop, 7: Adam
nn.trainingMethod = 2;
%-----------

% initialise weights
[W, biases] = initWeights(inputSize, nn.weightInitParams, hiddenLayers, hiddenActivationFunctions);

nn.W = W;
nn.biases = biases;


% use bernoulli dropout
nn.dropoutParams.dropoutType = 1;
% if dropout is used then use max-norm constraint and a
%high learning rate + momentum with scheduling
% see the function below for suggested values
nn = useSomeDefaultNNparams(nn);

if type == 1 % AE
    [nn, Lbatch, L_train, L_val]  = trainNN(nn, train_x, train_x, val_x, val_x);
elseif type == 2 % classifier
    [nn, Lbatch, L_train, L_val, clsfError_train, clsfError_val]  = trainNN(nn, train_x, train_y, val_x, val_y);
end

nn = prepareNet4Testing(nn);

% visualise weights of first layer
%figure()
%visualiseHiddenLayerWeights(nn.W{1},visParams.col,visParams.row,visParams.noSubplots);


if type == 1 % AE
    [stats, output, e, L] = evaluateNNperformance( nn, test_x, test_x);
elseif type == 2 % classifier
    [stats, output, e, L] = evaluateNNperformance( nn, test_x, test_y);
end



