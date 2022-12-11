
SetNum=7178;

digitDatasetPath = 'C:\Users\ymerz\OneDrive\Documents\McGill\Fall 2022\COMP 558\Project\Data set\FER2\train';
% fullfile(matlabroot,'toolbox','nnet','nndemos', 'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

figure;
perm = randperm(SetNum,20);
%{
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
%}
labelCount = countEachLabel(imds)

img = readimage(imds,1);

numTrainFiles = 300;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
digitDatasetPath = 'C:\Users\ymerz\OneDrive\Documents\McGill\Fall 2022\COMP 558\Project\Data set\FER2\test';
imdsValidation= imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

layers = [
    imageInputLayer([48 48 1]);
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
%{
digitDatasetPath = 'C:\Users\ymerz\OneDrive\Documents\McGill\Fall 2022\COMP 558\Project\Data set\FER2\test';
testSet= imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% Pass CNN image features to trained classifier
predictedLabels = predict(net, testSet);

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
%}