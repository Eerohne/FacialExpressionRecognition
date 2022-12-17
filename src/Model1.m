
%a few comments and lines in this code come from an official matlab
%tutorial for setting up a CNN
%(https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html)
%parameters and layers have been changed in order to fit our problem, and
%an augmentation step for the training data has been added, as well as the
%computation and display of the confusion matrix and the saving of our
%resulting model

%training set
digitDatasetPath = 'Data set\FER2\train';
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%testing set
digitDatasetPath = 'Data set\FER2\test';
imdsTest= imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%showing off details of data set
labelCount = countEachLabel(imds)



%Selecting a constant number of picture per emotion for training
numTrainFiles = 400;
[imdsTrain0,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%augmentation process which increases the size of the input by 5
images1=zeros(48,48,1,numTrainFiles*7);
images2=zeros(48,48,1,numTrainFiles*7);
images3=zeros(48,48,1,numTrainFiles*7);
images4=zeros(48,48,1,numTrainFiles*7);
images5=zeros(48,48,1,numTrainFiles*7);
for j=1:numTrainFiles*7
    img=readimage(imdsTrain0,j);
    images1(:,:,1,j)= img;
    augI = imageDataAugmenter( RandXTranslation=[-5 5],RandYTranslation=[-5 5],RandXReflection=1,RandYReflection=1);
    images2(:,:,1,j)= augment(augI,img);
    augI = imageDataAugmenter( RandXTranslation=[-5 5],RandYTranslation=[-5 5],RandXReflection=1,RandYReflection=1);
    images3(:,:,1,j)= augment(augI,img);
    augI = imageDataAugmenter( RandXTranslation=[-5 5],RandYTranslation=[-5 5],RandXReflection=1,RandYReflection=1');
    images4(:,:,1,j)= augment(augI,img);
    augI = imageDataAugmenter( RandXTranslation=[-5 5],RandYTranslation=[-5 5],RandXReflection=1,RandYReflection=1);
    images5(:,:,1,j)= augment(augI,img);
end
labels=imdsTrain0.Labels;
trainSet= cat(4,images1,images2,images3,images4,images5);
labels=cat(1,labels,labels,labels,labels,labels);




%layers construction
layers = [
    imageInputLayer([48 48 1])
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
   
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    
    maxPooling2dLayer(2,'Stride',2)
    %dropoutLayer
    convolution2dLayer(3,64,'Padding','same')
    
    batchNormalizationLayer
    reluLayer

    flattenLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer];


%parameters
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',10, ...
    'Verbose',false, ...
    'Plots','training-progress');
%trainingFeatures=reshape(trainingFeatures,size(trainingFeatures,1),size(trainingFeatures,2),1,size(trainingFeatures,3));
net = trainNetwork(trainSet,labels,layers,options);

%saving the model
gregnet1 = net;
save gregnet1


%testing the model
YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)
% Tabulate the results using a confusion matrix.
confMat = confusionmat(imdsTest.Labels, YPred)
helperDisplayConfusionMatrix(confMat)






%this function was based from a helper function used in a matlab example (https://www.mathworks.com/help/vision/ug/digit-classification-using-hog-features.html)
function helperDisplayConfusionMatrix(confMat)
    % Display the confusion matrix in a formatted table.
    
    % Convert confusion matrix into percentage form
    confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
    
    digits = ["anger", "disgust", "fear", "happy", "neutral", "sadness", "surprise"];
    colHeadings = arrayfun(@(x)sprintf(digits(x),x),1:7,'UniformOutput',false);
    format = repmat('%-9s',1,11);
    header = sprintf(format,'emotion        |',colHeadings{:});
    fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
    for idx = 1:numel(digits)
        fprintf('%-9s',   [digits(idx) '      |']);
        fprintf('%-9.2f', confMat(idx,:));
        fprintf('\n')
    end
end