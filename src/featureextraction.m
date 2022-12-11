%img = imread("../assets/ck+/ck/CK+48/anger/S010_004_00000017.png");
% img = imread("../assets/ck+/ck/CK+48/anger/S042_004_00000020.png");
% img = preprocess_image(img);
% imshow(img)
% 
% [hogfeatures, visualization] = extractHOGFeatures(img, "CellSize", [2, 2]);
% subplot(1,2,1);
% imshow(img);
% subplot(1,2,2);
% plot(visualization);
% 
% 
% lbpfeatures = extractLBPFeatures(img);
% figure
% subplot(2,2,1);
% imshow(img);
% subplot(2,2,2);
% plot(vis2);


emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"];

X = [];
y = strings(1);

imcount = 1;

for i = 1:length(emotions)
    emotion = emotions(i);
    files = dir("../assets/ck+/ck/CK+48/" + emotion + "/*.png");
    for image = files'
        img = imread(image.folder + "\" + image.name);
        X(imcount, :) = preprocess_image(img);
        y(imcount) = emotion;

        imcount = imcount + 1;
    end
end

save("dataset", "X", "y");


% This function returns an image that is slightly blured and equalized
% using histogram equalization
function features = preprocess_image(img)
    preprocessed = imgaussfilt(img, 1);
    preprocessed = histeq(preprocessed);

    features = extractHOGFeatures(preprocessed, "CellSize", [4,4]);
end