function convolvedFeatures = cnnConvolve(images, W, b, con_matrix)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  images - large images to convolve with, matrix in the form
%           images(row, col, channel, image number)
%  W, b - W, b 
%         W is of shape (filterDim,filterDim,channel,numFilters)
%         b is of shape (numFilters,1)
%  con_matrix -
%         the connection between input channel and output maps. If the ith
%         input channel has connection with jth output map, then
%         con_matrix(i,j) = 1, otherwise, 0;
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

[filterDimRow,filterDimCol,channel,numFilters] = size(W);

if nargin < 4
    con_matrix = ones(channel, numFilters);
end

[imageDimRow, imageDimCol,~, numImages] = size(images);
convDimRow = imageDimRow - filterDimRow + 1;
convDimCol = imageDimCol - filterDimCol + 1;

convolvedFeatures = zeros(convDimRow, convDimCol, numFilters, numImages);

%   Convolve every filter with every image here to produce the convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum)



for imageNum = 1:numImages
  for filterNum = 1:numFilters
      convolvedImage = zeros(convDimRow, convDimCol);
      for channelNum = 1:channel
          if con_matrix(channelNum,filterNum) ~= 0
            % Obtain the feature (filterDim x filterDim) needed during the convolution
            filter = W(:,:,channelNum,filterNum); 

            % Flip the feature matrix because of the definition of convolution, as explained later
            filter = rot90(squeeze(filter),2);

            % Obtain the image
            im = squeeze(images(:, :, channelNum,imageNum));

            % Convolve "filter" with "im", adding the result to convolvedImage
            % be sure to do a 'valid' convolution
            convolvedImage = convolvedImage + conv2(im, filter, 'valid');
          end
            % Add the bias unit
            convolvedImage = convolvedImage + b(filterNum);
            convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
      end
  end
end

end

