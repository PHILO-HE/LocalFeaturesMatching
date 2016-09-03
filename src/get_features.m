% Local Feature Stencil Code
% CS 143 Computater Vision, Brown U.
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)



% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

% Placeholder that you can delete. Empty features.

function [features] = get_features(image, x, y, feature_width)

features = zeros(size(x,1), 128);
%%features = zeros(size(x,1), 160);
small_gaussian = fspecial('Gaussian', [feature_width feature_width], 1);  % 用于得到图像的两个梯度
large_gaussian = fspecial('Gaussian', [feature_width feature_width], feature_width*3/2);  % 用于对特征点周围点的幅值加权

[gmatx, gmaty] = imgradientxy(small_gaussian);
ix = imfilter(image, gmatx);  % 图像x方向的梯度值
iy = imfilter(image, gmaty);  % 图像y方向的梯度值

%%  求出每个点的幅值  %%
mag=zeros(size(ix,1),size(ix,2));
for i=1:size(ix,1)
    for j=1:size(ix,2)
        mag(i,j)=sqrt(ix(i,j)*ix(i,j)+iy(i,j)*iy(i,j));
    end
end

%% 确定象限  将坐标系分为8个象限，每pi/4一个象限；-pi与-3pi/4之间为第一象限，逆时针依此至第8象限  %%
  get_xiangxian = @(x,y) (ceil(atan2(y,x)/(pi/4)) + 4);
  %%get_xiangxian = @(x,y) (ceil(atan2(y,x)/(pi/5)) + 5); 
xiangxian = arrayfun(get_xiangxian, ix, iy);   % 得到图像每个点由ix&iy确定的象限
%%  每个特征点得到包含有4x4个cell的window，以特征点为中心，每个cell为4*4的矩阵  %%

window_size = feature_width;
for ii = 1: length(x)                                                                   
    window_x_range = (x(ii) - 0.5*window_size): (x(ii) + 0.5*window_size-1);   % 确定每个特征点的window的范围
    window_y_range = (y(ii) - 0.5*window_size): (y(ii) + 0.5*window_size-1);
    
    window_mag = mag(window_y_range, window_x_range);   
    window_mag = window_mag.*large_gaussian;   % 在window内的点的幅值要用一个大的高斯函数加权
    
    window_xiangxian = xiangxian(window_y_range, window_x_range);

    %%% 对每个4*4的cell，得到一个象限分布的直方图 %%%
    for xx = 1:4
        for yy = 1:4
            cell_xiangxian = window_xiangxian((xx-1)*4+1:(xx-1)*4+4, (yy-1)*4+1:(yy-1)*4+4);
            cell_mag = window_mag((xx-1)*4+1:(xx-1)*4+4, (yy-1)*4+1:(yy-1)*4+4);
            %%for orient=1:10
            for orient = 1:8
                om = (cell_xiangxian == orient);
                % 每增加xx,多增加一行，多4*8个值，每增加yy,多增加一个cell，多8个值
                features(ii, ((xx-1)*4*8 + (yy-1)*8) + orient) =sum(sum(om.*cell_mag));
                %features(ii, ((xx-1)*4*10 + (yy-1)*10) + orient) =sum(sum(om.*cell_mag));
            end
        end
    end
end


%%  normalize  %%
len1=size(features,1);
len2=size(features,2);
for i=1:len1
    for k=1:2  % 做两次normalize
        sum_features=0;
        for j=1:len2
          sum_features=sum_features+features(i,j);    % 将每个特征点的特征描述值求和
        end
        for j=1:len2
          features(i,j)=features(i,j)/sum_features;   % 将每个特征点的特征描述值normalize
          % 第一次normalize，将normalize后的幅值限定在0.2以下，依照对课件的理解：
          % threshold gradient magnitudes to avoid excessive influence of high gradients
          if(k==1)  
             if(features(i,j)>0.2)
                features(i,j)=0.2;
             end
          end
        end
    end
end

