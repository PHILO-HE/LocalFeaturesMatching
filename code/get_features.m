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
small_gaussian = fspecial('Gaussian', [feature_width feature_width], 1);  % ���ڵõ�ͼ��������ݶ�
large_gaussian = fspecial('Gaussian', [feature_width feature_width], feature_width*3/2);  % ���ڶ���������Χ��ķ�ֵ��Ȩ

[gmatx, gmaty] = imgradientxy(small_gaussian);
ix = imfilter(image, gmatx);  % ͼ��x������ݶ�ֵ
iy = imfilter(image, gmaty);  % ͼ��y������ݶ�ֵ

%%  ���ÿ����ķ�ֵ  %%
mag=zeros(size(ix,1),size(ix,2));
for i=1:size(ix,1)
    for j=1:size(ix,2)
        mag(i,j)=sqrt(ix(i,j)*ix(i,j)+iy(i,j)*iy(i,j));
    end
end

%% ȷ������  ������ϵ��Ϊ8�����ޣ�ÿpi/4һ�����ޣ�-pi��-3pi/4֮��Ϊ��һ���ޣ���ʱ����������8����  %%
  get_xiangxian = @(x,y) (ceil(atan2(y,x)/(pi/4)) + 4);
  %%get_xiangxian = @(x,y) (ceil(atan2(y,x)/(pi/5)) + 5); 
xiangxian = arrayfun(get_xiangxian, ix, iy);   % �õ�ͼ��ÿ������ix&iyȷ��������
%%  ÿ��������õ�������4x4��cell��window����������Ϊ���ģ�ÿ��cellΪ4*4�ľ���  %%

window_size = feature_width;
for ii = 1: length(x)                                                                   
    window_x_range = (x(ii) - 0.5*window_size): (x(ii) + 0.5*window_size-1);   % ȷ��ÿ���������window�ķ�Χ
    window_y_range = (y(ii) - 0.5*window_size): (y(ii) + 0.5*window_size-1);
    
    window_mag = mag(window_y_range, window_x_range);   
    window_mag = window_mag.*large_gaussian;   % ��window�ڵĵ�ķ�ֵҪ��һ����ĸ�˹������Ȩ
    
    window_xiangxian = xiangxian(window_y_range, window_x_range);

    %%% ��ÿ��4*4��cell���õ�һ�����޷ֲ���ֱ��ͼ %%%
    for xx = 1:4
        for yy = 1:4
            cell_xiangxian = window_xiangxian((xx-1)*4+1:(xx-1)*4+4, (yy-1)*4+1:(yy-1)*4+4);
            cell_mag = window_mag((xx-1)*4+1:(xx-1)*4+4, (yy-1)*4+1:(yy-1)*4+4);
            %%for orient=1:10
            for orient = 1:8
                om = (cell_xiangxian == orient);
                % ÿ����xx,������һ�У���4*8��ֵ��ÿ����yy,������һ��cell����8��ֵ
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
    for k=1:2  % ������normalize
        sum_features=0;
        for j=1:len2
          sum_features=sum_features+features(i,j);    % ��ÿ�����������������ֵ���
        end
        for j=1:len2
          features(i,j)=features(i,j)/sum_features;   % ��ÿ�����������������ֵnormalize
          % ��һ��normalize����normalize��ķ�ֵ�޶���0.2���£����նԿμ�����⣺
          % threshold gradient magnitudes to avoid excessive influence of high gradients
          if(k==1)  
             if(features(i,j)>0.2)
                features(i,j)=0.2;
             end
          end
        end
    end
end

