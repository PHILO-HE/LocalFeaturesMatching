% Local Feature Stencil Code
% CS 143 Computater Vision, Brown U.
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or(b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.


% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% Placeholder that you can delete. 20 random points

%function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width)


function[x, y] = get_interest_points(image, feature_width)
%%  �����ĸ�������line79��thresholdΪ�����������Ҫ������ȷ�����ֵ  %%
gau_width=15;      % ��˹�����ά��
gau_small_std=1.2;   % ��С��˹�����sigmaֵ
gau_large_std=2.5;   % �ϴ�ĸ�˹�����sigmaֵ
alpha=0.06;        % harris��ʽ�ļ�Ȩ����


gmat=fspecial('gaussian',[gau_width,gau_width],gau_small_std);
[gmat_x,gmat_y]=imgradientxy(gmat);

Ix=imfilter(image,gmat_x);
Iy=imfilter(image,gmat_y);

%%  ͼ��߽��ݶ�ֵ��Ϊ0,ȥ���߽��ݶȱ仯�����������Ӱ��  %%
Ix((1:feature_width),:)=0;
Ix((size(Ix,1)-feature_width:size(Ix,1)),:)=0;
Ix(:,(1:feature_width))=0;
Ix(:,(size(Ix,2)-feature_width:size(Ix,2)))=0;

Iy((1:feature_width),:)=0;
Iy((size(Iy,1)-feature_width:size(Iy,1)),:)=0;
Iy(:,(1:feature_width))=0;
Iy(:,(size(Iy,2)-feature_width:size(Iy,2)))=0;

%%  �õ�harris����  %%
large_gmat=fspecial('gaussian',[gau_width,gau_width],gau_large_std);
Ixx=imfilter(Ix.*Ix,large_gmat);   % Ix & Iy �������˻����ϴ�ĸ�˹�����˲�
Iyy=imfilter(Iy.*Iy,large_gmat);
Ixy=imfilter(Ix.*Iy,large_gmat);

harris=Ixx.*Iyy-Ixy.*Ixy-alpha.*(Ixx+Iyy).*(Ixx+Iyy);  %%% det(M)-alpha*trace(M)

%% �趨threshold������harris����thresholdֵ���ϵĵ�  %%
%%%  threshold��Ҫ����  %%%%
threshold=harris>0.31*(mean2(harris)); 

harris=harris.*threshold;    %����threshold���ϵ�harris�ǵ�
max_harris=colfilt(harris,[feature_width feature_width],'sliding',@max);     % ��������Ѱ�Ҵ����ڵ�harris���ֵ
true_mat=harris==max_harris;
harris=harris.*true_mat;      % ֻ����max_harris

[y,x]=find(harris>0);  % ȷ���������λ�ã�yΪharris�������������xΪ����������Ӧ��ͼ��������
end


     









