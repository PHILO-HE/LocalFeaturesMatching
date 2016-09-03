% Local Feature Stencil Code
% CS 143 Computater Vision, Brown U.
% Written by James Hays

% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the features as additional inputs.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features 1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.


% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).

% To start with, simply implement the "ratio test", equation 4.18 in
% section 4.1.3 of Szeliski. For extra credit you can implement various
% forms of spatial verification of matches.

% Placeholder that you can delete. Random matches and confidences

% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.

function [matches, confidences] = match_features(features1, features2)
threshold=0.67;   % 经验值，需要实验调整  threshold限定了最近点与次近点的距离之比，小于该值才认为为匹配点
%threshold=0.71;
%%  计算对应于两幅图像的两个特征点集合的欧式距离，每一行对应于features1中某一个特征点与features2中所有特征点的距离  %%
len1=size(features1,1);
len2=size(features2,1);
euc_dist=zeros(size(features1,1),size(features2,1));  % euc_dist为两个特征点集合的（特征点描述量的）欧式距离矩阵
dim=size(features1,2);
for i=1:len1
  for j=1:len2
      sum_value=0;
      for k=1:dim
        sum_value=sum_value+power(features1(i,k)-features2(j,k),2);
      end
    euc_dist(i,j)=sqrt(sum_value);
  end
end

%%  将距离矩阵每行按照升序排列  %%
[sort_dist,index]=sort(euc_dist,2);  %%升序排列，index记录排序后的sort_dist矩阵每个点在features中的索引

%%  confidece_temp记录次近点与最近点的距离之比，将符合threshold限制的值给confidences矩阵
confidences_temp=sort_dist(:,2)./sort_dist(:,1);  
% confidence_temp为次近点与最近点之比，越大越好，所以保留大于1/threshold的值
confidences=confidences_temp(confidences_temp>1/threshold);  
matches=zeros(size(confidences,1),2);

%%  matches矩阵来存放匹配的特征点索引，且confidence最大的，放在第一行  %%
k=1;
for i=1:len1
    if(sort_dist(i,1)/sort_dist(i,2)<threshold)
        matches(k,1)=i;
        matches(k,2)=index(i,1);
        k=k+1;
    end
end
% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
[confidences,features1_index]=sort(confidences,'descend');  % 将confidences按降序排列
matches=matches(features1_index,:);   
end
