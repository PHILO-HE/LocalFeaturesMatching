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
threshold=0.67;   % ����ֵ����Ҫʵ�����  threshold�޶����������ν���ľ���֮�ȣ�С�ڸ�ֵ����ΪΪƥ���
%threshold=0.71;
%%  �����Ӧ������ͼ������������㼯�ϵ�ŷʽ���룬ÿһ�ж�Ӧ��features1��ĳһ����������features2������������ľ���  %%
len1=size(features1,1);
len2=size(features2,1);
euc_dist=zeros(size(features1,1),size(features2,1));  % euc_distΪ���������㼯�ϵģ��������������ģ�ŷʽ�������
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

%%  ���������ÿ�а�����������  %%
[sort_dist,index]=sort(euc_dist,2);  %%�������У�index��¼������sort_dist����ÿ������features�е�����

%%  confidece_temp��¼�ν����������ľ���֮�ȣ�������threshold���Ƶ�ֵ��confidences����
confidences_temp=sort_dist(:,2)./sort_dist(:,1);  
% confidence_tempΪ�ν����������֮�ȣ�Խ��Խ�ã����Ա�������1/threshold��ֵ
confidences=confidences_temp(confidences_temp>1/threshold);  
matches=zeros(size(confidences,1),2);

%%  matches���������ƥ�����������������confidence���ģ����ڵ�һ��  %%
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
[confidences,features1_index]=sort(confidences,'descend');  % ��confidences����������
matches=matches(features1_index,:);   
end
