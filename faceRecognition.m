%% Part a: computing the principal components of the first 190 individuals' neutral expression
clear all;
close all;
theCurDir = '/Users/alberttan/Documents/ECE 269/Project/training';
filesA = dir(strcat(theCurDir,'/*a.jpg'));
theNeutralImages = [];

for i  = 1:length(filesA)
    anImage = imread(strcat('Project/training/',filesA(i).name));
    anImage = reshape(anImage, [31266 1]);
    theNeutralImages = [theNeutralImages anImage];
end
theNeutralImages=double(theNeutralImages);

filesB = dir(strcat(theCurDir,'/*b.jpg'));
smilingImages = [];

for i  = 1:length(filesB)
    anImage = imread(strcat('Project/training/',filesB(i).name));
    anImage = reshape(anImage, [31266 1]);
    smilingImages = [smilingImages anImage];
end
smilingImages=double(smilingImages);



theCurDir = '/Users/alberttan/Documents/ECE 269/Project/testing';
filesC = dir(strcat(theCurDir,'/*a.jpg'));
theNeutralImagesTest = [];

for i  = 1:length(filesC)
    anImage = imread(strcat('Project/testing/',filesC(i).name));
    anImage = reshape(anImage, [31266 1]);
    theNeutralImagesTest = [theNeutralImagesTest anImage];
end
theNeutralImagesTest=double(theNeutralImagesTest);




M = mean(theNeutralImages,2);
A = double(theNeutralImages)-M;
[V,D]=eig(A'*A);
diagD=diag(D);
singularValues=svd(A'*A);
x=1:1:190;
y=singularValues;
plot(x,sort(y));
title('Singular Values');
xlabel('Component Number');
ylabel('Singular Value');

PCs=normc(A*V);

flag=imread('yinyanh.jpg');
greyFlag=rgb2gray(flag);
imshow(greyFlag)
figure
flagMatrix=uint8(imresize(greyFlag,[193,162]));
imshow(flagMatrix);
flagFinal=double(imresize(flagMatrix,[31266,1]))-M;

%% part b, one image

%compute lowest for arbitrary k


errorArray=zeros(1,190);
imageArray=zeros(31266,6);
phi= theNeutralImages(:,136)-M; %independent of k
indexSelection=[1,30,60,90,120,190];


matrix=reshape(phi,[193,162]);
count=1;

for i=1:190
    arr = 191-i:190;
    eigVecMatrix = PCs(:,arr);
    phi_hat_i = eigVecMatrix*(eigVecMatrix'*phi);
    
    if ismember(i,indexSelection)
        imageArray(:,count)=phi_hat_i;
        count=count+1;
    end
    
    
    error_i=immse(phi,phi_hat_i);
    errorArray(i)=error_i;
end

selectedProjections
%% plot error part a.2

errorx= 1:1:190;
figure;
plot(errorx,errorArray);
title('MSE vs PCs for Image 195');
xlabel('Principal Components');
ylabel('MSE');

%% part f rotation
figure
for i=0:7
    rotation=i*45
    phi= theNeutralImages(:,136); %stays the same no matter k
    J = imrotate(reshape(phi,[193,162]),rotation,'bilinear','crop');
    phi_rotated=reshape(J,[31266,1])-M;
    
    phi_hat_rotated_i = PCs*(PCs'*phi_rotated);
    subplot(2,8,i+1)
    imshow(uint8(J));
    title(rotation);
    
    subplot(2,8,i+9)
    displayIm=uint8(phi_hat_rotated_i+M);
    displayImMatrix = reshape(displayIm,[193,162]);
    imshow((displayImMatrix));
    title(rotation);
end
    

%% display
flagFinal=phi;
displayIm = PCs(:,190)*(PCs(:,190)'*flagFinal)+M;
displayImMatrix = reshape(displayIm,[193,162]);
imshow(uint8(displayImMatrix)); 

%% display multiple of image 100
figure
for i=1:6
    subplot(2,3,i)
    displayIm=uint8(imageArray(:,i)+M);
    displayImMatrix = reshape(displayIm,[193,162]);
    imshow((displayImMatrix));
    title(['PC', num2str(indexSelection(i))]);
end


