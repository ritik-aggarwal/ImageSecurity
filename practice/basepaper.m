%% Image Encryption and Decryption:
clear
close all
clc

%% 1. Load plaintext images
img = imread('Lenna.png'); % Read image
[M,N,r] = size(img);
red = img(:,:,1); % Red channel
green = img(:,:,2); % Green channel
blue = img(:,:,3); % Blue channel

R = reshape(img(:,:,1),[],M);
G = reshape(img(:,:,2),[],M);
B = reshape(img(:,:,3),[],M);
D = cat(3, R, G, B);
R = double(R);
G = double(G);
B = double(B);
%{
a0 = zeros(size(img, 1), size(img, 2));
just_red = cat(3, red, a0, a0);
just_green = cat(3, a0, green, a0);
just_blue = cat(3, a0, a0, blue);
back_to_original_img = cat(3, red, green, blue);
%}
%% Display images:

%figure, imshow(img), title('Original image')
%figure, imshow(red), title('Red channel')
%figure, imshow(green), title('Green channel')
%figure, imshow(blue), title('Blue channel')
%figure, imshow(back_to_original_img), title('Back to original image')


%% 2. Encryption
maxlength = M*N;
divisor = M*N*255;
u1 = 3.399; u2 = 3.4499; k1 = 0.21; u = 3.85; k2 = 0.15;
%Range:
%3.56<u<=4; 2.75<u1<= 3.45; 2.75<u2<=3.45; 0.15<k1<=0.21; 0.13<k2<=0.15;
x = ones(maxlength,1);
y = ones(maxlength,1);
z = ones(maxlength,1);

x(1,1)=0.345;
y(1,1)=0.365;
z(1,1)=0.537;

for i = 2:maxlength
  x(i,1) = u1*x(i-1,1)*(1-x(i-1,1))+ k1*(power(y(i-1,1),2));
  y(i,1) = u2*y(i-1,1)*(1-y(i-1,1))+ k2*((power(x(i-1,1),2))+x(i-1,1)*y(i-1,1));
  z(i,1) = u*z(i-1,1)*(1-z(i-1,1));
end
x = reshape(x,M,N);
y = reshape(y,M,N);
z = reshape(z,M,N);

