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

