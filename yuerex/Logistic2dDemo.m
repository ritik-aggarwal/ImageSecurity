
%% Image Encryption Demo - Encryption and Decryption
clear all
close all
clc

%% 1. Load plaintext images
% Image 1
I = imread('cameraman.tif');

%% 2. Encryption
[CI,K] = Logistic2D_ImageCipher(I,'encryption');

%% 3. Decryption
DI = Logistic2D_ImageCipher(CI,'decryption',K);

%% 4. Analaysis
% Histogram
figure,subplot(221),imshow(I,[]),subplot(222),imshow(CI,[])
subplot(223),imhist(I),subplot(224),imhist(CI)