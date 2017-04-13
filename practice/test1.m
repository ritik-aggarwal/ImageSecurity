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


%% Step-2:
e1 = 0;e2 = 0;e3 = 0;
for i = 1:M
    for j = 1:N
        e1 = e1 + (R(i,j)/divisor);
        e2 = e2 + (G(i,j)/divisor);
        e3 = e3 + (B(i,j)/divisor);
    end
end

a = floor(mod((e1 + e2 + e3)*power(10,12),256));
b = floor(mod((e1 + e2 + e3)*power(10,12)/2,256));
c = floor(mod((e1 + e2 + e3)*power(10,12)/3,256));

%% Step-3:
m1=ones(M,N,1);
m2=ones(M,N,1);
m3=ones(M,N,1);
for i = 1:M
   for j = 1:N
    m1(i,j,1) = mod(x(i,j,1)*power(10,15),256);
    m2(i,j,1) = mod(y(i,j,1)*power(10,15),256);
    m3(i,j,1) = mod(z(i,j,1)*power(10,15),256);
   end
end

S = ones(M,N,3);
for i = 1:M
    for j = 1:N
        if(a ~= m1(i,j))
           S(i,j,1) = abs(a-m1(i,j));
        else
            S(i,j,1) = a;
        end
    end
end
for i = 1:M
    for j = 1:N
        if(b ~= m2(i,j))
           S(i,j,2) = abs(b-m2(i,j));
        else
           S(i,j,2) = b;
        end
    end
end
for i = 1:M
    for j = 1:N
        if(c ~= m3(i,j))
           S(i,j,3) = abs(c-m3(i,j));
        else
           S(i,j,3) = m3(i,j);
        end
    end
end
for i = 1:M
    for j = 1:N
    if (((i-1)*N)+j) ~= M*N
       t1 = floor(mod(x(i,j,1)*power(10,15),maxlength));
       t2 = floor(mod(y(i,j,1)*power(10,15),maxlength));
       t3 = floor(mod((x(i,j,1)+y(i,j,1)-(floor(x(i,j,1)+y(i,j,1))))*power(10,15),maxlength));
    else
        t1 = floor(mod(z(i,j,1)* power(10,15),maxlength));
        t2 = floor(mod((1-z(i,j,1))* power(10,15),maxlength));
        t3 = floor((t1 + t2 +c)/2);
    end
    S(i,j,1) = circshift(S(i,j,1),t1);
    S(i,j,2) = circshift(S(i,j,2),t2);
    S(i,j,3) = circshift(S(i,j,3),t3);
    end
end

%% Step-4:
for i = 1:M
    for j = 1:N
    if (((i-1)*N)+j) ~= M*N
       t1 = floor(mod(x(i,j,1)*power(10,15),maxlength));
       t2 = floor(mod(y(i,j,1)*power(10,15),maxlength));
       t3 = floor(mod((x(i,j,1)+y(i,j,1)-(floor(x(i,j,1)+y(i,j,1))))*power(10,15),maxlength));
    else
        t1 = floor(mod(z(i,j,1)* power(10,15),maxlength));
        t2 = floor(mod((1-z(i,j,1))* power(10,15),maxlength));
        t3 = floor((t1 + t2 +c)/2);
    end
    S(i,j,1) = circshift(S(i,j,1),t1);
    S(i,j,2) = circshift(S(i,j,2),t2);
    S(i,j,3) = circshift(S(i,j,3),t3);
    end
end
H1 = hadamard(512);
H2 = hadamard(512);
H3 = hadamard(512);
D = double(D);
E(:,:,1) = abs(((log(D(:,:,1))/log(256)) + H1)*128);
E(:,:,2) = abs(((log(D(:,:,2))/log(256)) + H2)*128);
E(:,:,3) = abs(((log(D(:,:,3))/log(256)) + H3)*128);

%E(:,:,1) = mod(abs(S(:,:,1)*H1 + D(:,:,1)),256);
%E(:,:,2) = mod(abs(S(:,:,2)*H2 + D(:,:,2)),256);
%E(:,:,3) = mod(abs(S(:,:,3)*H3 + D(:,:,3)),256);
%E = mat2gray(E);

%% Step-5:
%E = reshape(E,maxlength,1,3);
%C = ones(maxlength,1,3);
%C = uint8(C);
%E = uint8(E);
%C = zeros(M,N,3);
C = E;
%{
for k = 1:3
    for i = 1:M
        for j = 1:N
            newi = mod((2*i + j),M)+1;
            newj = mod((i + j),N)+1;
            C(newi,newj,k) = E(i,j,k);
        end
    end
end
X = zeros(M,N,3);
for k = 1:3
    for i = 1:M
        for j = 1:N
            newi = mod((2*i + j),M)+1;
            newj = mod((i + j),N)+1;
            X(newi,newj,k) = C(i,j,k);
        end
    end
end
%}
%X = uint8(X);
for i = 1:65
   C = Arnold_map_encrypt(C);
   %figure(1);image(X);title(i);
   %refresh;
   %pause(0.0001);
end
%C = mat2gray(C);
%figure, imshow(C), title('Final Encrypted RGB image')

%C1 = C(:,:,1);
%C2 = C(:,:,2);
%C3 = C(:,:,3);
%X = zeros(M,N,3);
%{
for k = 1:3
    for i = 1:M
        for j = 1:N
            X(i,j,k) = E(i,j,k) + C(i,j,k);         
        end
    end
end
%}
%S = uint8(S);
S = uint8(S);
C = uint8(C);
%figure, imshow(C), title('Original input RGB image')
temp1 = C(255,255,1);
temp2 = C(255,255,2);
temp3 = C(255,255,3);
X = bitxor(S,C);
%C = reshape(C,M,N,3);
%D = uint8(D);
X = mat2gray(X);
%{
%figure, imshow(D), title('Original input RGB image')
%figure, imshow(X), title('Final Encrypted RGB image')
%figure, imshow(X(:,:,1)), title('Final Encrypted RGB image1')
%figure, imshow(X(:,:,2)), title('Final Encrypted RGB image2')
%figure, imshow(X(:,:,3)), title('Final Encrypted RGB image3')
%figure, imshow(C(:,:,1)), title('Red channel')
%figure, imshow(C(:,:,2)), title('Green channel')
%figure, imshow(C(:,:,3)), title('Blue channel')
%subplot(3,1,1), imhist(X(:,:,1)), title('Histogram of Red component(Encrypted image)')
%subplot(3,1,2), imhist(X(:,:,2)), title('Histogram of Green component(Encrypted image)')
%subplot(3,1,3), imhist(X(:,:,3)), title('Histogram of Blue component(Encrypted image)')
%}
%% 3. Decryption

maxlength = M*N;
divisor = M*N*255;
X(:,:,1) = reshape(X(:,:,1),[],M);
X(:,:,2) = reshape(X(:,:,2),[],M);
X(:,:,3) = reshape(X(:,:,3),[],M);
%C = reshape(C,M*N,1,3);
%E = ones(M*N,1,3);
%C = double(C);
%{
for k = 1:3
    E(1,1,k) = C(1,1,k);
    for i = 2:maxlength
        E(i,1,k) = bitxor(C(i,1,k),C(i-1,1,k));
    end
end
C = reshape(C,M,N,3);
E = reshape(E,M,N,3);
%}
u1 = 3.399; u2 = 3.4499; k1 = 0.21; u = 3.85; k2 = 0.15;
%Range:
%2.75<u1<= 3.45; 2.75<u2<=3.45, 0.15<k1<=0.21, 3.56<u<=4, 0.13<k2<=0.15;
% At decrypting end, logistic map equation are known.
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

m1=ones(M,N,1);
m2=ones(M,N,1);
m3=ones(M,N,1);
for i = 1:M
   for j = 1:N
    m1(i,j,1) = mod(x(i,j,1)*power(10,15),256);
    m2(i,j,1) = mod(y(i,j,1)*power(10,15),256);
    m3(i,j,1) = mod(z(i,j,1)*power(10,15),256);
   end
end
% e1,e2,e3 are secret keys, thus at decrypting end e1,e2,e3 are known.
% Hence a,b,c are also known.
S = ones(M,N,3);
for i = 1:M
    for j = 1:N
        if(a ~= m1(i,j))
            S(i,j,1) = abs(a - m1(i,j));
        else
            S(i,j,1) = a;
        end
    end
end
for i = 1:M
    for j = 1:N
        if(b ~= m2(i,j))
           S(i,j,2) = abs(b - m2(i,j));
        else
           S(i,j,2) = b;
        end
    end
end
for i = 1:M
    for j = 1:N
        if(c ~= m3(i,j))
           S(i,j,3) = abs(c - m3(i,j));
        else
           S(i,j,3) = c;
        end
    end
end
for i = 1:M
    for j = 1:N
    if (((i-1)*N)+j) ~= M*N
       t1 = floor(mod(x(i,j,1)*power(10,15),maxlength));
       t2 = floor(mod(y(i,j,1)*power(10,15),maxlength));
       t3 = floor(mod((x(i,j,1)+y(i,j,1)-(floor(x(i,j,1)+y(i,j,1))))*power(10,15),maxlength));
    else
        t1 = floor(mod(z(i,j,1)* power(10,15),maxlength));
        t2 = floor(mod((1-z(i,j,1))* power(10,15),maxlength));
        t3 = floor((t1 + t2 +c)/2);
    end
    S(i,j,1) = circshift(S(i,j,1),t1);
    S(i,j,2) = circshift(S(i,j,2),t2);
    S(i,j,3) = circshift(S(i,j,3),t3);
    end
end
X = uint8(X);
S = uint8(S);
C = bitxor(X,S);
temp4 = C(255,255,1);
temp5 = C(255,255,2);
temp6 = C(255,255,3);

%{
for i = 1:65
   C = Arnold_map_decrypt(C);
   %figure(1);image(X);title(i);
   %refresh;
   %pause(0.0001);
end
E = C;
figure, imshow(E), title('Final Encrypted RGB image')
%{
E = double(E);
H1 = hadamard(M);
H2 = hadamard(M);
H3 = hadamard(M);
%H1 = uint8(H1);
%H2 = uint8(H2);
%H3 = uint8(H3);
if(H1(:,:) == -1)
    D(:,:,1) = exp(((-E(:,:,1)/128) - H1)*log(256));
else
    D(:,:,1) = exp(((E(:,:,1)/128) - H1)*log(256));
end
if(H2(:,:) == -1)
    D(:,:,2) = exp(((-E(:,:,2)/128) - H2)*log(256));
else
    D(:,:,2) = exp(((E(:,:,2)/128) - H2)*log(256));
end
if(H3(:,:) == -1)
    D(:,:,3) = exp(((-E(:,:,3)/128) - H3)*log(256));
else
    D(:,:,3) = exp(((E(:,:,3)/128) - H3)*log(256));
end

D = uint8(D);
D = mat2gray(D);
figure, imshow(D), title('Recovered image1')
%figure, imshow(D(:,:,1)), title('Recovered image1')
%figure, imshow(D(:,:,2)), title('Recovered image2')
%figure, imshow(D(:,:,3)), title('Recovered image3')
%% 4. Analaysis
% Histogram
%figure,subplot(221),imshow(img(:,:,1),[]),subplot(222),imshow(CI,[])
%subplot(223),imhist(I),subplot(224),imhist(CI)
%}
%}
