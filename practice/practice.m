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
E(:,:,1) = (D(:,:,1)*H1(:,:,1))*S(:,:,1);
E(:,:,2) = (D(:,:,2)*H2(:,:,1))*S(:,:,2);
E(:,:,3) = (D(:,:,3)*H3(:,:,1))*S(:,:,3);
%E(:,:,1) = (S(:,:,1) + D(:,:,1))+H1;
%E(:,:,2) = (S(:,:,2) + D(:,:,2))+H2;
%E(:,:,3) = (S(:,:,3) + D(:,:,3))+H3;

%% Step-5:
%E = double(E);
E = reshape(E,maxlength,1,3);
C = ones(maxlength,1,3);
for k = 1:3
    C(1,1,k) = E(1,1,k);
    for i = 2:maxlength
        C(i,1,k) = E(i,1,k) + C(i-1,1,k);
    end
end
C = reshape(C,M,N,3);
C = mod(C(:,:,:),256);
C = uint8(C);
D = uint8(D);
C = mat2gray(C);

figure, imshow(D), title('Original input RGB image')
figure, imshow(C), title('Final Encrypted RGB image')
%figure, imshow(C(:,:,1)), title('Red channel')
%figure, imshow(C(:,:,2)), title('Green channel')
%figure, imshow(C(:,:,3)), title('Blue channel')
%subplot(3,1,1), imhist(C(:,:,1)), title('Histogram of Red component(Encrypted image)')
%subplot(3,1,2), imhist(C(:,:,2)), title('Histogram of Green component(Encrypted image)')
%subplot(3,1,3), imhist(C(:,:,3)), title('Histogram of Blue component(Encrypted image)')
%% 3. Decryption
%%{
maxlength = M*N;
divisor = M*N*255;
C(:,:,1) = reshape(C(:,:,1),[],M);
C(:,:,2) = reshape(C(:,:,2),[],M);
C(:,:,3) = reshape(C(:,:,3),[],M);
C = reshape(C,M*N,1,3);
E1 = ones(M*N,1,3);
C = double(C);
for k = 1:3
    E1(1,1,k) = C(1,1,k);
    for i = 2:maxlength
        E1(i,1,k) = mod(C(i,1,k),256) - C(i-1,1,k);
    end
end
C = reshape(C,M,N,3);
E = reshape(E,M,N,3);
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
H1 = hadamard(M);
H2 = hadamard(M);
H3 = hadamard(M);
D(:,:,1) = (E(:,:,1)*pinv(S(:,:,1)))*pinv(H1);
D(:,:,2) = (E(:,:,2)*pinv(S(:,:,2)))*pinv(H2);
D(:,:,3) = (E(:,:,3)*pinv(S(:,:,3)))*pinv(H3);
D = mat2gray(D);
figure, imshow(D), title('Recovered image1')
%figure, imshow(D(:,:,1)), title('Recovered image1')
%figure, imshow(D(:,:,2)), title('Recovered image2')
%figure, imshow(D(:,:,3)), title('Recovered image3')
%% 4. Analaysis
% Histogram
%figure,subplot(221),imshow(img(:,:,1),[]),subplot(222),imshow(CI,[])
%subplot(223),imhist(I),subplot(224),imhist(CI)
%%}